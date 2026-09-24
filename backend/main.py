from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse

import numpy as np
from backend.config import settings
from backend.models.database import init_db
from backend.api import routes_health, routes_detection, routes_history, routes_repeated_caller
from backend.utils.logger import logger

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing database tables...")
    init_db()
    logger.info(f"{settings.APP_NAME} started successfully on http://{settings.HOST}:{settings.PORT}")
    yield

app = FastAPI(
    title=settings.APP_NAME,
    description=(
        "AI-Powered Real-Time Detection and Prevention of Voice Cloning Impersonation Attacks. "
        "Processes live microphone streams and uploaded audio files to estimate synthetic speech probability "
        "and calculate real-time risk scores."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Enable CORS for web browsers and file:// origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_origin_regex=r".*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API Routers FIRST so API & WebSockets have routing priority
app.include_router(routes_health.router)
app.include_router(routes_detection.router)
app.include_router(routes_history.router)
app.include_router(routes_repeated_caller.router)

from backend.api.routes_detection import detect_uploaded_audio, websocket_detect
from fastapi import UploadFile, File, Depends, WebSocket
from sqlalchemy.orm import Session
from backend.models.database import get_db

@app.post("/api/analyze")
async def api_analyze_alias(file: UploadFile = File(...), db: Session = Depends(get_db)):
    return await detect_uploaded_audio(file=file, db=db)

@app.post("/api/audio/upload")
async def api_audio_upload_alias(file: UploadFile = File(...), db: Session = Depends(get_db)):
    return await detect_uploaded_audio(file=file, db=db)

@app.post("/api/audio/analyze")
async def api_audio_analyze_alias(file: UploadFile = File(...), db: Session = Depends(get_db)):
    return await detect_uploaded_audio(file=file, db=db)

@app.websocket("/ws/detect")
async def ws_detect_alias(websocket: WebSocket, db: Session = Depends(get_db)):
    await websocket_detect(websocket=websocket, db=db)

@app.websocket("/ws/monitor")
async def ws_monitor_alias(websocket: WebSocket, db: Session = Depends(get_db)):
    await websocket_detect(websocket=websocket, db=db)

@app.websocket("/ws/live-analysis")
async def ws_live_analysis_alias(websocket: WebSocket, db: Session = Depends(get_db)):
    await websocket_detect(websocket=websocket, db=db)

@app.post("/api/audio/start")
@app.post("/api/audio/start-live")
async def api_start_live():
    """
    Initializes a live audio stream monitoring session.
    """
    import uuid
    return {
        "status": "active",
        "session_id": f"sess_{uuid.uuid4().hex[:12]}",
        "websocket_url": "/ws/live-analysis",
        "sample_rate": settings.SAMPLE_RATE,
        "window_seconds": settings.WINDOW_SECONDS,
        "risk_threshold": settings.RISK_THRESHOLD,
        "message": "Live monitoring session initialized. Connect via WebSocket to stream raw 16-bit PCM audio chunks."
    }

@app.post("/api/audio/stop")
async def api_stop_session():
    return {
        "status": "stopped",
        "session": "live_monitoring",
        "message": "Microphone monitoring stream stopped."
    }



@app.get("/api/demo/{sample_type}")
async def api_demo_sample(sample_type: str, db: Session = Depends(get_db)):
    """
    Executes deep voice analysis on demo audio samples:
      - 'human': Genuine human voice sample with natural pitch dynamics & formants
      - 'synthetic': AI-generated synthetic voice with neural vocoder artifacts
      - 'cloned': Deepfake cloned voice impersonation attempt
    """
    from backend.services.audio_processor import audio_processor
    from backend.services.voice_analyzer import voice_analyzer
    from backend.models.database import DetectionHistoryDB, AudioFeaturesDB, AnalysisWindowDB
    from backend.models.detection_result import DetectionResponse, TemporalAnalysis
    import os, time

    sample_type = sample_type.lower()
    if sample_type == "human":
        file_path = settings.BASE_DIR / "dataset" / "human" / "human_001.wav"
        label_source = "Demo Genuine Human Voice"
    elif sample_type == "cloned":
        file_path = settings.BASE_DIR / "dataset" / "synthetic" / "synthetic_002.wav"
        label_source = "Demo Cloned Voice Impersonation"
    else: # synthetic
        file_path = settings.BASE_DIR / "dataset" / "synthetic" / "synthetic_001.wav"
        label_source = "Demo AI Synthetic Voice"

    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"Demo sample for '{sample_type}' not found. Run python ml/update_data.py to generate samples.")

    with open(file_path, "rb") as f:
        content = f.read()

    start_t = time.time()
    audio, sr = audio_processor.load_audio_file(content, filename=file_path.name)
    audio = audio_processor.preprocess_audio(audio, sr=sr)
    duration_sec = float(len(audio) / settings.SAMPLE_RATE)

    analysis = voice_analyzer.analyze_audio(audio, sr=settings.SAMPLE_RATE)
    proc_time_ms = round((time.time() - start_t) * 1000, 2)

    db_entry = DetectionHistoryDB(
        source=f"demo_{sample_type}",
        filename=f"{label_source} ({file_path.name})",
        duration_seconds=round(duration_sec, 2),
        classification=analysis["classification"],
        confidence=round(analysis["confidence"], 4),
        human_probability=round(analysis["human_probability"], 4),
        synthetic_probability=round(analysis["synthetic_probability"], 4),
        risk_score=analysis["risk_score"],
        risk_level=analysis["status"],
        processing_time_ms=proc_time_ms,
        recommendation=analysis["recommendation"],
        total_windows=1,
        suspicious_windows=1 if analysis["risk_score"] >= 60 else 0
    )
    db.add(db_entry)
    db.commit()
    db.refresh(db_entry)

    # Save AudioFeaturesDB
    import json
    f_dict_d = analysis.get("features_dict", {})
    mfcc_means_d = f_dict_d.get("mfcc_mean", [])
    mfcc_list_d = [float(x) for x in mfcc_means_d] if isinstance(mfcc_means_d, (list, np.ndarray)) else []
    feat_db = AudioFeaturesDB(
        analysis_id=db_entry.id,
        pitch_mean=float(f_dict_d.get("pitch_mean", 0.0)),
        pitch_std=float(f_dict_d.get("pitch_std", 0.0)),
        rms_mean=float(f_dict_d.get("rms_mean", f_dict_d.get("energy_mean", 0.0))),
        zcr_mean=float(f_dict_d.get("zcr_mean", 0.0)),
        spectral_centroid=float(f_dict_d.get("spectral_centroid_mean", 0.0)),
        spectral_bandwidth=float(f_dict_d.get("spectral_bandwidth_mean", 0.0)),
        spectral_rolloff=float(f_dict_d.get("spectral_rolloff_mean", 0.0)),
        harmonicity=float(f_dict_d.get("harmonicity", 0.5)),
        speech_duration=float(f_dict_d.get("speech_duration", duration_sec)),
        silence_ratio=float(f_dict_d.get("silence_ratio", 0.0)),
        mfcc_data=json.dumps(mfcc_list_d)
    )
    db.query(AudioFeaturesDB).filter(AudioFeaturesDB.analysis_id == db_entry.id).delete()
    db.add(feat_db)

    # Save window DB
    w_db = AnalysisWindowDB(
        analysis_id=db_entry.id,
        window_number=1,
        human_probability=analysis["human_probability"],
        synthetic_probability=analysis["synthetic_probability"],
        confidence=analysis["confidence"],
        risk_score=analysis["risk_score"],
        risk_level=analysis["status"].lower()
    )
    db.add(w_db)
    db.commit()

    from backend.services.alert_service import alert_service
    alert_service.evaluate_and_trigger_alert(
        db,
        risk_score=analysis["risk_score"],
        detection_id=db_entry.id,
        classification=analysis["classification"],
        confidence=analysis["confidence"]
    )

    from backend.api.routes_history import notify_new_detection
    notify_new_detection()

    audio_rel_url = f"/dataset/{'human' if sample_type == 'human' else 'synthetic'}/{file_path.name}"

    return {
        "status": "success",
        "sample_type": sample_type,
        "source": label_source,
        "filename": file_path.name,
        "audio_url": audio_rel_url,
        "analysis_id": db_entry.id,
        "classification": analysis["classification"],
        "voice_type": analysis["voice_type"],
        "human_probability": round(analysis["human_probability"], 4),
        "synthetic_probability": round(analysis["synthetic_probability"], 4),
        "clone_probability": round(analysis.get("clone_probability", 0.0), 4),
        "confidence": round(analysis["confidence"], 4),
        "risk_score": analysis["risk_score"],
        "risk_level": analysis["status"].lower(),
        "impersonation_status": analysis.get("impersonation_status", "SAFE"),
        "prevention_action": analysis.get("prevention_action", "CONTINUE MONITORING"),
        "speech_detected": analysis.get("speech_detected", True),
        "speech_duration": analysis.get("speech_duration", 0.0),
        "silence_ratio": analysis.get("silence_ratio", 0.0),
        "signal_quality": analysis.get("signal_quality", "GOOD"),
        "recommendation": analysis["recommendation"],
        "human_analysis": analysis.get("human_analysis", {}),
        "synthetic_analysis": analysis.get("synthetic_analysis", {}),
        "features_comparison": analysis.get("features_comparison", {}),
        "features_dict": {
            "pitch_mean": float(f_dict_d.get("pitch_mean", 0.0)),
            "pitch_std": float(f_dict_d.get("pitch_std", 0.0)),
            "rms_mean": float(f_dict_d.get("rms_mean", 0.0)),
            "zcr_mean": float(f_dict_d.get("zcr_mean", 0.0)),
            "spectral_centroid_mean": float(f_dict_d.get("spectral_centroid_mean", 0.0)),
            "spectral_bandwidth_mean": float(f_dict_d.get("spectral_bandwidth_mean", 0.0)),
            "spectral_rolloff_mean": float(f_dict_d.get("spectral_rolloff_mean", 0.0)),
            "harmonicity": float(f_dict_d.get("harmonicity", 0.5)),
            "spectral_flatness_mean": float(f_dict_d.get("spectral_flatness_mean", 0.0)),
            "mfcc_mean": mfcc_list_d[:13]
        },
        "processing_time_ms": proc_time_ms,
        "duration_seconds": round(duration_sec, 2),
        "model_status": analysis.get("model_status", "READY")
    }

# Serve Dataset Audio Files for Web Audio Player
@app.get("/dataset/{subpath:path}", include_in_schema=False)
async def serve_dataset(subpath: str):
    target = settings.BASE_DIR / "dataset" / subpath
    if target.is_file():
        return FileResponse(target)
    raise HTTPException(status_code=404, detail="Dataset audio file not found")

# Mount Frontend Static Directory for frontend files (index.html, dashboard.html, etc.)
frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
if frontend_dir.exists():
    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_frontend(full_path: str):
        target = frontend_dir / full_path
        if full_path and target.is_file():
            return FileResponse(target)
        # Default fallback to index.html
        return FileResponse(frontend_dir / "index.html")
else:
    logger.warning("Frontend directory not found. API mode only.")

# Global Exception Handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error on {request.url}: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "message": "An unexpected server error occurred. Please check system logs.",
            "detail": str(exc) if settings.DEBUG else None
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host=settings.HOST, port=settings.PORT, reload=settings.DEBUG)
