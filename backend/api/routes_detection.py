import io
import json
import time
from typing import List
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, WebSocket, WebSocketDisconnect
from sqlalchemy.orm import Session
from sqlalchemy import func
import numpy as np

from backend.config import settings
from backend.models.database import get_db, DetectionHistoryDB, AlertDB, AnalysisWindowDB, AudioFeaturesDB
from backend.models.detection_result import DetectionResponse, WindowAnalysisResult, TemporalAnalysis, DashboardStatsResponse
from backend.services.audio_processor import audio_processor
from backend.services.detector import voice_clone_detector
from backend.services.risk_scorer import risk_scorer
from backend.services.alert_service import alert_service
from backend.services.voice_analyzer import voice_analyzer
from backend.ml.model_loader import model_loader
from backend.utils.logger import logger

router = APIRouter(prefix="/api/detect", tags=["Voice Detection"])

ALLOWED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".mp4", ".mpeg", ".mpg", ".aac", ".opus", ".3gp", ".webm"}

@router.get("/stats", response_model=DashboardStatsResponse)
async def get_dashboard_stats(db: Session = Depends(get_db)):
    """
    Returns high-level SOC dashboard metrics (Total Analyses, Suspicious Detections, Active Alerts, Average Risk).
    """
    total_analyses = db.query(DetectionHistoryDB).count()
    suspicious_count = db.query(DetectionHistoryDB).filter(DetectionHistoryDB.classification == "possibly_synthetic").count()
    active_alerts = db.query(AlertDB).filter(AlertDB.acknowledged == False).count()
    
    avg_risk = db.query(func.avg(DetectionHistoryDB.risk_score)).scalar()
    avg_risk_int = int(round(avg_risk)) if avg_risk is not None else 0

    return DashboardStatsResponse(
        total_analyses=total_analyses,
        suspicious_detections=suspicious_count,
        active_alerts=active_alerts,
        average_risk=avg_risk_int
    )

@router.post("/upload", response_model=DetectionResponse)
async def detect_uploaded_audio(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Accepts uploaded audio file (.wav, .mp3, .m4a, .ogg, .flac), processes through analysis pipeline,
    computes risk score and temporal windows, stores metadata in database, and returns standardized analysis result.
    """
    start_time = time.time()

    filename = file.filename or "uploaded_audio.wav"
    ext = "." + filename.split(".")[-1].lower() if "." in filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{ext}'. Allowed formats: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    contents = await file.read()
    if len(contents) > settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail=f"File size exceeds maximum limit of {settings.MAX_UPLOAD_SIZE_MB} MB."
        )
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="AUDIO INPUT ERROR: Empty audio file provided.")

    try:
        audio, sr = audio_processor.load_audio_file(contents, filename=filename)
        audio = audio_processor.preprocess_audio(audio, sr=sr)

        if len(audio) == 0:
            raise HTTPException(status_code=400, detail="AUDIO INPUT ERROR: Audio file contains no usable speech signal.")

        duration_sec = float(len(audio) / settings.SAMPLE_RATE)
        windows_data = audio_processor.create_windows(audio, sr=settings.SAMPLE_RATE, window_seconds=settings.WINDOW_SECONDS)

        window_results: List[WindowAnalysisResult] = []
        synth_probs: List[float] = []
        human_probs: List[float] = []

        for idx, (w_audio, w_start, w_end) in enumerate(windows_data):
            pred = voice_clone_detector.predict(w_audio, sr=settings.SAMPLE_RATE)
            r_info = risk_scorer.calculate_risk(pred["synthetic_probability"], feature_dict=pred.get("features_dict"))
            
            w_res = WindowAnalysisResult(
                window_index=idx + 1,
                start_time_sec=round(w_start, 2),
                end_time_sec=round(w_end, 2),
                human_probability=round(pred["human_probability"], 4),
                synthetic_probability=round(pred["synthetic_probability"], 4),
                confidence=round(pred["confidence"], 4),
                risk_score=r_info["risk_score"],
                risk_level=r_info["risk_level"].lower().replace(" ", "_")
            )
            window_results.append(w_res)
            synth_probs.append(pred["synthetic_probability"])
            human_probs.append(pred["human_probability"])

        avg_synth_prob = float(np.mean(synth_probs)) if synth_probs else 0.5
        avg_human_prob = float(np.mean(human_probs)) if human_probs else 0.5
        confidence = float(np.max([avg_human_prob, avg_synth_prob]))

        overall_analysis = voice_analyzer.analyze_audio(audio, sr=settings.SAMPLE_RATE)
        temporal_eval = risk_scorer.analyze_temporal_windows([w.model_dump() for w in window_results])

        processing_time_ms = round((time.time() - start_time) * 1000, 2)
        risk_level_str = overall_analysis["status"].lower()

        db_entry = DetectionHistoryDB(
            source="upload",
            filename=filename,
            duration_seconds=round(duration_sec, 2),
            classification=overall_analysis["classification"],
            confidence=round(overall_analysis["confidence"], 4),
            human_probability=round(overall_analysis["human_probability"], 4),
            synthetic_probability=round(overall_analysis["synthetic_probability"], 4),
            risk_score=overall_analysis["risk_score"],
            risk_level=overall_analysis["status"],
            processing_time_ms=processing_time_ms,
            recommendation=overall_analysis["recommendation"],
            total_windows=temporal_eval["total_windows"],
            suspicious_windows=temporal_eval["suspicious_windows"]
        )
        db.add(db_entry)
        db.commit()
        db.refresh(db_entry)

        from backend.api.routes_history import notify_new_detection
        notify_new_detection()

        # Save individual window records
        for w in window_results:
            w_db = AnalysisWindowDB(
                analysis_id=db_entry.id,
                window_number=w.window_index,
                human_probability=w.human_probability,
                synthetic_probability=w.synthetic_probability,
                confidence=w.confidence,
                risk_score=w.risk_score,
                risk_level=w.risk_level
            )
            db.add(w_db)
        db.commit()

        # Save AudioFeatures record in database
        f_dict = overall_analysis.get("features_dict", {})
        mfcc_means = f_dict.get("mfcc_mean", [])
        mfcc_list = [float(x) for x in mfcc_means] if isinstance(mfcc_means, (list, np.ndarray)) else []
        features_record = AudioFeaturesDB(
            analysis_id=db_entry.id,
            pitch_mean=float(f_dict.get("pitch_mean", 0.0)),
            pitch_std=float(f_dict.get("pitch_std", 0.0)),
            rms_mean=float(f_dict.get("rms_mean", f_dict.get("energy_mean", 0.0))),
            zcr_mean=float(f_dict.get("zcr_mean", 0.0)),
            spectral_centroid=float(f_dict.get("spectral_centroid_mean", 0.0)),
            spectral_bandwidth=float(f_dict.get("spectral_bandwidth_mean", 0.0)),
            spectral_rolloff=float(f_dict.get("spectral_rolloff_mean", 0.0)),
            harmonicity=float(f_dict.get("harmonicity", 0.5)),
            speech_duration=float(f_dict.get("speech_duration", duration_sec)),
            silence_ratio=float(f_dict.get("silence_ratio", 0.0)),
            mfcc_data=json.dumps(mfcc_list)
        )
        db.add(features_record)
        db.commit()

        alert_obj = alert_service.evaluate_and_trigger_alert(
            db,
            risk_score=overall_analysis["risk_score"],
            detection_id=db_entry.id,
            classification=overall_analysis["classification"],
            confidence=overall_analysis["confidence"]
        )

        resp_dict = DetectionResponse(
            status="success",
            analysis_id=db_entry.id,
            classification=overall_analysis["classification"],
            confidence=round(overall_analysis["confidence"], 4),
            human_probability=round(overall_analysis["human_probability"], 4),
            synthetic_probability=round(overall_analysis["synthetic_probability"], 4),
            clone_probability=round(overall_analysis.get("clone_probability", 0.0), 4),
            risk_score=overall_analysis["risk_score"],
            risk_level=risk_level_str,
            impersonation_status=overall_analysis.get("impersonation_status", "SAFE"),
            prevention_action=overall_analysis.get("prevention_action", "CONTINUE MONITORING"),
            suspicious_windows=temporal_eval["suspicious_windows"],
            total_windows=temporal_eval["total_windows"],
            processing_time_ms=processing_time_ms,
            recommendation=overall_analysis["recommendation"],
            duration_seconds=round(duration_sec, 2),
            source="upload",
            filename=filename,
            timestamp=db_entry.timestamp,
            model_status=overall_analysis.get("model_status", model_loader.model_status),
            temporal_analysis=TemporalAnalysis(
                total_windows=temporal_eval["total_windows"],
                suspicious_windows=temporal_eval["suspicious_windows"],
                average_synthetic_probability=round(temporal_eval["average_synthetic_probability"], 4),
                trend=temporal_eval["trend"]
            ),
            windows=window_results,
            alert_triggered=alert_obj is not None
        ).model_dump()

        resp_dict["voice_type"] = overall_analysis["voice_type"]
        resp_dict["classification"] = overall_analysis["classification"]
        resp_dict["human_confidence"] = overall_analysis["human_confidence"]
        resp_dict["ai_confidence"] = overall_analysis["ai_confidence"]
        resp_dict["risk"] = overall_analysis["risk"]
        resp_dict["color"] = overall_analysis["color"]
        resp_dict["danger_alert"] = overall_analysis["danger_alert"]
        resp_dict["human_analysis"] = overall_analysis["human_analysis"]
        resp_dict["synthetic_analysis"] = overall_analysis["synthetic_analysis"]
        resp_dict["features_comparison"] = overall_analysis["features_comparison"]
        resp_dict["speech_detected"] = overall_analysis.get("speech_detected", True)
        resp_dict["speech_duration"] = overall_analysis.get("speech_duration", 0.0)
        resp_dict["silence_ratio"] = overall_analysis.get("silence_ratio", 0.0)
        resp_dict["signal_quality"] = overall_analysis.get("signal_quality", "GOOD")
        resp_dict["features_dict"] = {
            "pitch_mean": float(f_dict.get("pitch_mean", 0.0)),
            "pitch_std": float(f_dict.get("pitch_std", 0.0)),
            "rms_mean": float(f_dict.get("rms_mean", 0.0)),
            "zcr_mean": float(f_dict.get("zcr_mean", 0.0)),
            "spectral_centroid_mean": float(f_dict.get("spectral_centroid_mean", 0.0)),
            "spectral_bandwidth_mean": float(f_dict.get("spectral_bandwidth_mean", 0.0)),
            "spectral_rolloff_mean": float(f_dict.get("spectral_rolloff_mean", 0.0)),
            "harmonicity": float(f_dict.get("harmonicity", 0.5)),
            "spectral_flatness_mean": float(f_dict.get("spectral_flatness_mean", 0.0)),
            "mfcc_mean": mfcc_list[:13]
        }
        resp_dict["acoustic_evidence"] = overall_analysis["human_analysis"]["evidence"] if overall_analysis["voice_type"] == "HUMAN" else overall_analysis["synthetic_analysis"]["evidence"]

        return resp_dict

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing audio upload: {e}")
        raise HTTPException(status_code=400, detail=f"AUDIO INPUT ERROR: {str(e)}")

@router.post("/audio", response_model=DetectionResponse)
async def detect_raw_audio(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Alias endpoint for POST /api/detect/upload.
    """
    return await detect_uploaded_audio(file=file, db=db)

@router.post("/bulk")
async def process_batch_stream(
    files: List[UploadFile] = File(...),
    db: Session = Depends(get_db)
):
    """
    High-Speed Batch Stream Ingestion Endpoint.
    Processes multiple audio/video streams in batch mode using vectorized features & rule engine fusion.
    """
    results = []
    suspicious_total = 0
    
    for f in files:
        try:
            res = await detect_uploaded_audio(file=f, db=db)
            results.append(res)
            if res.risk_score >= settings.RISK_THRESHOLD:
                suspicious_total += 1
        except Exception as e:
            logger.warning(f"Batch item failed for {f.filename}: {e}")

    return {
        "status": "success",
        "records_processed": len(results),
        "anomalies_detected": suspicious_total,
        "items": results
    }

def sanitize_for_json(obj):
    """
    Recursively sanitizes data structures to ensure 100% JSON serializability,
    converting NumPy types (ndarray, float32, int64, bool_) into standard Python primitives.
    """
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [sanitize_for_json(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.floating, float)):
        return float(obj)
    elif isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj

@router.websocket("/ws/detect")
async def websocket_detect(websocket: WebSocket, db: Session = Depends(get_db)):
    """
    WebSocket endpoint for real-time live microphone stream detection.
    Streams back standardized JSON payload matching the SOC schema.
    """
    await websocket.accept()
    logger.info("WebSocket client connected for real-time voice monitoring.")

    audio_buffer = np.array([], dtype=np.float32)
    chunk_count = 0
    window_history = []

    try:
        while True:
            data = await websocket.receive_bytes()
            if not data:
                continue

            start_t = time.time()
            chunk_audio = audio_processor.pcm_bytes_to_float(data, sample_width=2)
            audio_buffer = np.concatenate([audio_buffer, chunk_audio])

            window_samples = int(settings.SAMPLE_RATE * settings.WINDOW_SECONDS)

            if len(audio_buffer) >= window_samples:
                analysis_window = audio_buffer[:window_samples]
                slide_samples = int(settings.SAMPLE_RATE * 0.5)
                audio_buffer = audio_buffer[window_samples - slide_samples:]

                # Voice Activity / Energy check (VAD) to prevent room silence from falsely triggering robotic AI alert
                peak_amp = float(np.max(np.abs(analysis_window)))
                rms_amp = float(np.sqrt(np.mean(analysis_window ** 2)))

                if peak_amp < 0.015 or rms_amp * 1000 < 2.0:
                    listening_payload = {
                        "status": "success",
                        "state": "listening",
                        "speech_detected": False,
                        "message": "Listening for voice...",
                        "impersonation_status": "MONITORING",
                        "classification": "LISTENING",
                        "color": "CYAN",
                        "risk": "LOW",
                        "risk_score": 0,
                        "danger_alert": False
                    }
                    await websocket.send_text(json.dumps(sanitize_for_json(listening_payload)))
                    continue

                processed_window = audio_processor.preprocess_audio(analysis_window, sr=settings.SAMPLE_RATE)
                if len(processed_window) > 0:
                    chunk_count += 1
                    analysis = voice_analyzer.analyze_audio(processed_window, sr=settings.SAMPLE_RATE)
                    
                    window_history.append({
                        "window_index": chunk_count,
                        "synthetic_probability": analysis["synthetic_probability"]
                    })
                    if len(window_history) > 10:
                        window_history.pop(0)

                    temp_eval = risk_scorer.analyze_temporal_windows(window_history)
                    proc_ms = round((time.time() - start_t) * 1000, 2)

                    f_dict_w = analysis.get("features_dict", {})
                    mfcc_means_w = f_dict_w.get("mfcc_mean", [])
                    mfcc_list_w = [float(x) for x in mfcc_means_w] if isinstance(mfcc_means_w, (list, np.ndarray)) else []

                    response_payload = {
                        "status": "success",
                        "analysis_id": chunk_count,
                        "voice_type": analysis["voice_type"],
                        "classification": analysis["classification"],
                        "human_confidence": analysis["human_confidence"],
                        "ai_confidence": analysis["ai_confidence"],
                        "risk": analysis["risk"],
                        "color": analysis["color"],
                        "danger_alert": analysis["danger_alert"],
                        "confidence": round(analysis["confidence"], 4),
                        "human_probability": round(analysis["human_probability"], 4),
                        "synthetic_probability": round(analysis["synthetic_probability"], 4),
                        "clone_probability": analysis.get("clone_probability", 0.0),
                        "risk_score": analysis["risk_score"],
                        "risk_level": analysis["status"].lower(),
                        "status_label": analysis["status"],
                        "impersonation_status": analysis.get("impersonation_status", "SAFE"),
                        "prevention_action": analysis.get("prevention_action", "CONTINUE MONITORING"),
                        "speech_detected": analysis.get("speech_detected", True),
                        "speech_duration": analysis.get("speech_duration", 0.0),
                        "silence_ratio": analysis.get("silence_ratio", 0.0),
                        "signal_quality": analysis.get("signal_quality", "GOOD"),
                        "suspicious_windows": temp_eval["suspicious_windows"],
                        "total_windows": temp_eval["total_windows"],
                        "processing_time_ms": proc_ms,
                        "recommendation": analysis["recommendation"],
                        "human_analysis": analysis["human_analysis"],
                        "synthetic_analysis": analysis["synthetic_analysis"],
                        "features_comparison": analysis.get("features_comparison", {}),
                        "features_dict": {
                            "pitch_mean": float(f_dict_w.get("pitch_mean", 0.0)),
                            "pitch_std": float(f_dict_w.get("pitch_std", 0.0)),
                            "rms_mean": float(f_dict_w.get("rms_mean", 0.0)),
                            "zcr_mean": float(f_dict_w.get("zcr_mean", 0.0)),
                            "spectral_centroid_mean": float(f_dict_w.get("spectral_centroid_mean", 0.0)),
                            "spectral_bandwidth_mean": float(f_dict_w.get("spectral_bandwidth_mean", 0.0)),
                            "spectral_rolloff_mean": float(f_dict_w.get("spectral_rolloff_mean", 0.0)),
                            "harmonicity": float(f_dict_w.get("harmonicity", 0.5)),
                            "spectral_flatness_mean": float(f_dict_w.get("spectral_flatness_mean", 0.0)),
                            "mfcc_mean": mfcc_list_w[:13]
                        },
                        "model_status": analysis.get("model_status", model_loader.model_status),
                        "trend": temp_eval["trend"],
                        "alert_triggered": bool(analysis["risk_score"] >= settings.RISK_THRESHOLD)
                    }

                    clean_payload = sanitize_for_json(response_payload)
                    await websocket.send_text(json.dumps(clean_payload))

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected.")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            err_payload = sanitize_for_json({"status": "error", "error": f"AUDIO INPUT ERROR: {str(e)}"})
            await websocket.send_text(json.dumps(err_payload))
        except Exception:
            pass

# Section 20 Specification Endpoints
@router.post("/audio/upload")
async def audio_upload_alias(file: UploadFile = File(...), db: Session = Depends(get_db)):
    return await detect_uploaded_audio(file=file, db=db)

@router.post("/audio/analyze")
async def audio_analyze_alias(file: UploadFile = File(...), db: Session = Depends(get_db)):
    return await detect_uploaded_audio(file=file, db=db)

@router.post("/audio/start")
async def audio_start_session():
    return {
        "status": "active",
        "session": "live_monitoring",
        "sample_rate": settings.SAMPLE_RATE,
        "channels": 1,
        "message": "Microphone monitoring stream initialized."
    }

@router.post("/audio/stop")
async def audio_stop_session():
    return {
        "status": "stopped",
        "session": "live_monitoring",
        "message": "Microphone monitoring stream stopped."
    }
