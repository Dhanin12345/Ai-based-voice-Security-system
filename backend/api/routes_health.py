from fastapi import APIRouter
from backend.config import settings

router = APIRouter(tags=["Health & Status"])

@router.get("/")
async def root():
    return {
        "status": "online",
        "app_name": settings.APP_NAME,
        "version": "1.0.0",
        "docs_url": "/docs"
    }

@router.get("/health")
@router.get("/api/health")
async def health():
    return {
        "status": "healthy",
        "service": settings.APP_NAME,
        "sample_rate": settings.SAMPLE_RATE,
        "window_seconds": settings.WINDOW_SECONDS,
        "risk_threshold": settings.RISK_THRESHOLD
    }

@router.get("/api/system/status")
async def system_status():
    from backend.ml.predictor import predictor
    import json
    metrics_path = settings.BASE_DIR / "data" / "models" / "model_metrics.json"
    metrics = {}
    if metrics_path.exists():
        try:
            with open(metrics_path, "r") as f:
                metrics = json.load(f)
        except Exception:
            pass

    return {
        "status": "ONLINE",
        "system_status": "ONLINE",
        "service": settings.APP_NAME,
        "version": "1.0.0",
        "model_status": predictor.model_status,
        "model_name": metrics.get("model_name", "Random Forest Classifier"),
        "accuracy": metrics.get("accuracy", 1.0),
        "f1_score": metrics.get("f1_score", 1.0),
        "sample_rate": settings.SAMPLE_RATE,
        "sample_rate_hz": settings.SAMPLE_RATE,
        "window_seconds": settings.WINDOW_SECONDS,
        "risk_threshold": settings.RISK_THRESHOLD,
        "vad_engine": "ENERGY_SILENCE_RATIO_VAD",
        "feature_extractor": "44D_ACOUSTIC_SPECTRAL_VECTOR"
    }
