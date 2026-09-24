import os
from pathlib import Path
from pydantic_settings import BaseSettings

_BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    APP_NAME: str = "AI Voice Security Monitor"
    DEBUG: bool = True
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    SECRET_KEY: str = "dev-secret-key-change-in-production"
    
    BASE_DIR: Path = _BASE_DIR
    DATABASE_URL: str = f"sqlite:///{_BASE_DIR}/data/database/voice_security.db"
    UPLOAD_DIR: Path = _BASE_DIR / "data" / "uploads"
    DB_DIR: Path = _BASE_DIR / "data" / "database"
    
    MAX_UPLOAD_SIZE_MB: int = 25
    SAMPLE_RATE: int = 16000
    WINDOW_SECONDS: float = 2.0
    RISK_THRESHOLD: float = 70.0
    
    CORS_ORIGINS: list[str] = ["*"]
    
    model_config = {
        "env_file": ".env",
        "extra": "ignore"
    }

settings = Settings()

# Ensure directories exist
settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
settings.DB_DIR.mkdir(parents=True, exist_ok=True)
