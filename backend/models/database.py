from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, ForeignKey, Text
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from backend.config import settings

engine = create_engine(
    settings.DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in settings.DATABASE_URL else {}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class UserDB(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    username = Column(String(50), unique=True, nullable=False)
    email = Column(String(100), unique=True, nullable=False)
    role = Column(String(20), default="analyst")
    created_at = Column(DateTime, default=datetime.utcnow)

class DetectionHistoryDB(Base):
    __tablename__ = "analysis_history"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    source = Column(String(50), nullable=False, default="upload") # 'upload' or 'live_microphone'
    filename = Column(String(255), nullable=True)
    duration_seconds = Column(Float, nullable=False, default=0.0)
    sample_rate = Column(Integer, nullable=False, default=16000)
    classification = Column(String(50), nullable=False) # 'HUMAN', 'SYNTHETIC / AI-GENERATED', 'UNCERTAIN'
    confidence = Column(Float, nullable=False) # 0.0 to 1.0
    human_probability = Column(Float, nullable=False, default=0.0)
    synthetic_probability = Column(Float, nullable=False, default=0.0)
    risk_score = Column(Integer, nullable=False) # 0 to 100
    risk_level = Column(String(20), nullable=False) # 'low', 'moderate', 'high', 'very_high'
    processing_time_ms = Column(Float, nullable=False)
    recommendation = Column(Text, nullable=True)
    total_windows = Column(Integer, default=1)
    suspicious_windows = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)

    alerts = relationship("AlertDB", back_populates="detection", cascade="all, delete-orphan")
    windows = relationship("AnalysisWindowDB", back_populates="detection", cascade="all, delete-orphan")
    features = relationship("AudioFeaturesDB", back_populates="detection", uselist=False, cascade="all, delete-orphan")

class AudioFeaturesDB(Base):
    __tablename__ = "audio_features"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    analysis_id = Column(Integer, ForeignKey("analysis_history.id"), nullable=False, index=True)
    pitch_mean = Column(Float, default=0.0)
    pitch_std = Column(Float, default=0.0)
    rms_mean = Column(Float, default=0.0)
    zcr_mean = Column(Float, default=0.0)
    spectral_centroid = Column(Float, default=0.0)
    spectral_bandwidth = Column(Float, default=0.0)
    spectral_rolloff = Column(Float, default=0.0)
    harmonicity = Column(Float, default=0.0)
    speech_duration = Column(Float, default=0.0)
    silence_ratio = Column(Float, default=0.0)
    mfcc_data = Column(Text, nullable=True) # JSON-serialized list of MFCC coefficients

    detection = relationship("DetectionHistoryDB", back_populates="features")

class AnalysisWindowDB(Base):
    __tablename__ = "analysis_windows"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    analysis_id = Column(Integer, ForeignKey("analysis_history.id"), nullable=False)
    window_number = Column(Integer, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    human_probability = Column(Float, nullable=False)
    synthetic_probability = Column(Float, nullable=False)
    confidence = Column(Float, nullable=False)
    risk_score = Column(Integer, nullable=False)
    risk_level = Column(String(20), nullable=False)

    detection = relationship("DetectionHistoryDB", back_populates="windows")

class DatasetSampleDB(Base):
    __tablename__ = "dataset_samples"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    filename = Column(String(255), nullable=False)
    label = Column(String(50), nullable=False) # 'HUMAN' or 'SYNTHETIC'
    duration = Column(Float, default=0.0)
    features = Column(Text, nullable=True) # JSON-serialized feature vector
    created_at = Column(DateTime, default=datetime.utcnow)

class AlertDB(Base):
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    detection_id = Column(Integer, ForeignKey("analysis_history.id"), nullable=True)
    alert_type = Column(String(50), nullable=False, default="SUSPICIOUS_VOICE_CLONE")
    classification = Column(String(50), nullable=True, default="SYNTHETIC / AI-GENERATED")
    confidence = Column(Float, nullable=True, default=0.0)
    message = Column(Text, nullable=False)
    risk_score = Column(Integer, nullable=False)
    acknowledged = Column(Boolean, default=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)

    detection = relationship("DetectionHistoryDB", back_populates="alerts")

class SettingsDB(Base):
    __tablename__ = "settings"

    id = Column(Integer, primary_key=True, default=1)
    risk_threshold = Column(Float, default=70.0)
    window_seconds = Column(Float, default=2.0)
    sample_rate = Column(Integer, default=16000)
    enable_audio_alerts = Column(Boolean, default=True)
    auto_delete_audio = Column(Boolean, default=True)
    data_retention_days = Column(Integer, default=30)
    updated_at = Column(DateTime, default=datetime.utcnow)

class UnknownCallerProfileDB(Base):
    __tablename__ = "unknown_caller_profiles"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    caller_id = Column(String(100), default="Unknown", index=True)
    voice_embedding = Column(Text, nullable=False) # JSON-serialized list of floats
    repeat_count = Column(Integer, default=1)
    last_similarity_score = Column(Float, default=100.0) # percentage e.g. 94.2
    status = Column(String(50), default="MONITORING") # 'MONITORING', 'WARNING', 'CALL BLOCKED'
    is_blocked = Column(Boolean, default=False)
    block_reason = Column(Text, nullable=True)
    detected_requests = Column(Text, default="[]") # JSON list e.g. ["OTP_REQUEST", "MONEY_REQUEST"]
    suspicious_repeat_count = Column(Integer, default=0)
    last_call_timestamp = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)

    logs = relationship("UnknownCallerCallLogDB", back_populates="profile", cascade="all, delete-orphan")

class UnknownCallerCallLogDB(Base):
    __tablename__ = "unknown_caller_call_logs"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    profile_id = Column(Integer, ForeignKey("unknown_caller_profiles.id"), nullable=True)
    caller_id = Column(String(100), default="Unknown")
    call_timestamp = Column(DateTime, default=datetime.utcnow)
    similarity_score = Column(Float, default=0.0)
    repeat_count = Column(Integer, default=1)
    status = Column(String(50), default="MONITORING")
    decision_reason = Column(Text, nullable=True)
    filename = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    profile = relationship("UnknownCallerProfileDB", back_populates="logs")

class UnknownCallerVoiceRecordDB(Base):
    __tablename__ = "unknown_caller_voice_records"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    caller_identifier = Column(String(100), default="Unknown", index=True)
    caller_status = Column(String(20), default="UNKNOWN")  # 'KNOWN', 'UNKNOWN'
    voice_embedding_reference = Column(Text, nullable=True)  # JSON-serialized embedding
    call_timestamp = Column(DateTime, default=datetime.utcnow)
    repeat_count = Column(Integer, default=1)
    similarity_score = Column(Float, default=0.0)  # percentage
    status = Column(String(50), default="MONITORING")  # 'MONITORING', 'WARNING', 'BLOCKED', 'REVIEW', 'UNBLOCKED', 'ALLOW'
    block_status = Column(Boolean, default=False)
    block_reason = Column(Text, nullable=True)
    request_type = Column(String(100), default="NONE") # Section 10: 'OTP_REQUEST', 'MONEY_REQUEST', etc.
    transcript_summary = Column(Text, nullable=True)
    suspicious_score = Column(Integer, default=0)
    review_status = Column(String(50), default="PENDING")  # 'PENDING', 'REVIEWED'
    reviewed_by = Column(String(100), nullable=True)
    review_notes = Column(Text, nullable=True)
    filename = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class UnknownCallerSettingsDB(Base):
    __tablename__ = "unknown_caller_settings"

    id = Column(Integer, primary_key=True, default=1)
    repeat_warn_threshold = Column(Integer, default=3)
    repeat_block_threshold = Column(Integer, default=4)
    similarity_threshold = Column(Float, default=85.0) # Percentage 0-100
    suspicious_repeat_threshold = Column(Integer, default=3) # Section 8: default 3
    updated_at = Column(DateTime, default=datetime.utcnow)

def init_db():
    Base.metadata.create_all(bind=engine)
    # Ensure SQLite columns are migrated if tables already exist
    try:
        from sqlalchemy import text
        with engine.connect() as conn:
            res = conn.execute(text("PRAGMA table_info(analysis_history)")).fetchall()
            col_names = [r[1] for r in res]
            if "sample_rate" not in col_names and len(col_names) > 0:
                conn.execute(text("ALTER TABLE analysis_history ADD COLUMN sample_rate INTEGER DEFAULT 16000"))
            res_a = conn.execute(text("PRAGMA table_info(alerts)")).fetchall()
            a_cols = [r[1] for r in res_a]
            if "classification" not in a_cols and len(a_cols) > 0:
                conn.execute(text("ALTER TABLE alerts ADD COLUMN classification VARCHAR(50) DEFAULT 'SYNTHETIC / AI-GENERATED'"))
            if "confidence" not in a_cols and len(a_cols) > 0:
                conn.execute(text("ALTER TABLE alerts ADD COLUMN confidence FLOAT DEFAULT 0.0"))

            # Unknown Caller migrations
            prof_info = conn.execute(text("PRAGMA table_info(unknown_caller_profiles)")).fetchall()
            prof_cols = [r[1] for r in prof_info]
            if "detected_requests" not in prof_cols and len(prof_cols) > 0:
                conn.execute(text("ALTER TABLE unknown_caller_profiles ADD COLUMN detected_requests TEXT DEFAULT '[]'"))
            if "suspicious_repeat_count" not in prof_cols and len(prof_cols) > 0:
                conn.execute(text("ALTER TABLE unknown_caller_profiles ADD COLUMN suspicious_repeat_count INTEGER DEFAULT 0"))

            rec_info = conn.execute(text("PRAGMA table_info(unknown_caller_voice_records)")).fetchall()
            rec_cols = [r[1] for r in rec_info]
            if "request_type" not in rec_cols and len(rec_cols) > 0:
                conn.execute(text("ALTER TABLE unknown_caller_voice_records ADD COLUMN request_type VARCHAR(100) DEFAULT 'NONE'"))
            if "transcript_summary" not in rec_cols and len(rec_cols) > 0:
                conn.execute(text("ALTER TABLE unknown_caller_voice_records ADD COLUMN transcript_summary TEXT"))
            if "suspicious_score" not in rec_cols and len(rec_cols) > 0:
                conn.execute(text("ALTER TABLE unknown_caller_voice_records ADD COLUMN suspicious_score INTEGER DEFAULT 0"))

            sett_info = conn.execute(text("PRAGMA table_info(unknown_caller_settings)")).fetchall()
            sett_cols = [r[1] for r in sett_info]
            if "suspicious_repeat_threshold" not in sett_cols and len(sett_cols) > 0:
                conn.execute(text("ALTER TABLE unknown_caller_settings ADD COLUMN suspicious_repeat_threshold INTEGER DEFAULT 3"))

            conn.commit()
    except Exception:
        pass

    # Ensure default settings row exists
    db = SessionLocal()
    try:
        if not db.query(SettingsDB).filter(SettingsDB.id == 1).first():
            default_sett = SettingsDB(
                id=1,
                risk_threshold=70.0,
                window_seconds=2.0,
                sample_rate=16000,
                enable_audio_alerts=True,
                auto_delete_audio=True,
                data_retention_days=30
            )
            db.add(default_sett)
            db.commit()

        if not db.query(UnknownCallerSettingsDB).filter(UnknownCallerSettingsDB.id == 1).first():
            default_caller_sett = UnknownCallerSettingsDB(
                id=1,
                repeat_warn_threshold=3,
                repeat_block_threshold=4,
                similarity_threshold=85.0
            )
            db.add(default_caller_sett)
            db.commit()
    except Exception:
        db.rollback()
    finally:
        db.close()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
