from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class WindowAnalysisResult(BaseModel):
    window_index: int
    start_time_sec: float
    end_time_sec: float
    human_probability: float
    synthetic_probability: float
    confidence: float
    risk_score: int
    risk_level: str

class TemporalAnalysis(BaseModel):
    total_windows: int
    suspicious_windows: int
    average_synthetic_probability: float
    trend: str

class DetectionResponse(BaseModel):
    model_config = {"extra": "allow"}

    status: str = "success"
    analysis_id: int
    classification: str # 'HUMAN VOICE', 'ROBOTIC VOICE / AI VOICE'
    human_probability: float
    synthetic_probability: float
    human_confidence: Optional[float] = None
    ai_confidence: Optional[float] = None
    risk: Optional[str] = None # 'LOW', 'HIGH'
    color: Optional[str] = None # 'GREEN', 'RED'
    danger_alert: Optional[bool] = None
    clone_probability: float = 0.0
    confidence: float # 0.0 to 1.0
    risk_score: int # 0 to 100
    risk_level: str # 'low', 'moderate', 'high', 'very_high'
    impersonation_status: str = "SAFE" # 'SAFE', 'SUSPICIOUS', 'POSSIBLE IMPERSONATION', 'HIGH-RISK IMPERSONATION'
    prevention_action: str = "CONTINUE MONITORING"
    suspicious_windows: int = 0
    total_windows: int = 1
    processing_time_ms: float
    recommendation: str = ""
    duration_seconds: float = 0.0
    source: str = "upload"
    filename: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    model_status: str = "READY (TRAINED MODEL LOADED)"
    temporal_analysis: Optional[TemporalAnalysis] = None
    windows: Optional[List[WindowAnalysisResult]] = None
    alert_triggered: bool = False
    disclaimer: str = "AI voice cloning detection is probabilistic and should not be treated as absolute forensic proof."

class AlertResponse(BaseModel):
    id: int
    analysis_id: Optional[int]
    alert_type: str
    classification: Optional[str] = "SYNTHETIC / AI-GENERATED"
    confidence: Optional[float] = 0.0
    message: str
    risk_score: int
    acknowledged: bool
    timestamp: datetime
    created_at: datetime

class HistoryItemResponse(BaseModel):
    id: int
    analysis_id: int
    timestamp: datetime
    source: str
    filename: Optional[str]
    duration_seconds: float
    classification: str
    human_probability: float = 0.0
    synthetic_probability: float = 0.0
    confidence: float
    risk_score: int
    risk_level: str
    processing_time_ms: float
    recommendation: Optional[str]
    alert_count: int = 0
    status: str = "COMPLETED"

class HistoryListResponse(BaseModel):
    status: str = "success"
    total: int
    items: List[HistoryItemResponse]

class DashboardStatsResponse(BaseModel):
    total_analyses: int
    suspicious_detections: int
    active_alerts: int
    average_risk: int

class AnalyticsDataResponse(BaseModel):
    total_analyses: int
    synthetic_count: int
    human_count: int
    avg_risk_score: float
    detection_ratio: float
    weekly_labels: List[str]
    weekly_normal: List[int]
    weekly_suspicious: List[int]

class SettingsUpdateModel(BaseModel):
    risk_threshold: float = 70.0
    window_seconds: float = 2.0
    sample_rate: int = 16000
    enable_audio_alerts: bool = True
    auto_delete_audio: bool = True
    data_retention_days: int = 30
