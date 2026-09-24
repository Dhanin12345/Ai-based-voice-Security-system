from typing import Optional, List, Any
from datetime import datetime
from pydantic import BaseModel, Field

class CallerAnalysisResponse(BaseModel):
    success: bool = True
    caller_status: str = "UNKNOWN"        # 'KNOWN', 'UNKNOWN'
    caller_identifier: str = "Unknown"
    caller_id: str = "Unknown"             # alias for frontend compatibility
    voice_similarity: float = 100.0        # percentage 0.0 - 100.0
    repeat_count: int = 1
    repeat_calls: int = 1                  # alias for frontend compatibility
    status: str = "MONITORING"             # 'ALLOW', 'MONITORING', 'WARNING', 'BLOCKED', 'CALL BLOCKED', 'REVIEW'
    block_required: bool = False
    is_blocked: bool = False               # alias for frontend compatibility
    block_reason: Optional[str] = None
    decision_reason: Optional[str] = "First unknown call recorded. Initiating baseline voice monitoring."
    profile_id: Optional[int] = None
    event_id: Optional[int] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    disclaimer: str = "VOICE SIMILARITY DETECTED: Similar voice alone does not block; blocked only after configured repeat threshold."
    danger_alert: bool = False
    danger_alert_message: Optional[str] = None
    scam_threat_type: Optional[str] = None  # 'OTP_THEFT', 'MONEY_DEMAND', 'VOICE_IMPERSONATION'
    security_action: Optional[str] = "CONTINUE MONITORING"
    
    # Conversation & Intent Analysis Additions
    conversation_text: Optional[str] = None
    detected_intent: Optional[str] = "NONE"
    request_type: Optional[str] = "NONE"   # Section 3 & 10: 'OTP_REQUEST', 'MONEY_REQUEST', etc.
    transcript_summary: Optional[str] = None
    suspicious_score: Optional[int] = 0
    suspicious_repeat_count: int = 0
    suspicious_repeat_threshold: int = 3
    detected_requests: List[str] = Field(default_factory=list)
    security_status: Optional[str] = None   # alias for status
    risk_category: Optional[str] = "NORMAL CONVERSATION"
    conversation_risk_level: Optional[str] = "LOW RISK"
    conversation_risk_score: Optional[int] = 10
    recommendation: Optional[str] = None
    warning_title: Optional[str] = None
    warning_message: Optional[str] = None
    highlighted_spans: List[str] = Field(default_factory=list)

class ConversationAnalyzeRequest(BaseModel):
    conversation_text: Optional[str] = None
    caller_identifier: Optional[str] = "Unknown"
    caller_status: Optional[str] = "UNKNOWN"

class ConversationAnalysisResult(BaseModel):
    success: bool = True
    caller_status: str = "UNKNOWN"
    detected_intent: str = "NONE"
    risk_category: str = "NORMAL CONVERSATION"
    risk_level: str = "LOW RISK"
    risk_score: int = 10
    urgency_detected: bool = False
    is_safety_context: bool = False
    action: str = "MONITOR"
    warning_title: Optional[str] = None
    warning_message: Optional[str] = None
    recommendation: str
    highlighted_spans: List[str] = Field(default_factory=list)
    analyzed_text: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class CallerProfileResponse(BaseModel):
    id: int
    caller_id: str
    repeat_count: int
    last_similarity_score: float
    status: str
    is_blocked: bool
    block_reason: Optional[str] = None
    last_call_timestamp: datetime
    created_at: datetime
    scam_threat_type: Optional[str] = None

class SecurityEventLogResponse(BaseModel):
    id: int
    event_id: int
    timestamp: datetime
    caller_identifier: str
    caller_status: str
    repeat_count: int
    voice_similarity: float
    status: str
    security_status: Optional[str] = None
    request_type: Optional[str] = "NONE"
    transcript_summary: Optional[str] = None
    suspicious_score: Optional[int] = 0
    block_status: bool
    block_reason: Optional[str] = None
    review_status: str = "PENDING"
    reviewed_by: Optional[str] = None
    review_notes: Optional[str] = None
    filename: Optional[str] = None

class CallerCallLogResponse(BaseModel):
    id: int
    profile_id: Optional[int] = None
    caller_id: str
    call_timestamp: datetime
    similarity_score: float
    repeat_count: int
    status: str
    decision_reason: Optional[str] = None
    filename: Optional[str] = None
    danger_alert: bool = False
    scam_threat_type: Optional[str] = None

class CallerSettingsModel(BaseModel):
    voice_similarity_threshold: float = Field(default=85.0, ge=50.0, le=99.9)
    repeated_call_warn_threshold: int = Field(default=3, ge=1, le=20)
    repeated_call_block_threshold: int = Field(default=4, ge=1, le=50)
    suspicious_repeat_threshold: int = Field(default=3, ge=1, le=20)

    # Aliases
    similarity_threshold: Optional[float] = None
    repeat_warn_threshold: Optional[int] = None
    repeat_block_threshold: Optional[int] = None

    def model_post_init(self, __context: Any) -> None:
        if self.similarity_threshold is not None:
            self.voice_similarity_threshold = self.similarity_threshold
        else:
            self.similarity_threshold = self.voice_similarity_threshold

        if self.repeat_warn_threshold is not None:
            self.repeated_call_warn_threshold = self.repeat_warn_threshold
        else:
            self.repeat_warn_threshold = self.repeated_call_warn_threshold

        if self.repeat_block_threshold is not None:
            self.repeated_call_block_threshold = self.repeat_block_threshold
        else:
            self.repeat_block_threshold = self.repeated_call_block_threshold

class UnblockRequest(BaseModel):
    review_notes: Optional[str] = "Authorized security analyst review: voice unblocked"

class ReviewEventRequest(BaseModel):
    reviewed_by: str = "security_analyst"
    review_notes: Optional[str] = "Reviewed and validated by authorized security operator"
