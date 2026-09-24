from typing import Optional
from sqlalchemy.orm import Session
from backend.config import settings
from backend.models.database import AlertDB, DetectionHistoryDB
from backend.utils.logger import logger

class AlertService:
    def evaluate_and_trigger_alert(
        self,
        db: Session,
        risk_score: int,
        detection_id: Optional[int] = None,
        threshold: float = None,
        classification: str = "SYNTHETIC / AI-GENERATED",
        confidence: float = 0.0
    ) -> Optional[AlertDB]:
        """
        Evaluates risk score against threshold. If risk_score >= threshold, creates alert record in database.
        """
        threshold = threshold if threshold is not None else settings.RISK_THRESHOLD

        if risk_score >= threshold:
            message = (
                f"SECURITY ALERT: Potential synthetic or voice-cloned speech detected (Risk Score {risk_score}/100, Confidence {int(confidence*100)}%). "
                "Recommended: Perform independent caller verification."
            )
            try:
                alert = AlertDB(
                    detection_id=detection_id,
                    alert_type="SYNTHETIC_VOICE_DETECTION",
                    classification=classification,
                    confidence=confidence,
                    message=message,
                    risk_score=risk_score,
                    acknowledged=False
                )
                db.add(alert)
                db.commit()
                db.refresh(alert)
                logger.info(f"Security Alert #{alert.id} generated for Detection #{detection_id} (Risk Score: {risk_score})")
                return alert
            except Exception as e:
                logger.error(f"Failed to record alert in database: {e}")
                db.rollback()
                return None
        return None

alert_service = AlertService()
