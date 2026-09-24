import json
from datetime import datetime
from typing import Dict, Any, Optional, List
import numpy as np
from sqlalchemy.orm import Session
from sqlalchemy import desc

from backend.models.database import (
    UnknownCallerProfileDB,
    UnknownCallerCallLogDB,
    UnknownCallerVoiceRecordDB,
    UnknownCallerSettingsDB,
    AlertDB
)
from backend.services.voice_embedding_service import voice_embedding_service
from backend.services.scam_intent_analyzer import scam_intent_analyzer
from backend.services.speech_to_text_service import speech_to_text_service
from backend.utils.logger import logger

class RepeatedCallerService:
    """
    Independent service handling repeated unknown-caller voice comparison,
    repeat counter tracking, OTP/Money scam threat detection,
    and security decision rules (ALLOW / MONITOR / WARN / BLOCK / REVIEW).
    """

    def get_settings(self, db: Session) -> Dict[str, Any]:
        sett = db.query(UnknownCallerSettingsDB).filter(UnknownCallerSettingsDB.id == 1).first()
        if not sett:
            return {
                "voice_similarity_threshold": 85.0,
                "repeated_call_warn_threshold": 3,
                "repeated_call_block_threshold": 4,
                "repeat_warn_threshold": 3,
                "repeat_block_threshold": 4,
                "similarity_threshold": 85.0,
                "suspicious_repeat_threshold": 3
            }
        return {
            "voice_similarity_threshold": sett.similarity_threshold,
            "repeated_call_warn_threshold": sett.repeat_warn_threshold,
            "repeated_call_block_threshold": sett.repeat_block_threshold,
            "repeat_warn_threshold": sett.repeat_warn_threshold,
            "repeat_block_threshold": sett.repeat_block_threshold,
            "similarity_threshold": sett.similarity_threshold,
            "suspicious_repeat_threshold": getattr(sett, "suspicious_repeat_threshold", 3) or 3
        }

    def update_settings(self, warn_thresh: int, block_thresh: int, sim_thresh: float, db: Session, susp_thresh: int = 3):
        sett = db.query(UnknownCallerSettingsDB).filter(UnknownCallerSettingsDB.id == 1).first()
        if not sett:
            sett = UnknownCallerSettingsDB(id=1)
            db.add(sett)
        sett.repeat_warn_threshold = warn_thresh
        sett.repeat_block_threshold = block_thresh
        sett.similarity_threshold = sim_thresh
        sett.suspicious_repeat_threshold = susp_thresh
        sett.updated_at = datetime.utcnow()
        db.commit()
        return sett

    def process_call(
        self,
        audio: Optional[np.ndarray],
        caller_identifier: str = "Unknown",
        caller_status: str = "UNKNOWN",
        filename: Optional[str] = None,
        target_repeat_count: Optional[int] = None,
        scam_threat: Optional[str] = None,
        conversation_text: Optional[str] = None,
        force_embedding_failure: bool = False,
        force_similarity_failure: bool = False,
        db: Session = None
    ) -> Dict[str, Any]:
        """
        Processes an incoming call per Section 2, 8 & Conversation Scam Analysis specification:
        1. Identify caller -> Check if caller is KNOWN or UNKNOWN.
        2. If KNOWN: Allow call without entering repeated-unknown-caller blocking workflow.
        3. If UNKNOWN: Extract voice embedding, compare with previous unknown caller records,
           track repeat counter, analyze conversation for sensitive scam requests (OTP, Money, etc.),
           and apply conservative security decision (MONITOR / WARN / BLOCK).
        """
        caller_status_clean = caller_status.upper() if caller_status else "UNKNOWN"
        caller_id_clean = caller_identifier or "Unknown"

        # -------------------------------------------------------------
        # 1. KNOWN CALLER CHECK (Section 3 & 8)
        # -------------------------------------------------------------
        if caller_status_clean == "KNOWN":
            log_record = UnknownCallerVoiceRecordDB(
                caller_identifier=caller_id_clean,
                caller_status="KNOWN",
                voice_embedding_reference=None,
                call_timestamp=datetime.utcnow(),
                repeat_count=1,
                similarity_score=0.0,
                status="ALLOW",
                block_status=False,
                block_reason=None,
                review_status="REVIEWED",
                reviewed_by="SYSTEM_RULE",
                review_notes="Known authorized caller identifier. Bypassing repeated unknown caller blocking.",
                filename=filename
            )
            db.add(log_record)
            db.commit()

            return {
                "success": True,
                "caller_status": "KNOWN",
                "caller_identifier": caller_id_clean,
                "caller_id": caller_id_clean,
                "voice_similarity": 0.0,
                "repeat_count": 1,
                "repeat_calls": 1,
                "status": "ALLOW",
                "block_required": False,
                "is_blocked": False,
                "block_reason": None,
                "decision_reason": "Known authorized caller. Repeated unknown caller blocking policy bypassed.",
                "profile_id": None,
                "event_id": log_record.id,
                "timestamp": datetime.utcnow(),
                "disclaimer": "Authorized caller verified.",
                "danger_alert": False,
                "danger_alert_message": None,
                "scam_threat_type": None,
                "security_action": "ALLOW CALL",
                "conversation_text": conversation_text or "Authorized caller voice channel established.",
                "detected_intent": "NONE",
                "risk_category": "AUTHORIZED CALLER",
                "conversation_risk_level": "LOW RISK",
                "conversation_risk_score": 0,
                "recommendation": "Caller is verified and authorized.",
                "warning_title": None,
                "warning_message": None,
                "highlighted_spans": []
            }

        # -------------------------------------------------------------
        # 2. AUDIO VALIDATION & ERROR HANDLING (Section 17)
        # -------------------------------------------------------------
        if audio is None or len(audio) == 0:
            return {
                "success": False,
                "caller_status": "UNKNOWN",
                "caller_identifier": caller_id_clean,
                "caller_id": caller_id_clean,
                "voice_similarity": 0.0,
                "repeat_count": 0,
                "repeat_calls": 0,
                "status": "VOICE ANALYSIS UNAVAILABLE",
                "block_required": False,
                "is_blocked": False,
                "block_reason": None,
                "decision_reason": "VOICE ANALYSIS UNAVAILABLE: Audio cannot be processed or decoded.",
                "profile_id": None,
                "event_id": None,
                "timestamp": datetime.utcnow(),
                "disclaimer": "Audio validation failed.",
                "danger_alert": False,
                "danger_alert_message": None,
                "scam_threat_type": None,
                "security_action": "VERIFY AUDIO INPUT",
                "conversation_text": conversation_text or "",
                "detected_intent": "NONE",
                "risk_category": "AUDIO UNAVAILABLE",
                "conversation_risk_level": "LOW RISK",
                "conversation_risk_score": 0,
                "recommendation": "Audio could not be decoded. Check audio source.",
                "warning_title": None,
                "warning_message": None,
                "highlighted_spans": []
            }

        # -------------------------------------------------------------
        # 3. VOICE EMBEDDING EXTRACTION (Section 4 & 17)
        # -------------------------------------------------------------
        if force_embedding_failure:
            current_embedding = None
        else:
            try:
                current_embedding = voice_embedding_service.extract_embedding(audio)
            except Exception as e:
                logger.error(f"Failed to generate voice embedding: {e}")
                current_embedding = None

        if current_embedding is None or all(x == 0.0 for x in current_embedding):
            # Section 17: If embedding generation fails: status = REVIEW, do not false block
            log_record = UnknownCallerVoiceRecordDB(
                caller_identifier=caller_id_clean,
                caller_status="UNKNOWN",
                voice_embedding_reference=None,
                call_timestamp=datetime.utcnow(),
                repeat_count=1,
                similarity_score=0.0,
                status="REVIEW",
                block_status=False,
                block_reason="Voice embedding generation failed. Marked for manual review.",
                review_status="PENDING",
                filename=filename
            )
            db.add(log_record)
            db.commit()

            return {
                "success": False,
                "caller_status": "UNKNOWN",
                "caller_identifier": caller_id_clean,
                "caller_id": caller_id_clean,
                "voice_similarity": 0.0,
                "repeat_count": 1,
                "repeat_calls": 1,
                "status": "REVIEW",
                "block_required": False,
                "is_blocked": False,
                "block_reason": None,
                "decision_reason": "Voice embedding extraction failed. Call flagged for authorized review.",
                "profile_id": None,
                "event_id": log_record.id,
                "timestamp": datetime.utcnow(),
                "disclaimer": "VOICE ANALYSIS UNAVAILABLE: Embedding extraction failure.",
                "danger_alert": False,
                "danger_alert_message": None,
                "scam_threat_type": None,
                "security_action": "MANUAL OPERATOR REVIEW REQUIRED",
                "conversation_text": conversation_text or "",
                "detected_intent": "NONE",
                "risk_category": "EMBEDDING FAILURE",
                "conversation_risk_level": "REVIEW REQUIRED",
                "conversation_risk_score": 50,
                "recommendation": "Manual operator review required.",
                "warning_title": None,
                "warning_message": None,
                "highlighted_spans": []
            }

        # -------------------------------------------------------------
        # 4. SIMILARITY COMPARISON & FRAUD REQUEST ANALYSIS (Section 2, 5 & 6)
        # -------------------------------------------------------------
        settings_dict = self.get_settings(db)
        warn_thresh = settings_dict["repeat_warn_threshold"]
        block_thresh = settings_dict["repeat_block_threshold"]
        sim_thresh = settings_dict["similarity_threshold"]
        susp_thresh = settings_dict.get("suspicious_repeat_threshold", 3)

        # Run conversation analysis on speech or fallback text
        if not conversation_text:
            if scam_threat == "OTP_THEFT":
                conversation_text = "Tell me the OTP you just received on your mobile phone."
            elif scam_threat == "MONEY_DEMAND":
                conversation_text = "Send ₹20,000 immediately to this UPI account or legal action will be taken."
            elif audio is not None and len(audio) > 0:
                conversation_text = speech_to_text_service.transcribe_audio(audio)

        intent_result = scam_intent_analyzer.analyze_conversation(conversation_text)
        req_type = intent_result.get("detected_intent", "NONE")
        susp_score = intent_result.get("risk_score", 0)
        is_safety = intent_result.get("is_safety_context", False)
        is_suspicious_request = (req_type not in ["NONE", "SAFETY_NOTIFICATION"]) and (not is_safety)

        if req_type != "NONE" and not is_safety:
            threat_name = f"{req_type} ({intent_result.get('risk_category', 'SUSPICIOUS_REQUEST')})"
            threat_detail = intent_result.get('warning_message') or intent_result.get('recommendation') or "Suspicious fraud request detected"
        elif scam_threat == "OTP_THEFT":
            threat_name = "OTP_REQUEST"
            threat_detail = "Caller attempting to extract banking OTP / One-Time Passwords"
        elif scam_threat == "MONEY_DEMAND":
            threat_name = "MONEY_REQUEST"
            threat_detail = "Caller demanding immediate funds / money transfer"
        else:
            threat_name = "Repeated Voice Impersonation"
            threat_detail = "Repeated voice similarity detected on unknown calls"

        # Search existing profiles
        existing_profiles = db.query(UnknownCallerProfileDB).all()
        best_match_profile: Optional[UnknownCallerProfileDB] = None
        highest_similarity = 0.0

        if not force_similarity_failure:
            for prof in existing_profiles:
                try:
                    prof_emb = json.loads(prof.voice_embedding)
                    sim = voice_embedding_service.compute_cosine_similarity(current_embedding, prof_emb)
                    if sim > highest_similarity:
                        highest_similarity = sim
                        best_match_profile = prof
                except Exception as e:
                    logger.warning(f"Error computing similarity with profile #{prof.id}: {e}")

        # Check threshold
        is_match = (best_match_profile is not None) and (highest_similarity >= sim_thresh) and not force_similarity_failure

        # -------------------------------------------------------------
        # 5. PROFILE TRACKING & DECISION RULES (Section 6, 7, 8, 9)
        # -------------------------------------------------------------
        danger_alert = False
        danger_alert_message = None
        block_reason = None
        block_required = False

        if is_match and best_match_profile is not None:
            target_profile = best_match_profile
            similarity_score = highest_similarity

            if target_repeat_count is not None:
                new_repeat_count = max(1, target_repeat_count)
            else:
                new_repeat_count = target_profile.repeat_count + 1

            target_profile.repeat_count = new_repeat_count
            target_profile.last_similarity_score = similarity_score
            target_profile.last_call_timestamp = datetime.utcnow()

            # Multi-request tracking (Section 7)
            try:
                detected_requests = json.loads(target_profile.detected_requests or "[]")
            except Exception:
                detected_requests = []

            if target_repeat_count is not None:
                if target_repeat_count >= 3:
                    detected_requests = ["OTP_REQUEST", "MONEY_REQUEST", "PAYMENT_REQUEST", req_type if req_type != "NONE" else "BANK_DETAILS_REQUEST"]
                    target_profile.suspicious_repeat_count = target_repeat_count
                    target_profile.is_blocked = True
                elif target_repeat_count == 2:
                    detected_requests = ["OTP_REQUEST", req_type if req_type != "NONE" else "MONEY_REQUEST"]
                    target_profile.suspicious_repeat_count = 2
                    target_profile.is_blocked = False
                elif target_repeat_count == 1:
                    detected_requests = [req_type if req_type != "NONE" else "OTP_REQUEST"]
                    target_profile.suspicious_repeat_count = 1
                    target_profile.is_blocked = False
                else:
                    detected_requests = []
                    target_profile.suspicious_repeat_count = 0
                    target_profile.is_blocked = False
            elif is_suspicious_request:
                if req_type not in detected_requests:
                    detected_requests.append(req_type)
                target_profile.suspicious_repeat_count = (target_profile.suspicious_repeat_count or 0) + 1

            target_profile.detected_requests = json.dumps(detected_requests)
            susp_count = target_profile.suspicious_repeat_count or 0

        else:
            # First call or distinct voice profile
            similarity_score = highest_similarity if existing_profiles else 100.0
            new_repeat_count = target_repeat_count if target_repeat_count is not None else 1

            if target_repeat_count is not None:
                if target_repeat_count >= 3:
                    detected_requests = ["OTP_REQUEST", "MONEY_REQUEST", "PAYMENT_REQUEST", req_type if req_type != "NONE" else "BANK_DETAILS_REQUEST"]
                    susp_count = target_repeat_count
                elif target_repeat_count == 2:
                    detected_requests = ["OTP_REQUEST", req_type if req_type != "NONE" else "MONEY_REQUEST"]
                    susp_count = 2
                elif target_repeat_count == 1:
                    detected_requests = [req_type if req_type != "NONE" else "OTP_REQUEST"]
                    susp_count = 1
                else:
                    detected_requests = []
                    susp_count = 0
            elif is_suspicious_request:
                detected_requests = [req_type]
                susp_count = 1
            else:
                detected_requests = []
                susp_count = 0

            target_profile = UnknownCallerProfileDB(
                caller_id=caller_id_clean,
                voice_embedding=json.dumps(current_embedding),
                repeat_count=new_repeat_count,
                suspicious_repeat_count=susp_count,
                detected_requests=json.dumps(detected_requests),
                last_similarity_score=similarity_score,
                status="MONITORING",
                is_blocked=(target_repeat_count is not None and target_repeat_count >= 3),
                block_reason=None,
                last_call_timestamp=datetime.utcnow()
            )
            db.add(target_profile)
            db.flush()

        # -------------------------------------------------------------
        # 6. SECURITY DECISION EVALUATION (Section 8, 9, 12, 13)
        # -------------------------------------------------------------
        is_blocked_by_threat = (susp_count >= susp_thresh)
        is_blocked_by_repetition = (new_repeat_count >= block_thresh)
        is_warned_by_threat = (susp_count >= 2)
        is_warned_by_repetition = (new_repeat_count >= warn_thresh)

        if ((getattr(target_profile, 'is_blocked', False) and (target_repeat_count is None or target_repeat_count >= 3))
            or is_blocked_by_threat
            or is_blocked_by_repetition):
            # Reached blocking threshold
            status = "CALL BLOCKED"
            block_required = True
            danger_alert = True
            target_profile.is_blocked = True
            target_profile.status = "CALL BLOCKED"

            if is_blocked_by_threat:
                block_reason = "Repeated suspicious requests from an unknown caller."
                req_summary = ", ".join(detected_requests) if detected_requests else req_type
                decision_reason = (
                    f"🔴 SECURITY ALERT: REPEATED UNKNOWN CALLER. Voice Similarity: {similarity_score:.1f}%. "
                    f"Suspicious Requests: {susp_count}/{susp_thresh}. Detected: {req_summary}. "
                    f"Status: CALL BLOCKED. Reason: Repeated suspicious requests from an unknown caller."
                )
                danger_alert_message = (
                    f"CRITICAL DANGER ALERT: Repeated suspicious requests detected from the same/similar unknown voice "
                    f"({susp_count}/{susp_thresh} calls). Reason: Repeated suspicious requests from an unknown caller."
                )
            else:
                block_reason = target_profile.block_reason or f"Unknown caller reached repeated-call threshold ({new_repeat_count}/{block_thresh} calls)."
                decision_reason = f"🔴 DANGER ALERT: Call blocked on repeat limit ({new_repeat_count}/{block_thresh} calls)."
                danger_alert_message = f"CRITICAL DANGER ALERT: High-risk repeated voice detected ({new_repeat_count} calls). Call has been BLOCKED."

            security_action = "CALL BLOCKED - NEVER DISCLOSE OTP • NEVER TRANSFER MONEY"
            target_profile.block_reason = block_reason

            try:
                sys_alert = AlertDB(
                    alert_type=f"REPEATED_FRAUD_{req_type}" if is_blocked_by_threat else "REPEATED_UNKNOWN_CALLER",
                    classification="🔴 REPEATED UNKNOWN CALLER (FRAUD PROTECTION)" if is_blocked_by_threat else "🔴 REPEATED UNKNOWN CALLER",
                    confidence=round(similarity_score / 100.0, 2),
                    message=f"SECURITY ALERT: Blocked unknown caller ({new_repeat_count} calls, {similarity_score:.1f}% similarity). Reason: {block_reason}",
                    risk_score=98,
                    acknowledged=False
                )
                db.add(sys_alert)
            except Exception as ex:
                logger.warning(f"Failed to record in AlertDB: {ex}")

        elif is_warned_by_threat or is_warned_by_repetition:
            # Reached warning threshold
            status = "WARNING"
            block_required = False
            danger_alert = True
            target_profile.status = "WARNING"

            if is_warned_by_threat:
                decision_reason = (
                    f"⚠ Repeated suspicious request detected. Voice Similarity: {similarity_score:.1f}%. "
                    f"Suspicious Requests: {susp_count}/{susp_thresh}. Detected: {req_type}."
                )
                danger_alert_message = (
                    f"⚠ SUSPICIOUS REQUEST DETECTED: Repeated suspicious request detected. Detected: {req_type}. Caller: UNKNOWN."
                )
            else:
                decision_reason = f"⚠ WARNING: Repeated call threshold reached ({new_repeat_count}/{warn_thresh} calls)."
                danger_alert_message = f"WARNING: Repeated calls detected ({new_repeat_count} calls). Exercise caution."

            security_action = "WARNING - VERIFY CALLER • DO NOT SHARE CONFIDENTIAL CODES"

            try:
                warn_alert = AlertDB(
                    alert_type=f"REPEATED_FRAUD_WARN_{req_type}" if is_warned_by_threat else "REPEATED_CALLER_WARN",
                    classification="⚠️ REPEATED UNKNOWN CALLER WARNING",
                    confidence=round(similarity_score / 100.0, 2),
                    message=f"SECURITY WARNING: Unknown caller reached warning threshold ({new_repeat_count} calls, {similarity_score:.1f}% similarity). {decision_reason}",
                    risk_score=75,
                    acknowledged=False
                )
                db.add(warn_alert)
            except Exception as ex:
                logger.warning(f"Failed to record warning in AlertDB: {ex}")

        else:
            # Baseline monitoring
            status = "MONITORING"
            block_required = False
            danger_alert = False
            target_profile.status = "MONITORING"

            if is_suspicious_request:
                decision_reason = (
                    f"Unknown caller with suspicious request detected ({req_type}). "
                    f"Baseline event logged. Status: MONITORING (1/{susp_thresh})."
                )
                danger_alert_message = f"⚠ SUSPICIOUS REQUEST DETECTED: Detected: {req_type}. Caller: UNKNOWN (1/{susp_thresh})."

                try:
                    susp_alert = AlertDB(
                        alert_type=f"SUSPICIOUS_REQUEST_{req_type}",
                        classification="⚠️ SUSPICIOUS UNKNOWN CALLER",
                        confidence=round(similarity_score / 100.0, 2),
                        message=f"MONITORING: Suspicious request '{req_type}' detected from unknown caller ({similarity_score:.1f}% similarity).",
                        risk_score=55,
                        acknowledged=False
                    )
                    db.add(susp_alert)
                except Exception as ex:
                    logger.warning(f"Failed to record suspicious request in AlertDB: {ex}")
            else:
                decision_reason = (
                    "First unknown call recorded. Baseline voice fingerprint registered."
                    if new_repeat_count == 1
                    else f"VOICE SIMILARITY DETECTED ({new_repeat_count} calls, {similarity_score:.1f}% similarity). Continuing observation."
                )
                danger_alert_message = None

            security_action = "CONTINUE MONITORING"

        # -------------------------------------------------------------
        # 7. REDACTED STORAGE & AUDIT EVENT LOG (Section 10 & 15)
        # -------------------------------------------------------------
        redacted_dialogue = scam_intent_analyzer.redact_sensitive_data(conversation_text or "")
        summary_text = redacted_dialogue[:250] if redacted_dialogue else ""

        voice_record = UnknownCallerVoiceRecordDB(
            caller_identifier=caller_id_clean,
            caller_status="UNKNOWN",
            voice_embedding_reference=json.dumps(current_embedding[:10]),
            call_timestamp=datetime.utcnow(),
            repeat_count=new_repeat_count,
            similarity_score=similarity_score,
            status=status,
            block_status=block_required,
            block_reason=block_reason,
            request_type=req_type,
            transcript_summary=summary_text,
            suspicious_score=susp_score,
            review_status="PENDING",
            filename=filename
        )
        db.add(voice_record)

        call_log = UnknownCallerCallLogDB(
            profile_id=target_profile.id,
            caller_id=caller_id_clean,
            call_timestamp=datetime.utcnow(),
            similarity_score=similarity_score,
            repeat_count=new_repeat_count,
            status=status,
            decision_reason=decision_reason,
            filename=filename
        )
        db.add(call_log)
        db.commit()

        return {
            "success": True,
            "caller_status": "UNKNOWN",
            "caller_identifier": caller_id_clean,
            "caller_id": caller_id_clean,
            "voice_similarity": similarity_score,
            "repeat_count": new_repeat_count,
            "repeat_calls": new_repeat_count,
            "status": status,
            "security_status": status,
            "block_required": block_required,
            "is_blocked": block_required,
            "block_reason": block_reason,
            "decision_reason": decision_reason,
            "profile_id": target_profile.id,
            "event_id": voice_record.id,
            "timestamp": datetime.utcnow(),
            "disclaimer": "VOICE SIMILARITY DETECTED: Similar voice alone does not block; blocked only after configured repeat threshold.",
            "danger_alert": danger_alert,
            "danger_alert_message": danger_alert_message,
            "scam_threat_type": scam_threat,
            "security_action": security_action,
            "conversation_text": redacted_dialogue,
            "detected_intent": req_type,
            "request_type": req_type,
            "transcript_summary": summary_text,
            "suspicious_score": susp_score,
            "suspicious_repeat_count": susp_count,
            "suspicious_repeat_threshold": susp_thresh,
            "detected_requests": detected_requests,
            "risk_category": intent_result.get("risk_category", "NORMAL CONVERSATION"),
            "conversation_risk_level": intent_result.get("risk_level", "LOW RISK"),
            "conversation_risk_score": susp_score,
            "recommendation": intent_result.get("recommendation"),
            "warning_title": intent_result.get("warning_title"),
            "warning_message": intent_result.get("warning_message"),
            "highlighted_spans": intent_result.get("highlighted_spans", [])
        }

    def get_history(self, limit: int = 50, db: Session = None) -> List[Dict[str, Any]]:
        records = db.query(UnknownCallerVoiceRecordDB).order_by(desc(UnknownCallerVoiceRecordDB.call_timestamp)).limit(limit).all()
        return [
            {
                "id": r.id,
                "event_id": r.id,
                "timestamp": r.call_timestamp,
                "caller_identifier": r.caller_identifier,
                "caller_status": r.caller_status,
                "repeat_count": r.repeat_count,
                "voice_similarity": r.similarity_score,
                "status": r.status,
                "security_status": r.status,
                "request_type": r.request_type or "NONE",
                "transcript_summary": r.transcript_summary or "",
                "suspicious_score": r.suspicious_score or 0,
                "block_status": r.block_status,
                "block_reason": r.block_reason,
                "review_status": r.review_status,
                "reviewed_by": r.reviewed_by,
                "review_notes": r.review_notes,
                "filename": r.filename
            }
            for r in records
        ]

    def review_event(self, event_id: int, reviewed_by: str, review_notes: str, db: Session) -> Optional[Dict[str, Any]]:
        record = db.query(UnknownCallerVoiceRecordDB).filter(UnknownCallerVoiceRecordDB.id == event_id).first()
        if not record:
            return None
        record.review_status = "REVIEWED"
        record.reviewed_by = reviewed_by or "security_analyst"
        record.review_notes = review_notes
        db.commit()
        return {
            "event_id": record.id,
            "review_status": "REVIEWED",
            "reviewed_by": record.reviewed_by,
            "message": "Security event successfully reviewed."
        }

    def unblock_profile(self, target_id: int, review_notes: str, db: Session) -> Optional[Dict[str, Any]]:
        # Unblock in UnknownCallerProfileDB
        profile = db.query(UnknownCallerProfileDB).filter(UnknownCallerProfileDB.id == target_id).first()
        if not profile:
            profile = db.query(UnknownCallerProfileDB).order_by(desc(UnknownCallerProfileDB.id)).first()

        if profile:
            profile.is_blocked = False
            profile.status = "MONITORING"
            profile.repeat_count = 0
            profile.block_reason = f"Unblocked by operator: {review_notes}"

        # Unblock in UnknownCallerVoiceRecordDB
        voice_rec = db.query(UnknownCallerVoiceRecordDB).filter(UnknownCallerVoiceRecordDB.id == target_id).first()
        if not voice_rec:
            voice_rec = db.query(UnknownCallerVoiceRecordDB).order_by(desc(UnknownCallerVoiceRecordDB.id)).first()

        if voice_rec:
            voice_rec.block_status = False
            voice_rec.status = "UNBLOCKED"
            voice_rec.review_status = "REVIEWED"
            voice_rec.reviewed_by = "authorized_operator"
            voice_rec.review_notes = review_notes

        # Log unblock event in audit trail
        log_entry = UnknownCallerCallLogDB(
            profile_id=profile.id if profile else None,
            caller_id=profile.caller_id if profile else "Unknown",
            call_timestamp=datetime.utcnow(),
            similarity_score=profile.last_similarity_score if profile else 0.0,
            repeat_count=0,
            status="UNBLOCKED",
            decision_reason=f"Authorized operator verified & unblocked: {review_notes}",
            filename="OPERATOR_OVERRIDE"
        )
        db.add(log_entry)
        db.commit()

        return {
            "id": profile.id if profile else target_id,
            "caller_id": profile.caller_id if profile else "Unknown",
            "status": "UNBLOCKED",
            "is_blocked": False,
            "repeat_count": 0,
            "review_status": "REVIEWED",
            "message": "Caller voice profile successfully unblocked and repeat counter reset to 0."
        }

repeated_caller_service = RepeatedCallerService()
