import io
import os
import numpy as np
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import desc

from backend.models.database import (
    get_db,
    UnknownCallerProfileDB,
    UnknownCallerCallLogDB,
    UnknownCallerVoiceRecordDB,
    UnknownCallerSettingsDB
)
from backend.models.unknown_caller import (
    CallerAnalysisResponse,
    CallerProfileResponse,
    CallerCallLogResponse,
    SecurityEventLogResponse,
    CallerSettingsModel,
    UnblockRequest,
    ReviewEventRequest,
    ConversationAnalysisResult,
    ConversationAnalyzeRequest
)
from backend.services.repeated_caller_service import repeated_caller_service
from backend.services.scam_intent_analyzer import scam_intent_analyzer
from backend.services.audio_processor import audio_processor
from backend.utils.logger import logger

router = APIRouter(tags=["Unknown Caller Repeated Voice Protection"])

# =====================================================================
# 1. ANALYZE UNKNOWN CALLER (Section 10 API Specification)
# =====================================================================
@router.post("/api/security/unknown-caller/analyze", response_model=CallerAnalysisResponse)
@router.post("/api/caller-protection/process-call", response_model=CallerAnalysisResponse)
async def analyze_unknown_caller_call(
    file: UploadFile = File(...),
    caller_identifier: str = Form("Unknown"),
    caller_status: str = Form("UNKNOWN"),
    target_repeat_count: Optional[int] = Form(None),
    scam_threat: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    """
    Analyzes an unknown caller's voice call:
    Extracts acoustic voice embedding, validates caller status, compares with previous records,
    evaluates repeat counts, and returns security decision (ALLOW / MONITOR / WARN / BLOCK).
    """
    try:
        content = await file.read()
        if not content or len(content) == 0:
            res = repeated_caller_service.process_call(
                audio=None,
                caller_identifier=caller_identifier,
                caller_status=caller_status,
                filename=file.filename,
                db=db
            )
            return CallerAnalysisResponse(**res)

        audio, sr = audio_processor.load_audio_file(content, filename=file.filename or "audio.wav")
        if len(audio) == 0:
            res = repeated_caller_service.process_call(
                audio=None,
                caller_identifier=caller_identifier,
                caller_status=caller_status,
                filename=file.filename,
                db=db
            )
            return CallerAnalysisResponse(**res)

        result = repeated_caller_service.process_call(
            audio=audio,
            caller_identifier=caller_identifier,
            caller_status=caller_status,
            filename=file.filename,
            target_repeat_count=target_repeat_count,
            scam_threat=scam_threat,
            db=db
        )
        return CallerAnalysisResponse(**result)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing unknown caller call: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to analyze call: {str(e)}")


# =====================================================================
# 1B. CONVERSATION INTENT ANALYSIS (Scam & Social Engineering Protection)
# =====================================================================
@router.post("/api/security/conversation/analyze", response_model=ConversationAnalysisResult)
async def analyze_conversation_content(
    payload: ConversationAnalyzeRequest
):
    """
    Analyzes conversation transcript or live dialogue for sensitive scam requests
    (OTP, money, bank account, UPI, card, ATM PIN, password, CVV, remote access).
    """
    res = scam_intent_analyzer.analyze_conversation(payload.conversation_text)
    return ConversationAnalysisResult(
        success=True,
        caller_status=payload.caller_status or "UNKNOWN",
        **res
    )


# =====================================================================
# 2. SECURITY EVENT HISTORY (Section 10 & 14)
# =====================================================================
@router.get("/api/security/unknown-caller/history", response_model=List[SecurityEventLogResponse])
async def get_security_unknown_caller_history(limit: int = 50, db: Session = Depends(get_db)):
    """
    Returns historical unknown caller security events per Section 10 & 14.
    """
    events = repeated_caller_service.get_history(limit=limit, db=db)
    return [SecurityEventLogResponse(**ev) for ev in events]


@router.get("/api/caller-protection/logs", response_model=List[CallerCallLogResponse])
async def get_caller_protection_logs(limit: int = 50, db: Session = Depends(get_db)):
    """
    Legacy / Alias route for caller audit logs.
    """
    logs = db.query(UnknownCallerCallLogDB).order_by(desc(UnknownCallerCallLogDB.call_timestamp)).limit(limit).all()
    return [
        CallerCallLogResponse(
            id=getattr(l, "id"),
            profile_id=getattr(l, "profile_id"),
            caller_id=getattr(l, "caller_id"),
            call_timestamp=getattr(l, "call_timestamp"),
            similarity_score=getattr(l, "similarity_score"),
            repeat_count=getattr(l, "repeat_count"),
            status=getattr(l, "status"),
            decision_reason=getattr(l, "decision_reason"),
            filename=getattr(l, "filename")
        )
        for l in logs
    ]


# =====================================================================
# 3. SETTINGS (Section 6, 7 & 10)
# =====================================================================
@router.get("/api/security/unknown-caller/settings", response_model=CallerSettingsModel)
@router.get("/api/caller-protection/settings", response_model=CallerSettingsModel)
async def get_caller_protection_settings(db: Session = Depends(get_db)):
    sett = repeated_caller_service.get_settings(db)
    return CallerSettingsModel(
        voice_similarity_threshold=sett["voice_similarity_threshold"],
        repeated_call_warn_threshold=sett["repeated_call_warn_threshold"],
        repeated_call_block_threshold=sett["repeated_call_block_threshold"],
        suspicious_repeat_threshold=sett.get("suspicious_repeat_threshold", 3)
    )


@router.put("/api/security/unknown-caller/settings", response_model=CallerSettingsModel)
@router.put("/api/caller-protection/settings", response_model=CallerSettingsModel)
async def update_caller_protection_settings(data: CallerSettingsModel, db: Session = Depends(get_db)):
    repeated_caller_service.update_settings(
        warn_thresh=data.repeated_call_warn_threshold,
        block_thresh=data.repeated_call_block_threshold,
        sim_thresh=data.voice_similarity_threshold,
        susp_thresh=data.suspicious_repeat_threshold,
        db=db
    )
    return data


# =====================================================================
# 4. ADMIN REVIEW & UNBLOCK (Section 10 & 15)
# =====================================================================
@router.post("/api/security/unknown-caller/{id}/review")
async def review_security_event(
    id: int,
    request: ReviewEventRequest = ReviewEventRequest(),
    db: Session = Depends(get_db)
):
    """
    Authorized review interface: marks event as reviewed per Section 15.
    """
    res = repeated_caller_service.review_event(id, request.reviewed_by, request.review_notes or "", db)
    if not res:
        raise HTTPException(status_code=404, detail=f"Security event #{id} not found.")
    return res


@router.post("/api/security/unknown-caller/{id}/unblock")
@router.post("/api/caller-protection/profiles/{id}/unblock")
async def unblock_unknown_caller(
    id: int,
    request: UnblockRequest = UnblockRequest(),
    db: Session = Depends(get_db)
):
    """
    Authorized action: reverses block, resets repeat counter, marks unblocked per Section 10 & 15.
    """
    res = repeated_caller_service.unblock_profile(id, request.review_notes or "", db)
    if not res:
        raise HTTPException(status_code=404, detail=f"Unknown caller record #{id} not found.")
    return res


# =====================================================================
# 5. PROFILES (Alias / Diagnostic)
# =====================================================================
@router.get("/api/caller-protection/profiles", response_model=List[CallerProfileResponse])
async def get_caller_profiles(db: Session = Depends(get_db)):
    profiles = db.query(UnknownCallerProfileDB).order_by(desc(UnknownCallerProfileDB.last_call_timestamp)).all()
    return [
        CallerProfileResponse(
            id=getattr(p, "id"),
            caller_id=getattr(p, "caller_id"),
            repeat_count=getattr(p, "repeat_count"),
            last_similarity_score=getattr(p, "last_similarity_score"),
            status=getattr(p, "status"),
            is_blocked=getattr(p, "is_blocked"),
            block_reason=getattr(p, "block_reason"),
            last_call_timestamp=getattr(p, "last_call_timestamp"),
            created_at=getattr(p, "created_at")
        )
        for p in profiles
    ]


@router.get("/api/caller-protection/profiles/{profile_id}", response_model=CallerProfileResponse)
async def get_caller_profile(profile_id: int, db: Session = Depends(get_db)):
    profile = db.query(UnknownCallerProfileDB).filter(UnknownCallerProfileDB.id == profile_id).first()
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found.")
    return CallerProfileResponse(
        id=getattr(profile, "id"),
        caller_id=getattr(profile, "caller_id"),
        repeat_count=getattr(profile, "repeat_count"),
        last_similarity_score=getattr(profile, "last_similarity_score"),
        status=getattr(profile, "status"),
        is_blocked=getattr(profile, "is_blocked"),
        block_reason=getattr(profile, "block_reason"),
        last_call_timestamp=getattr(profile, "last_call_timestamp"),
        created_at=getattr(profile, "created_at")
    )


# =====================================================================
# 6. SIMULATE CALL (Interactive testing helper)
# =====================================================================
@router.post("/api/security/unknown-caller/simulate", response_model=CallerAnalysisResponse)
@router.post("/api/caller-protection/simulate", response_model=CallerAnalysisResponse)
async def simulate_unknown_caller_call(
    caller_identifier: str = Query("Unknown"),
    caller_status: str = Query("UNKNOWN"),
    target_call_number: Optional[int] = Query(None),
    simulation_type: str = Query("repeat_caller"), # 'repeat_caller', 'new_caller', 'known_caller'
    scam_threat: Optional[str] = Query(None),       # 'OTP_THEFT', 'MONEY_DEMAND', 'CREDENTIAL_THEFT', 'SAFETY_CONTEXT'
    conversation_text: Optional[str] = Query(None),
    db: Session = Depends(get_db)
):
    """
    Interactive test helper to simulate incoming unknown or known calls.
    Allows exact step execution:
      - Call 1 -> MONITORING (Repeat 1)
      - Call 2 -> MONITORING (Repeat 2)
      - Call 3 -> WARNING (Repeat 3)
      - Call 4 -> BLOCKED / 🔴 DANGER ALERT (Repeat 4)
      - Known caller -> ALLOW
      - OTP Scam / Money Scam testing
      - Conversational social engineering analysis
    """
    sr = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    # Preset conversation text if scam_threat specified and no custom text passed
    if not conversation_text:
        if scam_threat in ["OTP_THEFT", "DEMO_CALL_1"]:
            conversation_text = "Give me the OTP."
        elif scam_threat in ["MONEY_DEMAND", "DEMO_CALL_2"]:
            conversation_text = "Give me the money."
        elif scam_threat in ["INVOICE_DEMAND", "DEMO_CALL_3"]:
            conversation_text = "Send the invoice and make the payment."
        elif scam_threat == "CREDENTIAL_THEFT":
            conversation_text = "Give me your ATM PIN and 16-digit debit card number right now."
        elif scam_threat == "BANK_DETAILS":
            conversation_text = "Give me your bank details and account number."
        elif scam_threat == "SAFETY_CONTEXT":
            conversation_text = "Your OTP is 4921; never share your OTP with anyone."
        elif scam_threat == "NORMAL_CALL":
            conversation_text = "Hello, I am calling from courier delivery regarding your parcel address."

    if simulation_type == "known_caller" or caller_status.upper() == "KNOWN":
        # Known caller bypass
        res = repeated_caller_service.process_call(
            audio=None,
            caller_identifier=caller_identifier if caller_identifier != "Unknown" else "Trusted Employee #4102",
            caller_status="KNOWN",
            filename="trusted_known_caller.wav",
            conversation_text=conversation_text or "Authorized caller voice channel established.",
            db=db
        )
        return CallerAnalysisResponse(**res)

    if simulation_type == "new_caller":
        f0 = np.random.uniform(250.0, 420.0)
        audio = 0.3 * np.sin(2 * np.pi * f0 * t) + 0.15 * np.sin(2 * np.pi * (2 * f0) * t)
        sim_name = f"new_unknown_voice_{int(f0)}hz.wav"
    else:
        # Same unknown voice tone (175 Hz + harmonics)
        audio = (
            0.35 * np.sin(2 * np.pi * 175.0 * t) +
            0.20 * np.sin(2 * np.pi * 350.0 * t) +
            0.12 * np.sin(2 * np.pi * 525.0 * t) +
            0.05 * np.sin(2 * np.pi * 700.0 * t)
        )
        audio += 0.05 * np.sin(2 * np.pi * 175.0 * t * 1.02)
        sim_name = "simulated_unknown_caller_voice.wav"

    audio = audio.astype(np.float32)
    result = repeated_caller_service.process_call(
        audio=audio,
        caller_identifier=caller_identifier,
        caller_status="UNKNOWN",
        filename=sim_name,
        target_repeat_count=target_call_number,
        scam_threat=scam_threat,
        conversation_text=conversation_text,
        db=db
    )
    return CallerAnalysisResponse(**result)


# =====================================================================
# 7. RESET TEST DATA
# =====================================================================
@router.post("/api/security/unknown-caller/reset")
@router.post("/api/caller-protection/reset")
async def reset_unknown_caller_data(db: Session = Depends(get_db)):
    """
    Clears test records and profiles for fresh demonstration testing.
    """
    db.query(UnknownCallerVoiceRecordDB).delete()
    db.query(UnknownCallerCallLogDB).delete()
    db.query(UnknownCallerProfileDB).delete()
    db.commit()
    return {"status": "success", "message": "Unknown caller protection data reset."}
