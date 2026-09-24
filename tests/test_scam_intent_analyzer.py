import pytest
import numpy as np
from fastapi.testclient import TestClient

from backend.main import app
from backend.models.database import init_db, SessionLocal, UnknownCallerSettingsDB
from backend.services.scam_intent_analyzer import scam_intent_analyzer
from backend.services.repeated_caller_service import repeated_caller_service

client = TestClient(app)

@pytest.fixture(autouse=True)
def setup_test_db():
    init_db()
    client.post("/api/security/unknown-caller/reset")
    db = SessionLocal()
    sett = db.query(UnknownCallerSettingsDB).filter(UnknownCallerSettingsDB.id == 1).first()
    if not sett:
        sett = UnknownCallerSettingsDB(
            id=1,
            repeat_warn_threshold=3,
            repeat_block_threshold=4,
            similarity_threshold=85.0,
            suspicious_repeat_threshold=3
        )
        db.add(sett)
    else:
        sett.repeat_warn_threshold = 3
        sett.repeat_block_threshold = 4
        sett.similarity_threshold = 85.0
        sett.suspicious_repeat_threshold = 3
    db.commit()
    db.close()
    yield

def generate_voice_tone(freq=180.0, duration=1.5, sr=16000):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    audio = (
        0.4 * np.sin(2 * np.pi * freq * t) +
        0.2 * np.sin(2 * np.pi * (2 * freq) * t) +
        0.1 * np.sin(2 * np.pi * (3 * freq) * t)
    )
    return audio.astype(np.float32)

# =====================================================================
# CONVERSATION INTENT & SCAM DETECTION TESTS
# =====================================================================

def test_otp_request_intent_detection():
    """
    Test: 'Tell me the OTP you received' -> OTP_REQUEST, HIGH RISK.
    """
    res = scam_intent_analyzer.analyze_conversation("Tell me the OTP that you just received.")
    assert res["detected_intent"] == "OTP_REQUEST"
    assert res["risk_level"] == "HIGH RISK"
    assert res["is_safety_context"] is False
    assert len(res["highlighted_spans"]) > 0

    res2 = scam_intent_analyzer.analyze_conversation("Give me the OTP.")
    assert res2["detected_intent"] == "OTP_REQUEST"
    assert res2["risk_level"] == "HIGH RISK"


def test_money_request_intent_detection():
    """
    Test: 'Give me the money' / 'Transfer the money' -> MONEY_REQUEST, HIGH RISK.
    """
    res = scam_intent_analyzer.analyze_conversation("Give me the money.")
    assert res["detected_intent"] == "MONEY_REQUEST"
    assert res["risk_level"] == "HIGH RISK"

    res2 = scam_intent_analyzer.analyze_conversation("Send ₹10,000 immediately to this UPI account.")
    assert res2["detected_intent"] == "MONEY_REQUEST"
    assert res2["urgency_detected"] is True


def test_credential_request_intent_detection():
    """
    Test: 'Give me your ATM PIN' -> CREDENTIAL_REQUEST, HIGH RISK.
    """
    res = scam_intent_analyzer.analyze_conversation("Give me your ATM PIN right now.")
    assert res["detected_intent"] == "CREDENTIAL_REQUEST"
    assert res["risk_level"] == "HIGH RISK"


def test_bank_details_request_intent_detection():
    """
    Test: 'What is your bank account number and IFSC code?' -> BANK_DETAILS_REQUEST.
    """
    res = scam_intent_analyzer.analyze_conversation("What is your bank account number and IFSC code?")
    assert res["detected_intent"] == "BANK_DETAILS_REQUEST"


def test_invoice_and_payment_request_intent_detection():
    """
    Test: 'Send the invoice' -> INVOICE_REQUEST, 'Make the payment' -> PAYMENT_REQUEST.
    """
    res_inv = scam_intent_analyzer.analyze_conversation("Send the invoice.")
    assert res_inv["detected_intent"] == "INVOICE_REQUEST"

    res_pay = scam_intent_analyzer.analyze_conversation("Send the payment immediately.")
    assert res_pay["detected_intent"] == "PAYMENT_REQUEST"


def test_safety_context_false_positive_prevention():
    """
    Test: 'Your OTP is 1234; don't share it with anyone' -> SAFETY_NOTIFICATION, LOW RISK (No false alarm).
    """
    res1 = scam_intent_analyzer.analyze_conversation("Your OTP is 1234; don't share it with anyone.")
    assert res1["is_safety_context"] is True
    assert res1["risk_level"] == "LOW RISK"
    assert res1["action"] == "ALLOW"

    res2 = scam_intent_analyzer.analyze_conversation("Never share your OTP with anyone.")
    assert res2["is_safety_context"] is True
    assert res2["risk_level"] == "LOW RISK"


def test_normal_conversation_not_flagged():
    """
    Test: Normal conversation should remain LOW RISK with no scam intent.
    """
    res = scam_intent_analyzer.analyze_conversation("Hello, I am calling regarding your delivery package tomorrow afternoon.")
    assert res["detected_intent"] == "NONE"
    assert res["risk_level"] == "LOW RISK"
    assert res["action"] == "MONITOR"
    assert res["is_safety_context"] is False


def test_redaction_of_sensitive_data():
    """
    Test Section 15: Numeric OTPs, passwords, PINs, and account numbers must be redacted.
    """
    text = "My OTP is 384921 and account number is 987654321012"
    redacted = scam_intent_analyzer.redact_sensitive_data(text)
    assert "384921" not in redacted
    assert "[REDACTED CODE]" in redacted
    assert "987654321012" not in redacted


def test_conversation_analysis_api_endpoint():
    """
    Test: POST /api/security/conversation/analyze returns structured response.
    """
    resp = client.post("/api/security/conversation/analyze", json={
        "conversation_text": "Tell me the OTP that you just received.",
        "caller_identifier": "Unknown",
        "caller_status": "UNKNOWN"
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert data["detected_intent"] == "OTP_REQUEST"
    assert data["risk_level"] == "HIGH RISK"
    assert "Do not share OTP" in data["recommendation"]


def test_multi_call_timeline_section_13_and_16():
    """
    Test Section 13 & 16:
    Call 1: Unknown Caller, "Give me the OTP" -> OTP_REQUEST -> Repeat 1/3 -> MONITORING
    Call 2: Same unknown voice, "Give me the money" -> MONEY_REQUEST -> Repeat 2/3 -> WARNING
    Call 3: Same unknown voice, "Send the invoice" -> INVOICE_REQUEST -> Repeat 3/3 -> CALL BLOCKED
    """
    db = SessionLocal()
    voice = generate_voice_tone(freq=180.0)

    # CALL 1
    res1 = repeated_caller_service.process_call(
        audio=voice,
        caller_identifier="Unknown",
        conversation_text="Give me the OTP.",
        db=db
    )
    assert res1["caller_status"] == "UNKNOWN"
    assert res1["request_type"] == "OTP_REQUEST"
    assert res1["suspicious_repeat_count"] == 1
    assert res1["status"] == "MONITORING"
    assert res1["block_required"] is False
    assert "OTP_REQUEST" in res1["detected_requests"]

    # CALL 2
    res2 = repeated_caller_service.process_call(
        audio=voice,
        caller_identifier="Unknown",
        conversation_text="Give me the money.",
        db=db
    )
    assert res2["caller_status"] == "UNKNOWN"
    assert res2["request_type"] == "MONEY_REQUEST"
    assert res2["suspicious_repeat_count"] == 2
    assert res2["status"] == "WARNING"
    assert res2["block_required"] is False
    assert "OTP_REQUEST" in res2["detected_requests"]
    assert "MONEY_REQUEST" in res2["detected_requests"]

    # CALL 3: Reaches threshold 3 -> BLOCKED!
    res3 = repeated_caller_service.process_call(
        audio=voice,
        caller_identifier="Unknown",
        conversation_text="Send the invoice.",
        db=db
    )
    assert res3["caller_status"] == "UNKNOWN"
    assert res3["request_type"] == "INVOICE_REQUEST"
    assert res3["suspicious_repeat_count"] == 3
    assert res3["status"] in ["CALL BLOCKED", "BLOCKED"]
    assert res3["block_required"] is True
    assert res3["danger_alert"] is True
    assert "Repeated suspicious requests from an unknown caller." in res3["block_reason"]
    assert len(res3["detected_requests"]) >= 3
    db.close()


def test_existing_detection_system_intact():
    """
    Test: Ensure existing Human vs AI detection endpoint is untouched and operational.
    """
    sr = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    synthetic_wav = (0.5 * np.sign(np.sin(2 * np.pi * 440 * t))).astype(np.float32)

    import io
    import soundfile as sf
    buf = io.BytesIO()
    sf.write(buf, synthetic_wav, sr, format="WAV")
    buf.seek(0)

    response = client.post(
        "/api/analyze",
        files={"file": ("test_synthetic.wav", buf, "audio/wav")}
    )
    assert response.status_code == 200
    data = response.json()
    assert "classification" in data
    assert "confidence" in data
    assert "human_probability" in data or "human_confidence" in data

    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["status"] == "healthy"

    history = client.get("/api/history")
    assert history.status_code == 200
