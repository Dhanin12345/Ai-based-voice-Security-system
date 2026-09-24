import pytest
import numpy as np
from fastapi.testclient import TestClient

from backend.main import app
from backend.models.database import init_db, SessionLocal, UnknownCallerSettingsDB
from backend.services.voice_embedding_service import voice_embedding_service
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
            similarity_threshold=85.0
        )
        db.add(sett)
    else:
        sett.repeat_warn_threshold = 3  # type: ignore
        sett.repeat_block_threshold = 4  # type: ignore
        sett.similarity_threshold = 85.0  # type: ignore
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
# SECTION 20: TESTS 1 TO 10
# =====================================================================

def test_01_known_caller_no_blocking():
    """
    Test 1: Known caller -> no unknown-caller blocking.
    """
    db = SessionLocal()
    audio = generate_voice_tone(freq=175.0)
    res = repeated_caller_service.process_call(
        audio=audio,
        caller_identifier="Employee_101",
        caller_status="KNOWN",
        db=db
    )
    assert res["caller_status"] == "KNOWN"
    assert res["status"] == "ALLOW"
    assert res["block_required"] is False
    assert res["is_blocked"] is False
    db.close()


def test_02_unknown_caller_first_call_monitoring():
    """
    Test 2: Unknown caller first call -> MONITORING.
    """
    db = SessionLocal()
    audio = generate_voice_tone(freq=175.0)
    res = repeated_caller_service.process_call(
        audio=audio,
        caller_identifier="Unknown",
        caller_status="UNKNOWN",
        db=db
    )
    assert res["caller_status"] == "UNKNOWN"
    assert res["status"] == "MONITORING"
    assert res["repeat_count"] == 1
    assert res["block_required"] is False
    db.close()


def test_03_unknown_caller_low_similarity_no_repeat_increment():
    """
    Test 3: Unknown caller with low similarity -> no repeat increment.
    """
    db = SessionLocal()
    audio_voice1 = generate_voice_tone(freq=140.0)
    audio_voice2 = generate_voice_tone(freq=680.0)  # very distinct voice

    res1 = repeated_caller_service.process_call(audio=audio_voice1, caller_identifier="Unknown", db=db)
    assert res1["repeat_count"] == 1

    # Second call has completely distinct voice -> should register as new profile (repeat_count = 1)
    res2 = repeated_caller_service.process_call(audio=audio_voice2, caller_identifier="Unknown", db=db)
    assert res2["repeat_count"] == 1
    assert res2["status"] == "MONITORING"
    db.close()


def test_04_unknown_caller_sufficient_similarity_increments_repeat():
    """
    Test 4: Unknown caller with sufficient similarity -> repeat counter increments.
    """
    db = SessionLocal()
    same_voice = generate_voice_tone(freq=180.0)

    res1 = repeated_caller_service.process_call(audio=same_voice, caller_identifier="Unknown", db=db)
    assert res1["repeat_count"] == 1

    res2 = repeated_caller_service.process_call(audio=same_voice, caller_identifier="Unknown", db=db)
    assert res2["repeat_count"] == 2
    assert res2["voice_similarity"] >= 85.0
    assert res2["status"] == "MONITORING"
    assert res2["block_required"] is False
    db.close()


def test_05_repeated_matching_calls_below_block_threshold():
    """
    Test 5: Repeated matching calls below block threshold -> WARNING / MONITORING.
    """
    db = SessionLocal()
    same_voice = generate_voice_tone(freq=180.0)

    for i in range(2):
        repeated_caller_service.process_call(audio=same_voice, caller_identifier="Unknown", db=db)

    # Call 3 reaches warning threshold (3)
    res3 = repeated_caller_service.process_call(audio=same_voice, caller_identifier="Unknown", db=db)
    assert res3["repeat_count"] == 3
    assert res3["status"] == "WARNING"
    assert res3["block_required"] is False
    db.close()


def test_06_repeated_matching_calls_reaching_block_threshold():
    """
    Test 6: Repeated matching calls reaching block threshold -> BLOCKED.
    """
    db = SessionLocal()
    same_voice = generate_voice_tone(freq=180.0)

    for i in range(3):
        repeated_caller_service.process_call(audio=same_voice, caller_identifier="Unknown", db=db)

    # Call 4 reaches block threshold (4)
    res4 = repeated_caller_service.process_call(
        audio=same_voice,
        caller_identifier="Unknown",
        scam_threat="OTP_THEFT",
        db=db
    )
    assert res4["repeat_count"] == 4
    assert res4["status"] in ["BLOCKED", "CALL BLOCKED"]
    assert res4["block_required"] is True
    assert "Unknown caller reached" in res4["block_reason"] or "threshold" in res4["block_reason"]
    db.close()


def test_07_voice_embedding_failure():
    """
    Test 7: Voice embedding failure -> REVIEW / ANALYSIS UNAVAILABLE.
    """
    db = SessionLocal()
    # Empty audio
    res_empty = repeated_caller_service.process_call(audio=np.array([]), db=db)
    assert res_empty["status"] == "VOICE ANALYSIS UNAVAILABLE"
    assert res_empty["block_required"] is False

    # Forced embedding failure
    audio = generate_voice_tone(freq=180.0)
    res_fail = repeated_caller_service.process_call(audio=audio, force_embedding_failure=True, db=db)
    assert res_fail["status"] == "REVIEW"
    assert res_fail["block_required"] is False
    db.close()


def test_08_similarity_calculation_failure_no_false_block():
    """
    Test 8: Similarity calculation failure -> no false block.
    """
    db = SessionLocal()
    audio = generate_voice_tone(freq=180.0)
    # First call
    repeated_caller_service.process_call(audio=audio, db=db)

    # Subsequent call with similarity failure
    res = repeated_caller_service.process_call(audio=audio, force_similarity_failure=True, db=db)
    assert res["block_required"] is False
    assert res["status"] != "BLOCKED"
    db.close()


def test_09_admin_unblock():
    """
    Test 9: Admin unblock -> status changes to UNBLOCKED/REVIEWED.
    """
    db = SessionLocal()
    audio = generate_voice_tone(freq=180.0)

    # Trigger block
    res = None
    for _ in range(4):
        res = repeated_caller_service.process_call(audio=audio, db=db)
    assert res is not None and res["block_required"] is True
    profile_id = res["profile_id"]
    assert profile_id is not None

    # Admin unblocks
    unblock_res = repeated_caller_service.unblock_profile(profile_id, "Operator verified caller identity", db)
    assert unblock_res is not None
    assert unblock_res["status"] == "UNBLOCKED"
    assert unblock_res["is_blocked"] is False
    assert unblock_res["repeat_count"] == 0

    # API unblock endpoint test
    api_unblock = client.post(f"/api/security/unknown-caller/{profile_id}/unblock")
    assert api_unblock.status_code == 200
    assert api_unblock.json()["is_blocked"] is False
    db.close()


def test_10_existing_human_ai_detection_unaffected():
    """
    Test 10: Existing Human/AI detection -> verify it works exactly as before.
    """
    # Core health
    health_resp = client.get("/api/health")
    assert health_resp.status_code == 200
    assert health_resp.json()["status"] == "healthy"

    # Core detection history
    history_resp = client.get("/api/history")
    assert history_resp.status_code == 200

    # Section 10 API endpoints validation
    history_sec_resp = client.get("/api/security/unknown-caller/history")
    assert history_sec_resp.status_code == 200

    sett_resp = client.get("/api/security/unknown-caller/settings")
    assert sett_resp.status_code == 200
    assert sett_resp.json()["repeated_call_block_threshold"] == 4
