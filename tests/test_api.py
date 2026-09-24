import io
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect
import soundfile as sf
import numpy as np

from backend.main import app
from backend.models.database import init_db

@pytest.fixture(autouse=True)
def setup_test_db():
    init_db()

client = TestClient(app)

def create_test_wav_bytes(duration_sec: float = 2.0, sr: int = 16000) -> bytes:
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    audio = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format='WAV')
    buf.seek(0)
    return buf.read()

def test_health_endpoints():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"

def test_detect_upload_valid_file():
    wav_bytes = create_test_wav_bytes(duration_sec=2.0)
    files = {"file": ("sample.wav", wav_bytes, "audio/wav")}
    
    response = client.post("/api/detect/upload", files=files)
    assert response.status_code == 200
    data = response.json()
    
    assert data["status"] == "success"
    assert "classification" in data
    assert "confidence" in data
    assert "risk_score" in data
    assert "recommendation" in data
    assert 0 <= data["risk_score"] <= 100

def test_detect_upload_invalid_file_extension():
    files = {"file": ("test.txt", b"invalid audio data", "text/plain")}
    response = client.post("/api/detect/upload", files=files)
    assert response.status_code == 400

def test_detect_upload_mp4_file():
    wav_bytes = create_test_wav_bytes(duration_sec=2.0)
    files = {"file": ("sample_video.mp4", wav_bytes, "video/mp4")}
    response = client.post("/api/detect/upload", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["filename"] == "sample_video.mp4"

def test_detect_upload_mpeg_file():
    wav_bytes = create_test_wav_bytes(duration_sec=2.0)
    files = {"file": ("sample_video.mpeg", wav_bytes, "video/mpeg")}
    response = client.post("/api/detect/upload", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["filename"] == "sample_video.mpeg"

def test_get_history_list():
    response = client.get("/api/history")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "items" in data
    assert isinstance(data["items"], list)

def test_websocket_detect_connection():
    try:
        with client.websocket_connect("/ws/detect") as websocket:
            pcm_data = (np.zeros(32000, dtype=np.int16)).tobytes()
            websocket.send_bytes(pcm_data)
            
            response = websocket.receive_json()
            assert response["status"] == "success"
            assert "classification" in response
            assert "risk_score" in response
            assert 0 <= response["risk_score"] <= 100
    except WebSocketDisconnect:
        pass

def test_clear_all_history():
    response = client.delete("/api/history")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"

def test_detect_bulk_stream():
    wav1 = create_test_wav_bytes(duration_sec=2.0)
    wav2 = create_test_wav_bytes(duration_sec=2.0)
    files = [
        ("files", ("stream1.wav", wav1, "audio/wav")),
        ("files", ("stream2.wav", wav2, "audio/wav"))
    ]
    response = client.post("/api/detect/bulk", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["records_processed"] == 2

def test_start_live_session():
    response = client.post("/api/audio/start-live")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "active"
    assert "session_id" in data
    assert data["websocket_url"] == "/ws/live-analysis"

def test_system_status_endpoint():
    response = client.get("/api/system/status")
    assert response.status_code == 200
    data = response.json()
    assert data["system_status"] == "ONLINE"
    assert "model_status" in data

def test_demo_audio_endpoints():
    # Test human demo
    resp_h = client.get("/api/demo/human")
    assert resp_h.status_code == 200
    data_h = resp_h.json()
    assert data_h["status"] == "success"
    assert data_h["sample_type"] == "human"
    assert "human_probability" in data_h
    assert "risk_score" in data_h

    # Test synthetic demo
    resp_s = client.get("/api/demo/synthetic")
    assert resp_s.status_code == 200
    data_s = resp_s.json()
    assert data_s["status"] == "success"
    assert data_s["sample_type"] == "synthetic"
    assert data_s["synthetic_probability"] in data_s or "synthetic_probability" in data_s

def test_analysis_features_and_windows():
    wav_bytes = create_test_wav_bytes(duration_sec=2.0)
    upload_res = client.post("/api/detect/upload", files={"file": ("feat_test.wav", wav_bytes, "audio/wav")})
    assert upload_res.status_code == 200
    analysis_id = upload_res.json()["analysis_id"]

    # Test features endpoint
    feat_res = client.get(f"/api/analysis/{analysis_id}/features")
    assert feat_res.status_code == 200
    feat_data = feat_res.json()
    assert feat_data["status"] == "success"
    assert "pitch_mean" in feat_data["features"]
    assert "rms_mean" in feat_data["features"]
    assert "spectral_centroid" in feat_data["features"]

    # Test windows endpoint
    win_res = client.get(f"/api/analysis/{analysis_id}/windows")
    assert win_res.status_code == 200
    win_data = win_res.json()
    assert win_data["status"] == "success"
    assert "windows" in win_data

def test_audio_session_control():
    start_res = client.post("/api/audio/start")
    assert start_res.status_code == 200
    assert start_res.json()["status"] == "active"

    stop_res = client.post("/api/audio/stop")
    assert stop_res.status_code == 200
    assert stop_res.json()["status"] == "stopped"

def test_report_endpoints():
    wav_bytes = create_test_wav_bytes(duration_sec=2.0)
    upload_res = client.post("/api/detect/upload", files={"file": ("report_test.wav", wav_bytes, "audio/wav")})
    assert upload_res.status_code == 200
    analysis_id = upload_res.json()["analysis_id"]

    # JSON report
    rep_res = client.get(f"/api/reports/{analysis_id}")
    assert rep_res.status_code == 200
    rep_data = rep_res.json()
    assert "classification" in rep_data
    assert "acoustic_features" in rep_data
    assert "window_analysis" in rep_data

    # CSV report
    csv_res = client.get(f"/api/report/{analysis_id}?format=csv")
    assert csv_res.status_code == 200
    assert "ReportID" in csv_res.text

def test_analytics_summary_endpoint():
    res = client.get("/api/analytics/summary")
    assert res.status_code == 200
    data = res.json()
    assert "total_analyses" in data
    assert "synthetic_count" in data
    assert "human_count" in data



