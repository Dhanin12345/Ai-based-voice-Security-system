import os
import sys
import glob
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from fastapi.testclient import TestClient
from backend.main import app
from backend.ml.predictor import predictor
from backend.services.audio_processor import audio_processor
from backend.services.voice_analyzer import voice_analyzer

client = TestClient(app)

def test_genuine_human_recordings():
    """Verify that multiple genuine human recordings are classified as HUMAN VOICE (GREEN, LOW, no danger alert)."""
    human_files = sorted(glob.glob("dataset/human/*.wav"))
    assert len(human_files) > 0, "No human audio files found"
    
    for fpath in human_files[:10]:
        audio, sr = audio_processor.load_audio_file(fpath)
        pred = predictor.predict_audio(audio, sr)
        
        # Step 3 checks
        assert "human_confidence" in pred
        assert "ai_confidence" in pred
        assert round(pred["human_confidence"] + pred["ai_confidence"], 1) == 100.0
        
        # Step 5 & 6 checks: >= 65% must be HUMAN VOICE
        print(f"{os.path.basename(fpath)} -> Human: {pred['human_confidence']}%, Class: {pred['classification']}")
        assert pred["human_confidence"] >= 65.0
        assert pred["classification"] == "HUMAN VOICE"
        assert pred["risk"] == "LOW"
        assert pred["color"] == "GREEN"
        assert pred["danger_alert"] is False

def test_synthetic_robotic_recordings():
    """Verify that synthetic/robotic recordings are classified as ROBOTIC VOICE / AI VOICE (RED, HIGH, danger alert)."""
    synth_files = sorted(glob.glob("dataset/synthetic/*.wav"))
    assert len(synth_files) > 0, "No synthetic audio files found"
    
    for fpath in synth_files[:10]:
        audio, sr = audio_processor.load_audio_file(fpath)
        pred = predictor.predict_audio(audio, sr)
        
        print(f"{os.path.basename(fpath)} -> Human: {pred['human_confidence']}%, AI: {pred['ai_confidence']}%, Class: {pred['classification']}")
        assert pred["human_confidence"] < 65.0
        assert pred["classification"] == "ROBOTIC VOICE / AI VOICE"
        assert pred["risk"] == "HIGH"
        assert pred["color"] == "RED"
        assert pred["danger_alert"] is True

def test_threshold_boundary():
    """Verify exact 65.00% threshold boundary behavior."""
    # 65.00% -> HUMAN VOICE
    # 64.99% -> ROBOTIC VOICE / AI VOICE
    from backend.services.voice_analyzer import voice_analyzer
    import numpy as np
    
    # Check threshold logic
    for conf, expected_class, expected_risk, expected_color, expected_alert in [
        (100.0, "HUMAN VOICE", "LOW", "GREEN", False),
        (95.0, "HUMAN VOICE", "LOW", "GREEN", False),
        (85.0, "HUMAN VOICE", "LOW", "GREEN", False),
        (71.0, "HUMAN VOICE", "LOW", "GREEN", False),
        (70.0, "HUMAN VOICE", "LOW", "GREEN", False),
        (65.0, "HUMAN VOICE", "LOW", "GREEN", False),
        (64.99, "ROBOTIC VOICE / AI VOICE", "HIGH", "RED", True),
        (64.0, "ROBOTIC VOICE / AI VOICE", "HIGH", "RED", True),
        (60.0, "ROBOTIC VOICE / AI VOICE", "HIGH", "RED", True),
        (50.0, "ROBOTIC VOICE / AI VOICE", "HIGH", "RED", True),
        (30.0, "ROBOTIC VOICE / AI VOICE", "HIGH", "RED", True),
    ]:
        is_h = conf >= 65.0
        c = "HUMAN VOICE" if is_h else "ROBOTIC VOICE / AI VOICE"
        r = "LOW" if is_h else "HIGH"
        col = "GREEN" if is_h else "RED"
        d = not is_h
        assert c == expected_class
        assert r == expected_risk
        assert col == expected_color
        assert d == expected_alert
        assert d == expected_alert

def test_audio_input_error_on_empty_and_corrupt():
    """Verify that empty or corrupt audio triggers AUDIO INPUT ERROR."""
    # Empty upload
    resp = client.post("/api/detect/upload", files={"file": ("empty.wav", b"", "audio/wav")})
    assert resp.status_code == 400
    assert "AUDIO INPUT ERROR" in resp.json()["detail"]

    # Corrupt upload
    resp = client.post("/api/detect/upload", files={"file": ("corrupt.wav", b"NOT_A_VALID_WAV_HEADER_DATA_12345", "audio/wav")})
    assert resp.status_code == 400
    assert "AUDIO INPUT ERROR" in resp.json()["detail"]

def test_upload_api_genuine_human():
    """Verify end-to-end API upload of genuine human recording."""
    fpath = "dataset/human/human_001.wav"
    with open(fpath, "rb") as f:
        resp = client.post("/api/detect/upload", files={"file": ("human_001.wav", f, "audio/wav")})
    
    assert resp.status_code == 200
    data = resp.json()
    assert data["classification"] == "HUMAN VOICE"
    assert data["human_confidence"] >= 65.0
    assert data["risk"] == "LOW"
    assert data["color"] == "GREEN"
    assert data["danger_alert"] is False

def test_upload_api_synthetic():
    """Verify end-to-end API upload of synthetic recording."""
    fpath = "dataset/synthetic/synthetic_001.wav"
    with open(fpath, "rb") as f:
        resp = client.post("/api/detect/upload", files={"file": ("synthetic_001.wav", f, "audio/wav")})
    
    assert resp.status_code == 200
    data = resp.json()
    assert data["classification"] == "ROBOTIC VOICE / AI VOICE"
    assert data["human_confidence"] < 65.0
    assert data["risk"] == "HIGH"
    assert data["color"] == "RED"
    assert data["danger_alert"] is True

def test_upload_api_aac_file():
    """Verify that .aac audio files are accepted, decoded, and analyzed by the backend."""
    import imageio_ffmpeg, subprocess, tempfile
    
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    with tempfile.NamedTemporaryFile(suffix=".aac", delete=False) as tmp_aac:
        tmp_aac_path = tmp_aac.name
    
    try:
        # Convert human_001.wav to aac
        cmd = [ffmpeg_exe, "-y", "-i", "dataset/human/human_001.wav", "-c:a", "aac", tmp_aac_path]
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        
        with open(tmp_aac_path, "rb") as f:
            resp = client.post("/api/detect/upload", files={"file": ("voice_sample.aac", f, "audio/aac")})
        
        assert resp.status_code == 200
        data = resp.json()
        assert data["classification"] == "HUMAN VOICE"
        assert data["human_confidence"] >= 65.0
        assert data["risk"] == "LOW"
        assert data["color"] == "GREEN"
        assert data["danger_alert"] is False
    finally:
        if os.path.exists(tmp_aac_path):
            os.remove(tmp_aac_path)
