import pytest
import numpy as np

from backend.services.detector import voice_clone_detector
from backend.services.risk_scorer import risk_scorer

def test_voice_clone_detector_interface():
    sr = 16000
    dummy_audio = np.random.randn(sr * 2).astype(np.float32)

    result = voice_clone_detector.predict(dummy_audio, sr=sr)

    assert "human_probability" in result
    assert "synthetic_probability" in result
    assert "confidence" in result

    assert 0.0 <= result["human_probability"] <= 1.0
    assert 0.0 <= result["synthetic_probability"] <= 1.0
    assert abs((result["human_probability"] + result["synthetic_probability"]) - 1.0) < 1e-4

def test_risk_scorer():
    low_risk = risk_scorer.calculate_risk(0.15)
    assert low_risk["risk_score"] == 15
    assert low_risk["risk_level"] == "LOW"
    assert low_risk["classification"] == "likely_human"

    high_risk = risk_scorer.calculate_risk(0.85)
    assert high_risk["risk_score"] == 85
    assert high_risk["risk_level"] == "VERY HIGH"
    assert high_risk["classification"] == "possibly_synthetic"

def test_temporal_analysis():
    window_results = [
        {"synthetic_probability": 0.20},
        {"synthetic_probability": 0.25},
        {"synthetic_probability": 0.75},
        {"synthetic_probability": 0.82}
    ]
    eval_result = risk_scorer.analyze_temporal_windows(window_results)

    assert eval_result["total_windows"] == 4
    assert eval_result["suspicious_windows"] == 2
    assert eval_result["trend"] == "RISING"
