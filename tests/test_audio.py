import io
import pytest
import numpy as np
import soundfile as sf

from backend.services.audio_processor import audio_processor
from backend.services.feature_extractor import feature_extractor

def create_dummy_wav_bytes(duration_sec: float = 3.0, sr: int = 16000, freq: float = 440.0) -> bytes:
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    audio = (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format='WAV')
    buf.seek(0)
    return buf.read()

def test_load_and_preprocess_audio():
    wav_bytes = create_dummy_wav_bytes(duration_sec=3.0, sr=16000)
    audio, sr = audio_processor.load_audio_file(wav_bytes)
    
    assert sr == 16000
    assert len(audio) > 0

    processed = audio_processor.preprocess_audio(audio, sr=sr)
    assert isinstance(processed, np.ndarray)
    assert len(processed) > 0
    assert np.max(np.abs(processed)) <= 1.0

def test_create_windows():
    sr = 16000
    dummy_signal = np.random.randn(sr * 5).astype(np.float32)
    windows = audio_processor.create_windows(dummy_signal, sr=sr, window_seconds=2.0)
    
    assert len(windows) > 1
    w_audio, start_t, end_t = windows[0]
    assert len(w_audio) == int(sr * 2.0)
    assert start_t == 0.0
    assert end_t == 2.0

def test_feature_extraction():
    sr = 16000
    dummy_signal = np.random.randn(sr * 2).astype(np.float32)
    features_dict = feature_extractor.extract_features(dummy_signal, sr=sr)
    
    assert "mfcc_mean" in features_dict
    assert "mel_mean" in features_dict
    assert "spectral_centroid_mean" in features_dict
    assert "zcr_mean" in features_dict
    assert "energy_mean" in features_dict

    vector = feature_extractor.feature_dict_to_vector(features_dict)
    assert isinstance(vector, np.ndarray)
    assert vector.ndim == 1
    assert len(vector) == 44
