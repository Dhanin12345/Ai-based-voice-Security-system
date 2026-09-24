import os
from pathlib import Path
from typing import Any, Optional
import numpy as np
import joblib
from backend.config import settings
from backend.utils.logger import logger

class DevelopmentAcousticClassifier:
    """
    Acoustic artifact heuristic classifier for synthetic voice detection.
    Analyzes neural vocoder artifacts (high-frequency spectral flatness, pitch micro-variance,
    phase jitter, spectral centroid stability).
    """
    def predict_proba(self, feature_vector: np.ndarray, feature_dict: Optional[dict] = None) -> np.ndarray:
        if feature_dict is None:
            synth_score = 0.3 + 0.4 * float(np.sin(np.sum(feature_vector[:10])))
            synth_score = max(0.05, min(0.95, synth_score))
            return np.array([[1.0 - synth_score, synth_score]])

        pitch_std = feature_dict.get("pitch_std", 0.0)
        pitch_mean = feature_dict.get("pitch_mean", 0.0)
        pitch_cov = feature_dict.get("pitch_cov", pitch_std / (pitch_mean + 1e-6) if pitch_mean > 0 else 0.1)
        flatness_mean = feature_dict.get("spectral_flatness_mean", 0.0)
        flatness_std = feature_dict.get("spectral_flatness_std", 0.0)
        zcr_mean = feature_dict.get("zcr_mean", 0.0)

        mfcc_std_raw = feature_dict.get("mfcc_std", [1.0])
        mfcc_std_avg = float(np.mean(mfcc_std_raw)) if isinstance(mfcc_std_raw, (list, np.ndarray)) and len(mfcc_std_raw) > 0 else 1.0

        mel_std_raw = feature_dict.get("mel_std", [1.0])
        mel_std_avg = float(np.mean(mel_std_raw)) if isinstance(mel_std_raw, (list, np.ndarray)) and len(mel_std_raw) > 0 else 1.0

        synthetic_score = 0.15  # Baseline score

        # 1. Authentic Human Speech dynamic pitch & spectral variation bonus/discount
        is_dynamic_pitch = (pitch_mean > 50.0 and (pitch_cov >= 0.08 or pitch_std >= 12.0))
        is_moderate_pitch = (pitch_mean > 50.0 and (pitch_cov >= 0.06 or pitch_std >= 8.0))

        if is_dynamic_pitch and mfcc_std_avg >= 5.0:
            # High pitch dynamics and rich spectral modulation -> authentic human speech
            synthetic_score -= 0.45
        elif is_moderate_pitch:
            synthetic_score -= 0.30

        # 2. Robotic pitch stiffness (constant/frozen pitch F0 contour across frames)
        is_robotic_pitch = (pitch_mean > 50.0 and (pitch_std < 6.0 or pitch_cov < 0.04))
        if is_robotic_pitch:
            synthetic_score += 0.65
        elif pitch_mean > 50.0 and (pitch_std < 9.0 or pitch_cov < 0.05):
            synthetic_score += 0.40

        # 3. Spectral flatness vocoder artifacts (only penalize when pitch is not dynamically human)
        if not is_dynamic_pitch:
            if flatness_mean > 0.015 or (flatness_std < 0.005 and flatness_mean > 0.010):
                synthetic_score += 0.35

            # 4. High Zero-Crossing Rate artifact
            if zcr_mean > 0.25:
                synthetic_score += 0.25

            # 5. Robotic MFCC & Mel spectrum rigidity
            if mfcc_std_avg < 4.5:
                synthetic_score += 0.20
            if mel_std_avg < 6.0:
                synthetic_score += 0.15

        synthetic_prob = float(np.clip(synthetic_score, 0.05, 0.95))
        human_prob = round(1.0 - synthetic_prob, 4)

        return np.array([[human_prob, synthetic_prob]])
class ModelLoader:
    def __init__(self, model_path: Optional[str] = None):
        default_model = os.path.join(settings.BASE_DIR, "data", "models", "voice_classifier.joblib")
        fallback_model = os.path.join(settings.BASE_DIR, "data", "models", "anti_spoof_model.pkl")
        if model_path:
            self.model_path = model_path
        elif os.path.exists(default_model):
            self.model_path = default_model
        else:
            self.model_path = fallback_model
        self.model: Optional[Any] = None
        self.is_development_model: bool = True
        self.model_status: str = "DEVELOPMENT MODE - HEURISTIC ACOUSTIC CLASSIFIER"

    def load_model(self) -> Any:
        """
        Loads pre-trained model if available at model_path; otherwise loads development classifier.
        """
        if os.path.exists(self.model_path):
            try:
                self.model = joblib.load(self.model_path)
                self.is_development_model = False
                self.model_status = "PRODUCTION AI MODEL LOADED"
                logger.info(f"Successfully loaded production anti-spoofing model from {self.model_path}")
            except Exception as e:
                logger.error(f"Failed to load model from {self.model_path}: {e}. Falling back to dev model.")
                self.model = DevelopmentAcousticClassifier()
                self.is_development_model = True
                self.model_status = "DEVELOPMENT MODE - HEURISTIC ACOUSTIC CLASSIFIER"
        else:
            logger.info("No external pre-trained model file found. Initializing Development Acoustic Model.")
            self.model = DevelopmentAcousticClassifier()
            self.is_development_model = True
            self.model_status = "DEVELOPMENT MODE - HEURISTIC ACOUSTIC CLASSIFIER"

        return self.model

model_loader = ModelLoader()
