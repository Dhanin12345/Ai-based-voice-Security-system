from typing import Dict, Any, Tuple, Optional
import numpy as np
from backend.services.audio_processor import audio_processor
from backend.services.feature_extractor import feature_extractor
from backend.ml.inference import inference_engine
from backend.ml.model_loader import model_loader

class VoiceCloneDetector:
    """
    Standard modular Voice Clone Detector Interface.
    Integrates audio preprocessing, feature extraction, ML inference, and confidence estimation.
    """
    def __init__(self):
        self.model = self.load_model()
        self.last_confidence = 0.0

    def load_model(self):
        return model_loader.load_model()

    def preprocess(self, audio: np.ndarray, sr: int = None) -> np.ndarray:
        return audio_processor.preprocess_audio(audio, sr=sr)

    def extract_features(self, audio: np.ndarray, sr: int = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        features_dict = feature_extractor.extract_features(audio, sr=sr)
        feature_vector = feature_extractor.feature_dict_to_vector(features_dict)
        return feature_vector, features_dict

    def predict(self, audio: np.ndarray, sr: int = None) -> Dict[str, float]:
        """
        Main end-to-end detection method for a single audio buffer.
        """
        processed_audio = self.preprocess(audio, sr=sr)
        feature_vector, features_dict = self.extract_features(processed_audio, sr=sr)
        
        human_prob, synthetic_prob = inference_engine.predict(feature_vector, feature_dict=features_dict)
        
        # Calculate confidence metric: max probability scaled
        self.last_confidence = max(human_prob, synthetic_prob)

        return {
            "human_probability": human_prob,
            "synthetic_probability": synthetic_prob,
            "confidence": self.last_confidence,
            "features_dict": features_dict
        }

    def get_confidence(self) -> float:
        return self.last_confidence

voice_clone_detector = VoiceCloneDetector()
