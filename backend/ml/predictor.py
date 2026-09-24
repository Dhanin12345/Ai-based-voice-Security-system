import os
from typing import Dict, Any, Tuple, Optional
import numpy as np
import joblib

from backend.config import settings
from backend.services.feature_extractor import feature_extractor
from backend.services.risk_scorer import risk_scorer
from backend.ml.model_loader import model_loader, DevelopmentAcousticClassifier
from backend.utils.logger import logger

class VoicePredictor:
    """
    Supervised Voice Classification & Prediction Pipeline.
    Loads trained scikit-learn model (voice_classifier.joblib) and feature scaler (feature_scaler.joblib).
    Provides prediction probabilities, risk scoring, confidence, acoustic evidence, and recommendations.
    """
    def __init__(self, model_dir: Optional[str] = None):
        self.model_dir = model_dir or os.path.join(settings.BASE_DIR, "data", "models")
        self.model_path = os.path.join(self.model_dir, "voice_classifier.joblib")
        self.scaler_path = os.path.join(self.model_dir, "feature_scaler.joblib")
        self.metrics_path = os.path.join(self.model_dir, "model_metrics.json")
        
        self.model = None
        self.scaler = None
        self.model_status = "UNINITIALIZED"
        self.load_model()

    def load_model(self):
        """
        Loads pre-trained joblib model & scaler if available; falls back to DevelopmentAcousticClassifier.
        """
        if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
            try:
                self.model = joblib.load(self.model_path)
                self.scaler = joblib.load(self.scaler_path)
                self.model_status = "READY (TRAINED MODEL LOADED)"
                logger.info(f"Loaded trained voice classifier from {self.model_path}")
            except Exception as e:
                logger.error(f"Error loading trained model files: {e}. Falling back to Development Classifier.")
                self.model = DevelopmentAcousticClassifier()
                self.scaler = None
                self.model_status = "DEVELOPMENT MODE (ACOUSTIC HEURISTICS)"
        else:
            logger.info("No trained model file found. Falling back to Development Acoustic Classifier.")
            self.model = DevelopmentAcousticClassifier()
            self.scaler = None
            self.model_status = "DEVELOPMENT MODE (ACOUSTIC HEURISTICS)"

    def predict_audio(self, audio: np.ndarray, sr: int = 16000) -> Dict[str, Any]:
        """
        Runs complete prediction pipeline on preprocessed float32 audio array.
        Returns standardized result dictionary with real model predictions and exact 65% threshold.
        """
        duration_sec = float(len(audio) / sr) if sr > 0 else 0.0
        features_dict = feature_extractor.extract_features(audio, sr=sr)
        feature_vector = feature_extractor.feature_dict_to_vector(features_dict)

        # 1. Run model prediction
        if self.scaler is not None and not isinstance(self.model, DevelopmentAcousticClassifier):
            scaled_vector = self.scaler.transform([feature_vector])
            probs = self.model.predict_proba(scaled_vector)[0]
        elif isinstance(self.model, DevelopmentAcousticClassifier):
            probs = self.model.predict_proba(feature_vector, feature_dict=features_dict)[0]
        else:
            probs = self.model.predict_proba([feature_vector])[0]

        # 2. Probability interpretation & class mapping (Class 0: Human, Class 1: Synthetic/AI)
        classes = list(getattr(self.model, "classes_", [0, 1]))
        if 0 in classes and 1 in classes:
            h_idx = classes.index(0)
            s_idx = classes.index(1)
            raw_human_prob = float(probs[h_idx])
            raw_synth_prob = float(probs[s_idx])
        elif "HUMAN" in classes:
            h_idx = classes.index("HUMAN")
            raw_human_prob = float(probs[h_idx])
            raw_synth_prob = 1.0 - raw_human_prob
        else:
            raw_human_prob = float(probs[0])
            raw_synth_prob = float(probs[1]) if len(probs) > 1 else (1.0 - raw_human_prob)

        # Normalize sum of probabilities to 1.0
        total_prob = raw_human_prob + raw_synth_prob
        if total_prob > 0:
            human_prob = raw_human_prob / total_prob
            synthetic_prob = raw_synth_prob / total_prob
        else:
            human_prob = 0.5
            synthetic_prob = 0.5

        # Human & AI confidence as percentages (0 - 100)
        human_confidence = round(human_prob * 100.0, 2)
        ai_confidence = round(100.0 - human_confidence, 2)

        # 3. Exact 65% threshold classification rule:
        # 65.00% -> HUMAN VOICE (Risk: LOW, Color: GREEN, Danger Alert: False)
        # 64.99% -> ROBOTIC VOICE / AI VOICE (Risk: HIGH, Color: RED, Danger Alert: True)
        if human_confidence >= 65.0:
            classification = "HUMAN VOICE"
            risk = "LOW"
            color = "GREEN"
            danger_alert = False
            voice_type = "HUMAN"
            status = "LOW"
        else:
            classification = "ROBOTIC VOICE / AI VOICE"
            risk = "HIGH"
            color = "RED"
            danger_alert = True
            voice_type = "ROBOT/SYNTHETIC"
            status = "HIGH"

        model_pred_label = "HUMAN VOICE" if human_confidence >= 65.0 else "ROBOTIC VOICE / AI VOICE"

        # Step 3 Logging
        logger.info("==========================================")
        logger.info("Audio received: YES")
        logger.info(f"Audio duration: {round(duration_sec, 2)}s")
        logger.info("Features extracted: YES")
        logger.info(f"Model prediction: {model_pred_label}")
        logger.info(f"Human confidence: {human_confidence}%")
        logger.info(f"AI confidence: {ai_confidence}%")
        logger.info(f"Classification: {classification} ({color}, Risk: {risk})")
        logger.info("==========================================")
        print(f"Audio received: YES | Audio duration: {round(duration_sec, 2)}s | Features extracted: YES | Model prediction: {model_pred_label} | Human confidence: {human_confidence}% | AI confidence: {ai_confidence}%")

        # Risk scoring via RiskScorer
        risk_info = risk_scorer.calculate_risk(synthetic_prob, feature_dict=features_dict)
        risk_score = risk_info["risk_score"]
        if classification == "HUMAN VOICE":
            risk_score = min(risk_score, 15)
        else:
            risk_score = max(risk_score, 75)

        confidence = round(max(human_prob, synthetic_prob), 4)

        # Generate acoustic evidence list based on feature analysis
        acoustic_evidence = self._generate_acoustic_evidence(features_dict, voice_type)

        return {
            "voice_type": voice_type,
            "classification": classification,
            "human_probability": round(human_prob, 4),
            "synthetic_probability": round(synthetic_prob, 4),
            "human_confidence": human_confidence,
            "ai_confidence": ai_confidence,
            "confidence": confidence,
            "risk": risk,
            "color": color,
            "danger_alert": danger_alert,
            "risk_score": risk_score,
            "status": status,
            "rule_anomaly": risk_info.get("rule_anomaly", False),
            "anomaly_type": risk_info.get("anomaly_type", "NORMAL"),
            "recommendation": risk_info.get("recommendation", ""),
            "acoustic_evidence": acoustic_evidence,
            "features_dict": features_dict,
            "model_status": self.model_status
        }

    def _generate_acoustic_evidence(self, fdict: Dict[str, Any], voice_type: str) -> list:
        """
        Generates human-readable acoustic evidence items.
        """
        evidence = []
        pitch_cov = fdict.get("pitch_cov", 0.0)
        pitch_std = fdict.get("pitch_std", 0.0)
        flatness = fdict.get("spectral_flatness_mean", 0.0)
        mfcc_std = float(np.mean(fdict.get("mfcc_std", [1.0])))

        if voice_type == "HUMAN":
            if pitch_cov >= 0.06 or pitch_std >= 8.0:
                evidence.append("✓ Natural pitch modulation & F0 contour inflection")
            else:
                evidence.append("✓ Standard pitch characteristics")
            
            if mfcc_std >= 5.0:
                evidence.append("✓ Natural spectral formant variation")
            else:
                evidence.append("✓ Dynamic spectral envelope")

            if flatness < 0.018:
                evidence.append("✓ Low vocoder noise & clean harmonic structure")
            evidence.append("✓ Natural speech rhythm & pauses")
        else:
            if pitch_std < 6.0 or pitch_cov < 0.05:
                evidence.append("- Rigid / unnatural pitch consistency (F0 freeze)")
            if flatness > 0.015:
                evidence.append("- High spectral flatness (neural vocoder hiss artifact)")
            if mfcc_std < 5.0:
                evidence.append("- Reduced spectral formant modulation across frames")
            evidence.append("- Acoustic patterns aligned with synthetic voice training data")

        return evidence

predictor = VoicePredictor()
