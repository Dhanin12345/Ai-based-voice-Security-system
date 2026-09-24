from typing import Dict, Any, List
from backend.config import settings

class RiskScorer:
    def apply_rules(self, feature_dict: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Rule Engine: Deterministic acoustic signal rule detection (Spike, Robotic Pitch Freeze, Vocoder Noise, Clipping).
        """
        if not feature_dict:
            return {"rule_anomaly": False, "anomaly_type": "NORMAL", "rule_score": 0.0}

        rule_anomaly = False
        anomaly_type = "NORMAL"
        rule_score = 0.0

        flatness = feature_dict.get("spectral_flatness_mean", 0.0)
        pitch_std = feature_dict.get("pitch_std", 0.0)
        pitch_mean = feature_dict.get("pitch_mean", 0.0)
        pitch_cov = feature_dict.get("pitch_cov", pitch_std / (pitch_mean + 1e-6) if pitch_mean > 0 else 0.1)
        zcr = feature_dict.get("zcr_mean", 0.0)

        # Rule 1: High Spectral Flatness (Vocoder Artifact / Synthetic Noise Spike)
        if flatness > 0.02 and pitch_cov < 0.06:
            rule_anomaly = True
            anomaly_type = "VOCODER_NOISE_SPIKE"
            rule_score = max(rule_score, 0.88)

        # Rule 2: Unnatural Pitch Freeze / Rigid F0 Contour (Robotic TTS & Voice Changer)
        if pitch_mean > 50.0 and (pitch_std < 4.0 and pitch_cov < 0.035):
            rule_anomaly = True
            anomaly_type = "ROBOTIC_PITCH_FREEZE"
            rule_score = max(rule_score, 0.92)

        # Rule 3: High Zero Crossing Rate (Synthetic High-Frequency Hissing / Vocoder Artifact)
        if zcr > 0.25 and pitch_cov < 0.06:
            rule_anomaly = True
            anomaly_type = "SYNTHETIC_ZCR_SPIKE"
            rule_score = max(rule_score, 0.85)

        return {
            "rule_anomaly": rule_anomaly,
            "anomaly_type": anomaly_type,
            "rule_score": rule_score
        }

    def combine_ml_and_rules(self, ml_synthetic_prob: float, rule_info: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Combines ML Model prediction with Rule Engine evaluation to produce final Anomaly Score, Confidence %, and Severity.
        """
        rule_info = rule_info or {"rule_anomaly": False, "anomaly_type": "NORMAL", "rule_score": 0.0}
        rule_score = rule_info.get("rule_score", 0.0)
        rule_anomaly = rule_info.get("rule_anomaly", False)

        if rule_anomaly:
            combined_score = max(ml_synthetic_prob, rule_score)
        else:
            combined_score = ml_synthetic_prob

        confidence = round(combined_score * 100, 2)
        risk_score = int(round(combined_score * 100))

        if combined_score >= 0.80 or (rule_anomaly and confidence >= 85):
            severity = "CRITICAL"
            level = "VERY HIGH"
            classification = "possibly_synthetic"
        elif combined_score >= 0.65:
            severity = "HIGH"
            level = "HIGH"
            classification = "possibly_synthetic"
        elif combined_score >= 0.35:
            severity = "WARNING"
            level = "MODERATE"
            classification = "likely_human"
        else:
            severity = "NORMAL"
            level = "LOW"
            classification = "likely_human"

        recommendation = self.get_recommendation(risk_score, level)

        return {
            "risk_score": risk_score,
            "risk_level": level,
            "severity": severity,
            "confidence": confidence,
            "classification": classification,
            "rule_anomaly": rule_anomaly,
            "anomaly_type": rule_info.get("anomaly_type", "NORMAL"),
            "recommendation": recommendation
        }

    def calculate_risk(self, synthetic_probability: float, feature_dict: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Calculates risk score (0-100) and risk level classification using ML + Rule Engine fusion.
        """
        rule_info = self.apply_rules(feature_dict)
        return self.combine_ml_and_rules(synthetic_probability, rule_info)

    def analyze_temporal_windows(self, window_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyzes sequence of temporal window results to detect continuous suspicious patterns.
        """
        if not window_results:
            return {
                "total_windows": 0,
                "suspicious_windows": 0,
                "average_synthetic_probability": 0.0,
                "trend": "STABLE"
            }

        synth_probs = [w["synthetic_probability"] for w in window_results]
        avg_prob = float(sum(synth_probs) / len(synth_probs))
        
        # Window considered suspicious if risk_score >= 60 (or synthetic_prob >= 0.60)
        suspicious_count = sum(1 for p in synth_probs if p >= 0.60)

        # Estimate trend direction across sequence
        if len(synth_probs) >= 3:
            first_half = synth_probs[:len(synth_probs)//2]
            second_half = synth_probs[len(synth_probs)//2:]
            diff = sum(second_half)/len(second_half) - sum(first_half)/len(first_half)
            if diff > 0.15:
                trend = "RISING"
            elif diff < -0.15:
                trend = "DECREASING"
            else:
                trend = "STABLE"
        else:
            trend = "STABLE"

        return {
            "total_windows": len(window_results),
            "suspicious_windows": suspicious_count,
            "average_synthetic_probability": avg_prob,
            "trend": trend
        }

    def get_recommendation(self, risk_score: int, risk_level: str) -> str:
        """
        Returns security recommendation based on risk evaluation.
        """
        if risk_level in ["HIGH", "VERY HIGH"]:
            return (
                "CRITICAL WARNING: Potential synthetic or voice-cloned speech detected. "
                "1. Immediately verify the caller through an independent secondary communication channel (e.g. callback). "
                "2. Ask a unique challenge-response verification question. "
                "3. Do NOT disclose sensitive credentials, OTPs, or personal data. "
                "4. Do NOT approve financial transactions based solely on voice authorization."
            )
        elif risk_level == "MODERATE":
            return (
                "ATTENTION: Elevated acoustic distortion detected. "
                "Exercise caution before sharing sensitive information. Verify identity if performing high-risk actions."
            )
        else:
            return "Voice analysis indicates standard acoustic characteristics. Continue normal security procedures."

risk_scorer = RiskScorer()
