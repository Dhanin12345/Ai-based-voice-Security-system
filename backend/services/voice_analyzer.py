import time
from typing import Dict, Any, Optional
import numpy as np

from backend.ml.predictor import predictor
from backend.services.human_voice_analysis import human_voice_analysis
from backend.services.synthetic_voice_analysis import synthetic_voice_analysis
from backend.services.risk_scorer import risk_scorer
from backend.utils.logger import logger

class VoiceAnalyzer:
    """
    Combined Voice Analyzer Service.
    Integrates Human Voice Analysis and Robot / Synthetic Voice Analysis modules into a unified payload.
    Supports confidence threshold evaluation (voice_type="UNCERTAIN" when confidence < 0.55).
    """
    def __init__(self, confidence_threshold: float = 0.55):
        self.confidence_threshold = confidence_threshold

    def analyze_audio(self, audio: np.ndarray, sr: int = 16000) -> Dict[str, Any]:
        start_t = time.time()
        
        # 1. Run predictor pipeline
        pred_res = predictor.predict_audio(audio, sr=sr)
        human_prob = float(pred_res["human_probability"])
        synth_prob = float(pred_res["synthetic_probability"])
        features_dict = pred_res["features_dict"]

        # 2. Run dedicated Human Voice Analysis module
        h_analysis = human_voice_analysis.analyze(features_dict, human_prob)

        # 3. Run dedicated Robot / Synthetic Voice Analysis module
        s_analysis = synthetic_voice_analysis.analyze(features_dict, synth_prob)

        # 4. Calculate confidence & risk score
        confidence = float(np.max([human_prob, synth_prob]))
        risk_score = int(round(synth_prob * 100))
        clone_probability = round(synth_prob * (risk_score / 100.0), 4)

        # 5. Exact 65% threshold classification rule for Human Voice:
        classification = pred_res["classification"]
        human_confidence = pred_res["human_confidence"]
        ai_confidence = pred_res["ai_confidence"]
        risk = pred_res["risk"]
        color = pred_res["color"]
        danger_alert = pred_res["danger_alert"]
        voice_type = pred_res["voice_type"]
        status = pred_res["status"]

        if classification == "HUMAN VOICE":
            risk_score = min(risk_score, 15)
            impersonation_status = "SAFE"
            prevention_action = "CONTINUE MONITORING"
            recommendation = "Voice analysis indicates authentic human speech. Continue standard monitoring."
        else:
            risk_score = max(risk_score, 75)
            impersonation_status = "HIGH-RISK IMPERSONATION"
            prevention_action = "PROMINENT WARNING - REQUEST SECONDARY IDENTITY VERIFICATION"
            recommendation = pred_res.get("recommendation") or "CRITICAL WARNING: Potential synthetic or voice-cloned speech detected. Verify speaker via callback."

        # 6. Build direct comparison matrix (Section 8: Human vs Synthetic Comparison)
        from backend.services.human_analysis import human_dataset_analysis
        from backend.services.synthetic_analysis import synthetic_dataset_analysis
        h_baseline = human_dataset_analysis.get_analytics()
        s_baseline = synthetic_dataset_analysis.get_analytics()

        features_comparison = {}
        compare_keys = [
            ("pitch_mean", "average_pitch_hz", " Hz"),
            ("pitch_std", "pitch_variation_hz", " Hz"),
            ("rms_energy", "average_energy", ""),
            ("zcr_mean", "average_zcr", ""),
            ("spectral_centroid", "average_spectral_centroid_hz", " Hz"),
            ("spectral_bandwidth", "average_spectral_bandwidth_hz", " Hz"),
            ("spectral_rolloff", "average_spectral_rolloff_hz", " Hz"),
            ("harmonicity", None, ""),
            ("speech_duration", "average_duration_sec", " sec"),
            ("silence_ratio", "silence_ratio", "%"),
            ("mfcc_mean", "mfcc_overall_mean", "")
        ]

        for feat_key, base_key, unit in compare_keys:
            val_h = float(h_analysis["human_features"].get(feat_key, 0.0))
            val_s = float(s_analysis["synthetic_features"].get(feat_key, 0.0))
            ref_h = float(h_baseline.get(base_key, val_h)) if base_key else val_h
            ref_s = float(s_baseline.get(base_key, val_s)) if base_key else val_s

            diff = round(val_h - val_s, 4)
            pct_diff = round(((val_h - val_s) / (abs(val_s) + 1e-6)) * 100, 1)

            features_comparison[feat_key] = {
                "human": val_h,
                "synthetic": val_s,
                "dataset_human_ref": ref_h,
                "dataset_synthetic_ref": ref_s,
                "difference": diff,
                "pct_difference": pct_diff,
                "unit": unit
            }

        proc_time_ms = round((time.time() - start_t) * 1000, 2)

        return {
            "voice_type": voice_type,
            "classification": classification,
            "human_probability": round(human_prob, 4),
            "synthetic_probability": round(synth_prob, 4),
            "human_confidence": human_confidence,
            "ai_confidence": ai_confidence,
            "risk": risk,
            "color": color,
            "danger_alert": danger_alert,
            "clone_probability": clone_probability,
            "confidence": round(confidence, 4),
            "risk_score": risk_score,
            "status": status,
            "impersonation_status": impersonation_status,
            "prevention_action": prevention_action,
            "speech_detected": features_dict.get("speech_detected", True),
            "speech_duration": features_dict.get("speech_duration", 0.0),
            "silence_ratio": features_dict.get("silence_ratio", 0.0),
            "signal_quality": features_dict.get("signal_quality", "GOOD"),
            "human_analysis": {
                "classification": "HUMAN",
                "probability": round(human_prob, 4),
                "confidence": round(human_prob, 4),
                "features": h_analysis["human_features"],
                "evidence": h_analysis["human_evidence"]
            },
            "synthetic_analysis": {
                "classification": "SYNTHETIC / AI-GENERATED",
                "probability": round(synth_prob, 4),
                "confidence": round(synth_prob, 4),
                "features": s_analysis["synthetic_features"],
                "evidence": s_analysis["synthetic_evidence"]
            },
            "features_comparison": features_comparison,
            "processing_time_ms": proc_time_ms,
            "recommendation": recommendation,
            "model_status": pred_res.get("model_status", "READY"),
            "features_dict": features_dict
        }

voice_analyzer = VoiceAnalyzer()
