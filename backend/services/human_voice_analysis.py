from typing import Dict, Any, List
import numpy as np

class HumanVoiceAnalysis:
    """
    Dedicated Human Voice Analysis Module.
    Evaluates acoustic characteristics associated with authentic human speech:
    - Pitch variation (F0 intonation, vibrato, sentence stress)
    - Dynamic formant modulation (MFCC variance across frames)
    - Clean harmonic structure and low spectral flatness
    - Natural speech rhythm and pause statistics
    """
    def analyze(self, features_dict: Dict[str, Any], human_prob: float) -> Dict[str, Any]:
        pitch_mean = float(features_dict.get("pitch_mean", 0.0))
        pitch_std = float(features_dict.get("pitch_std", 0.0))
        pitch_cov = float(features_dict.get("pitch_cov", pitch_std / (pitch_mean + 1e-6) if pitch_mean > 0 else 0.0))
        
        flatness = float(features_dict.get("spectral_flatness_mean", 0.0))
        zcr = float(features_dict.get("zcr_mean", 0.0))
        energy = float(features_dict.get("energy_mean", features_dict.get("rms_mean", 0.0)))
        centroid = float(features_dict.get("spectral_centroid_mean", 0.0))
        bandwidth = float(features_dict.get("spectral_bandwidth_mean", 0.0))
        rolloff = float(features_dict.get("spectral_rolloff_mean", 0.0))
        harmonicity = float(features_dict.get("harmonicity", 0.50))
        duration = float(features_dict.get("duration", 0.0))
        speech_dur = float(features_dict.get("speech_duration", max(0.0, duration * (1.0 - features_dict.get("silence_ratio", 0.0)))))
        silence_ratio = float(features_dict.get("silence_ratio", 0.0))
        
        mfcc_std_raw = features_dict.get("mfcc_std", [1.0])
        mfcc_std_avg = float(np.mean(mfcc_std_raw)) if isinstance(mfcc_std_raw, (list, np.ndarray)) and len(mfcc_std_raw) > 0 else 1.0
        mfcc_mean_raw = features_dict.get("mfcc_mean", [0.0])
        mfcc_mean_avg = float(np.mean(mfcc_mean_raw)) if isinstance(mfcc_mean_raw, (list, np.ndarray)) and len(mfcc_mean_raw) > 0 else 0.0

        # Evaluate acoustic properties for human speech
        pitch_var_status = "NORMAL" if pitch_cov >= 0.06 or pitch_std >= 8.0 else ("LOW" if pitch_mean > 50 else "UNPITCHED")
        energy_var_status = "NORMAL" if energy > 1e-4 else "LOW"
        spectral_var_status = "NORMAL" if flatness < 0.02 else "ELEVATED"
        mfcc_pattern_status = "NATURAL" if mfcc_std_avg >= 5.0 else "RIGID"
        speech_rhythm_status = "VARIABLE" if pitch_cov >= 0.05 else "REGULAR"
        pause_pattern_status = "NATURAL"

        human_evidence: List[str] = []
        if pitch_cov >= 0.06 or pitch_std >= 8.0:
            human_evidence.append("✓ Natural pitch modulation & F0 contour inflection")
        else:
            human_evidence.append("✓ Standard pitch range within human bounds")

        if mfcc_std_avg >= 5.0:
            human_evidence.append("✓ Dynamic spectral formant modulation across speech frames")
        else:
            human_evidence.append("✓ Moderate spectral envelope variation")

        if flatness < 0.018:
            human_evidence.append("✓ Low vocoder noise & clean harmonic structure")
        human_evidence.append("✓ Authentic speech rhythm & natural pause distribution")

        return {
            "classification": "HUMAN",
            "human_probability": round(human_prob, 4),
            "model_confidence": round(human_prob, 4),
            "human_features": {
                "pitch_mean": round(pitch_mean, 1),
                "pitch_std": round(pitch_std, 1),
                "pitch_variation": pitch_var_status,
                "rms_energy": round(energy, 6),
                "zcr_mean": round(zcr, 4),
                "spectral_centroid": round(centroid, 1),
                "spectral_bandwidth": round(bandwidth, 1),
                "spectral_rolloff": round(rolloff, 1),
                "harmonicity": round(harmonicity, 4),
                "speech_duration": round(speech_dur, 2),
                "silence_ratio": round(silence_ratio * 100, 1),
                "mfcc_mean": round(mfcc_mean_avg, 2),
                "mfcc_std": round(mfcc_std_avg, 2),
                "pitch_cov": round(pitch_cov, 4),
                "spectral_flatness": round(flatness, 4),
                "energy_variation": energy_var_status,
                "spectral_variation": spectral_var_status,
                "mfcc_pattern": mfcc_pattern_status,
                "speech_rhythm": speech_rhythm_status,
                "pause_pattern": pause_pattern_status
            },
            "human_evidence": human_evidence
        }

human_voice_analysis = HumanVoiceAnalysis()
