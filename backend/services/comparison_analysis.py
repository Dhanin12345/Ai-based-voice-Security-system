from typing import Dict, Any
from backend.services.human_analysis import human_dataset_analysis
from backend.services.synthetic_analysis import synthetic_dataset_analysis

class ComparisonAnalysis:
    """
    Comparison Analytics Service.
    Compares statistical feature metrics between Human Voice and Robot/Synthetic Voice classes:
    - Pitch Mean & Std (Variation)
    - RMS Energy Mean & Std
    - Zero Crossing Rate (ZCR)
    - Spectral Centroid, Bandwidth, and Rolloff
    - MFCC Means & Duration
    """
    def get_comparison(self, db=None) -> Dict[str, Any]:
        h_stats = human_dataset_analysis.get_analytics()
        s_stats = synthetic_dataset_analysis.get_analytics()

        db_total = 0
        db_human = 0
        db_synth = 0
        if db is not None:
            try:
                from backend.models.database import DetectionHistoryDB
                db_total = db.query(DetectionHistoryDB).count()
                db_synth = db.query(DetectionHistoryDB).filter(DetectionHistoryDB.classification == "possibly_synthetic").count()
                db_human = db_total - db_synth
            except Exception:
                pass

        total_samples = h_stats["total_samples"] + s_stats["total_samples"] + db_total
        human_samples = h_stats["total_samples"] + db_human
        synthetic_samples = s_stats["total_samples"] + db_synth

        h_ratio = round((human_samples / total_samples * 100), 1) if total_samples > 0 else 0.0
        s_ratio = round((synthetic_samples / total_samples * 100), 1) if total_samples > 0 else 0.0

        def cmp(key_h, key_s=None):
            k_s = key_s or key_h
            val_h = float(h_stats.get(key_h, 0.0))
            val_s = float(s_stats.get(k_s, 0.0))
            diff = round(val_h - val_s, 6)
            return {
                "human": val_h,
                "synthetic": val_s,
                "difference": diff
            }

        return {
            "overview": {
                "total_samples": total_samples,
                "human_samples": human_samples,
                "synthetic_samples": synthetic_samples,
                "human_percentage": h_ratio,
                "synthetic_percentage": s_ratio
            },
            "comparison": {
                "pitch_mean": cmp("average_pitch_hz"),
                "pitch_std": cmp("pitch_variation_hz"),
                "rms_energy": cmp("average_energy"),
                "zcr": cmp("average_zcr"),
                "spectral_centroid": cmp("average_spectral_centroid_hz"),
                "spectral_bandwidth": cmp("average_spectral_bandwidth_hz"),
                "spectral_rolloff": cmp("average_spectral_rolloff_hz"),
                "mfcc_mean": cmp("mfcc_overall_mean"),
                "duration": cmp("average_duration_sec")
            },
            "human": h_stats,
            "synthetic": s_stats
        }

comparison_analysis = ComparisonAnalysis()
