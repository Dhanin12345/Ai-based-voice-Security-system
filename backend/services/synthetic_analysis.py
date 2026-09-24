import os
from typing import Dict, Any
import pandas as pd
import numpy as np
from backend.config import settings
from backend.utils.logger import logger

class SyntheticDatasetAnalysis:
    """
    Dataset-Level Robot / Synthetic Voice Analytics Service.
    Reads SYNTHETIC class records from dataset/features.csv and computes statistical averages and feature distributions.
    """
    def __init__(self, csv_path: str = None):
        self.csv_path = csv_path or os.path.join(settings.BASE_DIR, "dataset", "features.csv")

    def get_analytics(self) -> Dict[str, Any]:
        """
        Computes summary statistics for all SYNTHETIC / ROBOT voice samples.
        """
        if not os.path.exists(self.csv_path):
            from ml.build_dataset import build_features_csv
            df = build_features_csv(csv_out_path=self.csv_path)
        else:
            df = pd.read_csv(self.csv_path)

        if df is None or df.empty:
            return self._empty_summary()

        synth_df = df[df["class"].str.upper() == "SYNTHETIC"]
        if synth_df.empty:
            return self._empty_summary()

        n_samples = len(synth_df)

        def get_mean(col):
            return float(np.mean(synth_df[col])) if col in synth_df.columns else 0.0

        def get_std(col):
            return float(np.std(synth_df[col])) if col in synth_df.columns else 0.0

        mfcc_means = [get_mean(f"mfcc_{i}_mean") for i in range(1, 14)]

        return {
            "total_samples": n_samples,
            "average_pitch_hz": round(get_mean("pitch_mean"), 2),
            "pitch_variation_hz": round(get_mean("pitch_std"), 2),
            "average_energy": round(get_mean("rms_mean"), 6),
            "energy_variation": round(get_mean("rms_std"), 6),
            "average_zcr": round(get_mean("zcr_mean"), 6),
            "average_spectral_centroid_hz": round(get_mean("spectral_centroid_mean"), 2),
            "average_spectral_bandwidth_hz": round(get_mean("spectral_bandwidth_mean"), 2),
            "average_spectral_rolloff_hz": round(get_mean("spectral_rolloff_mean"), 2),
            "average_duration_sec": round(get_mean("duration"), 2),
            "silence_ratio": round(get_mean("silence_ratio"), 4),
            "voice_activity_ratio": round(get_mean("voice_activity_ratio"), 4),
            "mfcc_averages": [round(m, 2) for m in mfcc_means],
            "mfcc_overall_mean": round(float(np.mean(mfcc_means)), 2)
        }

    def _empty_summary(self) -> Dict[str, Any]:
        return {
            "total_samples": 0,
            "average_pitch_hz": 0.0,
            "pitch_variation_hz": 0.0,
            "average_energy": 0.0,
            "energy_variation": 0.0,
            "average_zcr": 0.0,
            "average_spectral_centroid_hz": 0.0,
            "average_spectral_bandwidth_hz": 0.0,
            "average_spectral_rolloff_hz": 0.0,
            "average_duration_sec": 0.0,
            "silence_ratio": 0.0,
            "voice_activity_ratio": 0.0,
            "mfcc_averages": [0.0] * 13,
            "mfcc_overall_mean": 0.0
        }

synthetic_dataset_analysis = SyntheticDatasetAnalysis()
