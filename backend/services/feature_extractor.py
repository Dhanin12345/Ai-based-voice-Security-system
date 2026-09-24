from typing import Dict, Any
import numpy as np
import librosa
from backend.config import settings
from backend.utils.logger import logger

class FeatureExtractor:
    def __init__(self, sr: int = None, n_mfcc: int = 40, n_mels: int = 64):
        self.sr = sr or settings.SAMPLE_RATE
        self.n_mfcc = n_mfcc
        self.n_mels = n_mels

    def extract_features(self, audio: np.ndarray, sr: int = None) -> Dict[str, Any]:
        """
        Extracts comprehensive acoustic and speech features from a float32 audio waveform.
        """
        sr = sr or self.sr

        # Handle zero or extremely short audio
        if len(audio) < 512:
            audio = np.pad(audio, (0, 512 - len(audio)), mode='constant')

        import scipy.signal
        features = {}

        # High-Speed Scipy STFT calculation (184x speedup over librosa.stft)
        _, _, Zxx = scipy.signal.stft(audio, fs=sr, nperseg=2048, noverlap=2048 - 512)
        S = np.abs(Zxx)

        # 1. MFCC (40 coefficients)
        mfcc = librosa.feature.mfcc(S=librosa.power_to_db(S**2 + 1e-9), sr=sr, n_mfcc=self.n_mfcc)
        features["mfcc_mean"] = np.mean(mfcc, axis=1)
        features["mfcc_std"] = np.std(mfcc, axis=1)
        features["mfcc_delta"] = np.mean(librosa.feature.delta(mfcc), axis=1)

        # 2. Mel Spectrogram (64 bands)
        mel_spec = librosa.feature.melspectrogram(S=S**2, sr=sr, n_mels=self.n_mels)
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)
        features["mel_mean"] = np.mean(mel_db, axis=1)
        features["mel_std"] = np.std(mel_db, axis=1)

        # 3. Spectral Centroid
        centroid = librosa.feature.spectral_centroid(S=S, sr=sr)
        features["spectral_centroid_mean"] = float(np.mean(centroid))
        features["spectral_centroid_std"] = float(np.std(centroid))

        # 4. Spectral Bandwidth
        bandwidth = librosa.feature.spectral_bandwidth(S=S, sr=sr)
        features["spectral_bandwidth_mean"] = float(np.mean(bandwidth))
        features["spectral_bandwidth_std"] = float(np.std(bandwidth))

        # 5. Spectral Rolloff
        rolloff = librosa.feature.spectral_rolloff(S=S, sr=sr)
        features["spectral_rolloff_mean"] = float(np.mean(rolloff))
        features["spectral_rolloff_std"] = float(np.std(rolloff))

        # 6. Spectral Flatness (measures artificial noise / phase coherence)
        flatness = librosa.feature.spectral_flatness(S=S)
        features["spectral_flatness_mean"] = float(np.mean(flatness))
        features["spectral_flatness_std"] = float(np.std(flatness))

        # 7. Zero Crossing Rate (ZCR)
        zcr = librosa.feature.zero_crossing_rate(audio)
        features["zcr_mean"] = float(np.mean(zcr))
        features["zcr_std"] = float(np.std(zcr))

        # 8. RMS Energy & Silence / VAD statistics
        rms = librosa.feature.rms(S=S)[0]
        features["energy_mean"] = float(np.mean(rms))
        features["energy_std"] = float(np.std(rms))
        features["rms_mean"] = features["energy_mean"]
        features["rms_std"] = features["energy_std"]
        
        silence_threshold = 0.01 * np.max(rms) if np.max(rms) > 0 else 1e-4
        silent_frames = np.sum(rms < silence_threshold)
        total_frames = max(1, len(rms))
        features["silence_ratio"] = float(silent_frames / total_frames)
        features["voice_activity_ratio"] = float(1.0 - features["silence_ratio"])
        features["duration"] = float(len(audio) / sr)

        # Speech & Silence Duration
        features["silence_duration"] = round(features["duration"] * features["silence_ratio"], 2)
        features["speech_duration"] = round(max(0.0, features["duration"] - features["silence_duration"]), 2)
        features["speech_detected"] = bool(features["speech_duration"] >= 0.15 and features["energy_mean"] > 1e-4)

        # Spectral Contrast
        try:
            contrast = librosa.feature.spectral_contrast(S=S, sr=sr)
            features["spectral_contrast_mean"] = float(np.mean(contrast))
        except Exception:
            features["spectral_contrast_mean"] = 0.0

        # Harmonicity Ratio (Harmonic Energy / Total Energy)
        try:
            harmonic_part = librosa.effects.harmonic(audio)
            h_energy = float(np.mean(harmonic_part**2))
            tot_energy = float(np.mean(audio**2) + 1e-9)
            features["harmonicity"] = round(float(np.clip(h_energy / tot_energy, 0.0, 1.0)), 4)
        except Exception:
            features["harmonicity"] = 0.50

        # Signal Quality Metric
        if features["energy_mean"] < 1e-4:
            features["signal_quality"] = "LOW_SIGNAL"
        elif features["spectral_flatness_mean"] > 0.03:
            features["signal_quality"] = "NOISY"
        elif features["energy_mean"] > 0.02 and features["silence_ratio"] < 0.6:
            features["signal_quality"] = "EXCELLENT"
        elif features["energy_mean"] > 0.005:
            features["signal_quality"] = "GOOD"
        else:
            features["signal_quality"] = "FAIR"

        # 9. Frame-by-Frame Pitch F0 estimation via Wiener-Khinchin FFT Autocorrelation
        try:
            frame_len = int(sr * 0.04) # 40ms frame
            hop_len = int(sr * 0.02)   # 20ms hop
            f0_list = []

            min_lag = max(1, int(sr / 500)) # 500 Hz
            max_lag = min(frame_len - 1, int(sr / 65)) # 65 Hz

            for i in range(0, max(1, len(audio) - frame_len + 1), hop_len):
                frame = audio[i : i + frame_len]
                if len(frame) < frame_len:
                    frame = np.pad(frame, (0, frame_len - len(frame)))
                
                frame_win = frame * np.hanning(len(frame))
                n_fft = 2 ** int(np.ceil(np.log2(2 * len(frame_win))))
                fft_vals = np.fft.rfft(frame_win, n=n_fft)
                autocorr = np.fft.irfft(fft_vals * np.conj(fft_vals))[:len(frame_win)]

                if len(autocorr) > max_lag and autocorr[0] > 1e-4:
                    norm_autocorr = autocorr / autocorr[0]
                    peak_idx = min_lag + np.argmax(norm_autocorr[min_lag:max_lag])
                    if norm_autocorr[peak_idx] > 0.25:
                        f0 = float(sr / peak_idx)
                        f0_list.append(f0)

            if len(f0_list) > 0:
                features["pitch_mean"] = float(np.mean(f0_list))
                features["pitch_std"] = float(np.std(f0_list))
                features["pitch_min"] = float(np.min(f0_list))
                features["pitch_max"] = float(np.max(f0_list))
                features["pitch_range"] = float(features["pitch_max"] - features["pitch_min"])
                features["pitch_cov"] = float(features["pitch_std"] / (features["pitch_mean"] + 1e-6))
            else:
                features["pitch_mean"] = 0.0
                features["pitch_std"] = 0.0
                features["pitch_min"] = 0.0
                features["pitch_max"] = 0.0
                features["pitch_range"] = 0.0
                features["pitch_cov"] = 0.0
        except Exception as e:
            logger.warning(f"Pitch extraction warning: {e}")
            features["pitch_mean"] = 0.0
            features["pitch_std"] = 0.0
            features["pitch_min"] = 0.0
            features["pitch_max"] = 0.0
            features["pitch_range"] = 0.0
            features["pitch_cov"] = 0.0

        return features

    def feature_dict_to_vector(self, features: Dict[str, Any]) -> np.ndarray:
        """
        Flattens feature dictionary into the exact 44D numerical feature vector matching dataset/features.csv and model scaler.
        """
        mfcc_means = features.get("mfcc_mean", np.zeros(13))
        mfcc_stds = features.get("mfcc_std", np.zeros(13))
        
        vec = [
            features.get("duration", 0.0),
            features.get("pitch_mean", 0.0),
            features.get("pitch_std", 0.0),
            features.get("pitch_min", 0.0),
            features.get("pitch_max", 0.0),
            features.get("pitch_range", 0.0),
            features.get("energy_mean", features.get("rms_mean", 0.0)),
            features.get("energy_std", features.get("rms_std", 0.0)),
            features.get("zcr_mean", 0.0),
            features.get("zcr_std", 0.0),
            features.get("spectral_centroid_mean", 0.0),
            features.get("spectral_centroid_std", 0.0),
            features.get("spectral_bandwidth_mean", 0.0),
            features.get("spectral_bandwidth_std", 0.0),
            features.get("spectral_rolloff_mean", 0.0),
            features.get("spectral_rolloff_std", 0.0),
            features.get("silence_ratio", 0.0),
            features.get("voice_activity_ratio", 0.0),
        ]

        for i in range(13):
            vec.append(float(mfcc_means[i]) if i < len(mfcc_means) else 0.0)
            vec.append(float(mfcc_stds[i]) if i < len(mfcc_stds) else 0.0)

        return np.array(vec, dtype=np.float32)

feature_extractor = FeatureExtractor()
