import numpy as np
import librosa
import scipy.signal
from typing import List, Union
from backend.utils.logger import logger
from backend.config import settings

class VoiceEmbeddingService:
    """
    Extracts compact acoustic voice embeddings and performs cosine similarity comparison
    for the Unknown Caller Repeated Voice Protection module.
    """
    def __init__(self, target_sr: int = 16000, n_mfcc: int = 40):
        self.target_sr = target_sr or settings.SAMPLE_RATE
        self.n_mfcc = n_mfcc

    def extract_embedding(self, audio: np.ndarray, sr: int = None) -> List[float]:
        """
        Extracts a unit-normalized 152-dimensional acoustic voice embedding from audio waveform.
        Features combine MFCC statistics, Mel energy distribution, spectral envelope, and pitch.
        """
        sr = sr or self.target_sr

        # Handle empty/short audio safely
        if len(audio) < 512:
            audio = np.pad(audio, (0, 512 - len(audio)), mode='constant')

        try:
            # Fast STFT
            _, _, Zxx = scipy.signal.stft(audio, fs=sr, nperseg=2048, noverlap=2048 - 512)
            S = np.abs(Zxx)
            power_spec = librosa.power_to_db(S**2 + 1e-9)

            # 1. MFCC statistics excluding c0 energy coefficient (1..39 = 39 * 3 = 117 features)
            mfcc = librosa.feature.mfcc(S=power_spec, sr=sr, n_mfcc=self.n_mfcc)[1:]
            mfcc_mean = np.mean(mfcc, axis=1)
            mfcc_std = np.std(mfcc, axis=1)
            mfcc_delta = np.mean(librosa.feature.delta(mfcc), axis=1)

            v_mfcc = np.concatenate([mfcc_mean, mfcc_std, mfcc_delta])
            v_mfcc = v_mfcc / (np.linalg.norm(v_mfcc) + 1e-9)

            # 2. Mel Filterbank distribution with mean spectral energy subtracted (24 features)
            mel = librosa.feature.melspectrogram(S=S**2, sr=sr, n_mels=24)
            mel_mean = np.mean(librosa.power_to_db(mel + 1e-9), axis=1)
            mel_mean = mel_mean - np.mean(mel_mean)
            v_mel = mel_mean / (np.linalg.norm(mel_mean) + 1e-9)

            # 3. Spectral dynamics normalized (6 features)
            centroid = librosa.feature.spectral_centroid(S=S, sr=sr) / (sr / 2.0)
            rolloff = librosa.feature.spectral_rolloff(S=S, sr=sr) / (sr / 2.0)
            flatness = librosa.feature.spectral_flatness(S=S)

            v_spec = np.array([
                float(np.mean(centroid)),
                float(np.std(centroid)),
                float(np.mean(rolloff)),
                float(np.std(rolloff)),
                float(np.mean(flatness)),
                float(np.std(flatness)),
            ])
            v_spec = v_spec / (np.linalg.norm(v_spec) + 1e-9)

            # 4. Pitch dynamics normalized (2 features)
            try:
                f0, _, _ = librosa.pyin(audio, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr)
                valid_f0 = f0[~np.isnan(f0)]
                if len(valid_f0) > 0:
                    pitch_stats = np.array([float(np.mean(valid_f0)) / 500.0, float(np.std(valid_f0)) / 100.0])
                else:
                    pitch_stats = np.array([0.32, 0.25])
            except Exception:
                pitch_stats = np.array([0.32, 0.25])

            v_pitch = pitch_stats / (np.linalg.norm(pitch_stats) + 1e-9)

            # Concatenate all acoustic vectors into single representation (117 + 24 + 6 + 2 = 149 features)
            raw_embedding = np.concatenate([
                v_mfcc,
                v_mel,
                v_spec,
                v_pitch
            ]).astype(np.float64)

            # L2 Normalization so dot product is exact cosine similarity
            norm = np.linalg.norm(raw_embedding)
            if norm > 1e-12:
                normalized_embedding = raw_embedding / norm
            else:
                normalized_embedding = raw_embedding

            return [round(float(x), 6) for x in normalized_embedding]

        except Exception as e:
            logger.error(f"Error extracting voice embedding: {e}")
            # Fallback zero vector
            return [0.0] * 149

    @staticmethod
    def compute_cosine_similarity(vec1: Union[List[float], np.ndarray], vec2: Union[List[float], np.ndarray]) -> float:
        """
        Computes cosine similarity between two voice embeddings.
        Returns percentage score between 0.0% and 100.0%.
        """
        u = np.array(vec1, dtype=np.float64)
        v = np.array(vec2, dtype=np.float64)

        if len(u) != len(v) or len(u) == 0:
            return 0.0

        norm_u = np.linalg.norm(u)
        norm_v = np.linalg.norm(v)

        if norm_u < 1e-12 or norm_v < 1e-12:
            return 0.0

        dot_prod = np.dot(u, v)
        cos_sim = float(dot_prod / (norm_u * norm_v))
        # Clip to [0, 1] range for acoustic similarity
        cos_sim = max(0.0, min(1.0, cos_sim))
        return round(cos_sim * 100.0, 2)

voice_embedding_service = VoiceEmbeddingService()
