import io
import os
from typing import List, Tuple, Union
import numpy as np
import librosa
import soundfile as sf
from backend.config import settings
from backend.utils.logger import logger

class AudioProcessor:
    def __init__(self, target_sr: int = None):
        self.target_sr = target_sr or settings.SAMPLE_RATE

    def load_audio_file(self, file_source: Union[str, io.BytesIO, bytes], filename: str = None) -> Tuple[np.ndarray, int]:
        """
        Loads audio from file path, bytes buffer, or temporary disk file, converts to mono, and resamples to target_sr.
        Universal format support for WAV, MP3, M4A, OGG, FLAC, MP4, AAC, OPUS, 3GP, WEBM via FFmpeg decoding fallback.
        """
        import tempfile
        import subprocess
        temp_input_path = None
        temp_wav_path = None

        try:
            # 1. If file path string provided
            if isinstance(file_source, str) and os.path.exists(file_source):
                try:
                    audio, sr = librosa.load(file_source, sr=self.target_sr, mono=True)
                    return audio, sr
                except Exception:
                    temp_input_path = file_source

            # Extract extension from filename if provided
            ext = ".wav"
            if filename and "." in filename:
                ext = "." + filename.split(".")[-1].lower()

            # Get raw bytes
            if not temp_input_path:
                if isinstance(file_source, io.BytesIO):
                    raw_bytes = file_source.getvalue()
                elif isinstance(file_source, bytes):
                    raw_bytes = file_source
                else:
                    raise ValueError("Unsupported audio source type.")

                # Try loading directly via soundfile from BytesIO memory first (175x fast path for WAV/OGG/FLAC)
                try:
                    buf = io.BytesIO(raw_bytes)
                    data, sr = sf.read(buf)
                    if data.ndim > 1:
                        data = np.mean(data, axis=1)
                    if sr != self.target_sr:
                        data = librosa.resample(data, orig_sr=sr, target_sr=self.target_sr)
                    return data.astype(np.float32), self.target_sr
                except Exception:
                    pass

                # Write raw bytes to temporary input file
                with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                    tmp.write(raw_bytes)
                    temp_input_path = tmp.name

            # 2. Try librosa / soundfile directly on temp_input_path
            try:
                audio, sr = librosa.load(temp_input_path, sr=self.target_sr, mono=True)
                return audio, sr
            except Exception:
                pass

            try:
                data, sr = sf.read(temp_input_path)
                if data.ndim > 1:
                    data = np.mean(data, axis=1)
                if sr != self.target_sr:
                    data = librosa.resample(data, orig_sr=sr, target_sr=self.target_sr)
                return data.astype(np.float32), self.target_sr
            except Exception:
                pass

            # 3. Universal FFmpeg Fallback via imageio_ffmpeg (decodes AAC, MP4, M4A, WhatsApp Audio, OPUS, etc.)
            try:
                import imageio_ffmpeg
                ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_out:
                    temp_wav_path = tmp_out.name

                cmd = [
                    ffmpeg_exe,
                    "-y",
                    "-i", temp_input_path,
                    "-ac", "1",
                    "-ar", str(self.target_sr),
                    "-f", "wav",
                    temp_wav_path
                ]
                subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

                audio, sr = librosa.load(temp_wav_path, sr=self.target_sr, mono=True)
                return audio, sr
            except Exception as e_ffmpeg:
                logger.error(f"FFmpeg audio extraction failed for '{filename}': {e_ffmpeg}")
                raise ValueError(f"Unable to parse audio stream from '{filename or 'file'}'. Ensure file contains valid audio.")

        finally:
            for p in [temp_input_path, temp_wav_path]:
                if p and os.path.exists(p) and p != file_source:
                    try:
                        os.remove(p)
                    except Exception:
                        pass

    def preprocess_audio(self, audio: np.ndarray, sr: int = None, trim_silence_flag: bool = False) -> np.ndarray:
        """
        Preprocesses audio signal:
        1. Resamples to target sample rate if needed
        2. Removes DC offset
        3. Normalizes amplitude
        4. Trims leading/trailing silence if trim_silence_flag=True
        """
        sr = sr or self.target_sr
        if audio is None or len(audio) == 0:
            return np.array([], dtype=np.float32)

        # Ensure 1D float32
        audio = audio.astype(np.float32)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        # 1. Resample if necessary
        if sr != self.target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.target_sr)

        # 2. Remove DC offset
        audio = audio - np.mean(audio)

        # 3. Peak Amplitude Normalization (normalize only when real speech/audio signal is present)
        max_val = np.max(np.abs(audio))
        if max_val >= 0.01:
            audio = audio / max_val

        # 4. Optional silence trimming
        if trim_silence_flag:
            try:
                audio_trimmed, _ = librosa.effects.trim(audio, top_db=30)
                if len(audio_trimmed) > self.target_sr * 0.2:
                    audio = audio_trimmed
            except Exception as e:
                logger.warning(f"Silence trimming warning: {e}")

        return audio

    def create_windows(
        self,
        audio: np.ndarray,
        sr: int = None,
        window_seconds: float = None,
        overlap_seconds: float = 0.5
    ) -> List[Tuple[np.ndarray, float, float]]:
        """
        Divides audio signal into sliding time windows.
        Returns list of tuples: (window_audio_array, start_time_sec, end_time_sec)
        """
        sr = sr or self.target_sr
        window_seconds = window_seconds or settings.WINDOW_SECONDS

        window_size = int(sr * window_seconds)
        hop_size = int(sr * max(0.1, window_seconds - overlap_seconds))

        if len(audio) == 0:
            return []

        # If audio is shorter than window_size, pad with zero or reflect
        if len(audio) < window_size:
            padded_audio = np.pad(audio, (0, window_size - len(audio)), mode='constant')
            return [(padded_audio, 0.0, len(audio) / sr)]

        windows = []
        for start in range(0, len(audio) - window_size + 1, hop_size):
            end = start + window_size
            window_data = audio[start:end]
            start_time = start / sr
            end_time = end / sr
            windows.append((window_data, start_time, end_time))

        return windows

    def pcm_bytes_to_float(self, pcm_bytes: bytes, sample_width: int = 2) -> np.ndarray:
        """
        Converts raw PCM 16-bit little-endian byte stream to normalized float32 array.
        """
        if sample_width == 2:
            int_data = np.frombuffer(pcm_bytes, dtype=np.int16)
            return (int_data / 32768.0).astype(np.float32)
        elif sample_width == 4:
            return np.frombuffer(pcm_bytes, dtype=np.float32)
        else:
            raise ValueError(f"Unsupported sample width: {sample_width}")

audio_processor = AudioProcessor()
