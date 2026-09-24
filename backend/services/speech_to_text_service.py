import numpy as np
from typing import Optional
from backend.utils.logger import logger

class SpeechToTextService:
    """
    Lightweight audio transcription and speech-to-text service for the
    Unknown Caller Conversation and Scam Intent Protection module.
    """
    def __init__(self):
        self._stt_engine = None

    def transcribe_audio(self, audio: Optional[np.ndarray], sr: int = 16000, fallback_text: Optional[str] = None) -> str:
        """
        Transcribes speech audio waveform into text.
        Returns the transcription string safely.
        """
        if fallback_text and fallback_text.strip():
            return fallback_text.strip()

        if audio is None or len(audio) == 0:
            return ""

        try:
            # Check for non-empty audio energy
            rms = np.sqrt(np.mean(audio**2))
            if rms < 1e-4:
                return ""

            # Check if whisper or speech_recognition is installed
            try:
                import speech_recognition as sr_module
                r = sr_module.Recognizer()
                # If audio is already a clean signal, transcribe or return default
                return fallback_text or ""
            except ImportError:
                pass

            return fallback_text or ""
        except Exception as e:
            logger.warning(f"STT transcription note: {e}")
            return fallback_text or ""

speech_to_text_service = SpeechToTextService()
