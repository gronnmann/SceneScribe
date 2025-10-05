"""Automatic Speech Recognition using WhisperX."""

import warnings
from pathlib import Path
from typing import Optional

import torch
import whisperx
from loguru import logger

# Suppress pyannote.audio compatibility warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote")
warnings.filterwarnings("ignore", message=".*pyannote.audio.*")
warnings.filterwarnings("ignore", message=".*TensorFloat-32.*")


class WhisperXTranscriber:
    """WhisperX-based transcription with word-level timestamps."""
    
    def __init__(
        self,
        device: str = "cuda",
        compute_type: str = "float16",
        batch_size: int = 16,
    ):
        """
        Initialize WhisperX transcriber.
        
        Args:
            device: Device to use ('cuda' or 'cpu')
            compute_type: Computation precision ('float16', 'int8', 'float32')
            batch_size: Batch size for processing
        """
        self.device = device
        self.compute_type = compute_type if device == "cuda" else "int8"
        self.batch_size = batch_size
        self.model: Optional[whisperx.asr.WhisperModel] = None
        self.align_model: Optional[any] = None
        self.align_metadata: Optional[any] = None
        
        # Configure PyTorch for better compatibility
        if device == "cuda" and torch.cuda.is_available():
            try:
                # Allow TF32 for better performance on Ampere GPUs
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            except Exception as e:
                logger.debug(f"Could not configure TF32: {e}")
        
        logger.info(
            f"WhisperX initialized: device={self.device}, "
            f"compute_type={self.compute_type}, batch_size={self.batch_size}"
        )
    
    def load_model(self, model_name: str = "large-v2") -> None:
        """
        Load WhisperX ASR model.
        
        Args:
            model_name: Model name ('large-v2', 'medium', 'small', etc.)
        """
        logger.info(f"Loading WhisperX model: {model_name}")
        self.model = whisperx.load_model(
            model_name,
            self.device,
            compute_type=self.compute_type,
        )
        logger.info("WhisperX model loaded successfully")
    
    def transcribe(
        self,
        audio_path: Path,
        language: Optional[str] = None,
    ) -> dict[str, any]:
        """
        Transcribe audio file with word-level timestamps.
        
        Args:
            audio_path: Path to audio file
            language: Language code ('en', 'no', etc.) or None for auto-detection
            
        Returns:
            Transcription result with segments and language
        """
        if self.model is None:
            self.load_model()
        
        logger.info(f"Transcribing audio: {audio_path.name}")
        
        # Load audio
        audio = whisperx.load_audio(str(audio_path))
        
        # Transcribe
        result = self.model.transcribe(
            audio,
            batch_size=self.batch_size,
            language=language,
        )
        
        detected_language = result.get("language", "unknown")
        logger.info(f"Transcription complete. Detected language: {detected_language}")
        
        return result
    
    def align_transcription(
        self,
        result: dict[str, any],
        audio_path: Path,
    ) -> dict[str, any]:
        """
        Perform forced alignment for word-level timestamps.
        
        Args:
            result: Transcription result from transcribe()
            audio_path: Path to audio file
            
        Returns:
            Aligned transcription with word-level timestamps
        """
        language_code = result.get("language", "en")
        
        logger.info(f"Loading alignment model for language: {language_code}")
        
        # Load alignment model
        self.align_model, self.align_metadata = whisperx.load_align_model(
            language_code=language_code,
            device=self.device,
        )
        
        # Load audio
        audio = whisperx.load_audio(str(audio_path))
        
        # Align
        logger.info("Performing forced alignment...")
        result = whisperx.align(
            result["segments"],
            self.align_model,
            self.align_metadata,
            audio,
            self.device,
            return_char_alignments=False,
        )
        
        logger.info("Alignment complete")
        return result
    
    def cleanup(self) -> None:
        """Free GPU memory."""
        del self.model
        del self.align_model
        del self.align_metadata
        
        self.model = None
        self.align_model = None
        self.align_metadata = None
        
        if self.device == "cuda":
            torch.cuda.empty_cache()
        
        logger.debug("WhisperX models unloaded")


def transcribe_audio(
    audio_path: Path,
    device: str = "cuda",
    language: Optional[str] = None,
) -> dict[str, any]:
    """
    High-level function to transcribe audio with alignment.
    
    Args:
        audio_path: Path to audio file
        device: Device to use
        language: Language code or None for auto-detection
        
    Returns:
        Dictionary with full text, segments, words, and language
    """
    transcriber = WhisperXTranscriber(device=device)
    
    # Transcribe
    result = transcriber.transcribe(audio_path, language=language)
    
    # Align for word-level timestamps
    aligned_result = transcriber.align_transcription(result, audio_path)
    
    # Extract full text
    full_text = " ".join([seg.get("text", "") for seg in aligned_result.get("segments", [])])
    
    # Extract word-level data
    words = []
    for segment in aligned_result.get("segments", []):
        for word_info in segment.get("words", []):
            words.append({
                "word": word_info.get("word", ""),
                "start_s": round(word_info.get("start", 0.0), 2),
                "end_s": round(word_info.get("end", 0.0), 2),
            })
    
    # Cleanup
    transcriber.cleanup()
    
    return {
        "text": full_text.strip(),
        "words": words,
        "language": result.get("language", "unknown"),
        "segments": aligned_result.get("segments", []),
    }
