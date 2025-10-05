"""Unit tests for ASR functionality."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from asr import WhisperXTranscriber, transcribe_audio


class TestWhisperXTranscriber:
    """Test WhisperX transcriber class."""
    
    def test_init_cuda(self):
        """Test initialization with CUDA."""
        transcriber = WhisperXTranscriber(device="cuda")
        assert transcriber.device == "cuda"
        assert transcriber.compute_type == "float16"
        assert transcriber.batch_size == 16
    
    def test_init_cpu(self):
        """Test initialization with CPU."""
        transcriber = WhisperXTranscriber(device="cpu")
        assert transcriber.device == "cpu"
        assert transcriber.compute_type == "int8"
    
    @patch('asr.whisperx.load_model')
    def test_load_model(self, mock_load):
        """Test model loading."""
        mock_model = MagicMock()
        mock_load.return_value = mock_model
        
        transcriber = WhisperXTranscriber()
        transcriber.load_model("large-v2")
        
        assert transcriber.model == mock_model
        mock_load.assert_called_once_with("large-v2", "cuda", compute_type="float16")
    
    @patch('asr.whisperx.load_audio')
    @patch('asr.whisperx.load_model')
    def test_transcribe(self, mock_load_model, mock_load_audio):
        """Test transcription."""
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {
            "segments": [{"text": "Hello world"}],
            "language": "en"
        }
        mock_load_model.return_value = mock_model
        
        mock_audio = MagicMock()
        mock_load_audio.return_value = mock_audio
        
        transcriber = WhisperXTranscriber()
        result = transcriber.transcribe(Path("test.wav"))
        
        assert result["language"] == "en"
        mock_model.transcribe.assert_called_once()
    
    @patch('asr.whisperx.load_align_model')
    @patch('asr.whisperx.load_audio')
    @patch('asr.whisperx.align')
    def test_align_transcription(self, mock_align, mock_load_audio, mock_load_align):
        """Test alignment."""
        mock_align_model = MagicMock()
        mock_metadata = MagicMock()
        mock_load_align.return_value = (mock_align_model, mock_metadata)
        
        mock_audio = MagicMock()
        mock_load_audio.return_value = mock_audio
        
        mock_align.return_value = {
            "segments": [{
                "text": "Hello",
                "words": [{"word": "Hello", "start": 0.0, "end": 0.5}]
            }]
        }
        
        transcriber = WhisperXTranscriber()
        result_dict = {"segments": [], "language": "en"}
        aligned = transcriber.align_transcription(result_dict, Path("test.wav"))
        
        assert "segments" in aligned
        mock_load_align.assert_called_once_with(language_code="en", device="cuda")
    
    @patch('torch.cuda.empty_cache')
    def test_cleanup(self, mock_empty_cache):
        """Test cleanup."""
        transcriber = WhisperXTranscriber(device="cuda")
        transcriber.model = MagicMock()
        transcriber.align_model = MagicMock()
        
        transcriber.cleanup()
        
        assert transcriber.model is None
        assert transcriber.align_model is None
        mock_empty_cache.assert_called_once()


class TestTranscribeAudio:
    """Test high-level transcribe function."""
    
    @patch('asr.WhisperXTranscriber')
    def test_transcribe_audio_success(self, mock_transcriber_class):
        """Test successful transcription."""
        mock_transcriber = MagicMock()
        mock_transcriber_class.return_value = mock_transcriber
        
        mock_transcriber.transcribe.return_value = {
            "segments": [{"text": "Hello world"}],
            "language": "en"
        }
        
        mock_transcriber.align_transcription.return_value = {
            "segments": [{
                "text": "Hello world",
                "words": [
                    {"word": "Hello", "start": 0.0, "end": 0.5},
                    {"word": "world", "start": 0.5, "end": 1.0}
                ]
            }]
        }
        
        result = transcribe_audio(Path("test.wav"), device="cuda")
        
        assert result["text"] == "Hello world"
        assert result["language"] == "en"
        assert len(result["words"]) == 2
        assert result["words"][0]["word"] == "Hello"
        mock_transcriber.cleanup.assert_called_once()
