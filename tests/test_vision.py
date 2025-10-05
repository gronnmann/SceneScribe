"""Unit tests for vision/captioning functionality."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from vision import BLIP2Captioner, caption_keyframes


class TestBLIP2Captioner:
    """Test BLIP-2 captioner class."""
    
    def test_init(self):
        """Test initialization."""
        captioner = BLIP2Captioner(device="cuda")
        assert captioner.device == "cuda"
        assert captioner.model_name == "Salesforce/blip2-opt-2.7b"
        assert captioner.model is None
        assert captioner.processor is None
    
    @patch('vision.Blip2Processor.from_pretrained')
    @patch('vision.Blip2ForConditionalGeneration.from_pretrained')
    def test_load_model_cuda(self, mock_model, mock_processor):
        """Test model loading with CUDA."""
        mock_processor_instance = MagicMock()
        mock_processor.return_value = mock_processor_instance
        
        mock_model_instance = MagicMock()
        mock_model_instance.to.return_value = mock_model_instance
        mock_model.return_value = mock_model_instance
        
        captioner = BLIP2Captioner(device="cuda")
        captioner.load_model()
        
        assert captioner.processor == mock_processor_instance
        mock_model_instance.to.assert_called_once_with("cuda")
    
    @patch('vision.Blip2Processor.from_pretrained')
    @patch('vision.Blip2ForConditionalGeneration.from_pretrained')
    @patch('vision.Image.open')
    @patch('vision.torch.no_grad')
    def test_generate_caption(self, mock_no_grad, mock_image_open, mock_model_class, mock_processor_class):
        """Test caption generation."""
        # Setup mocks
        mock_processor = MagicMock()
        mock_processor_class.return_value = mock_processor
        
        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_model_class.return_value = mock_model
        
        mock_image = MagicMock()
        mock_image.convert.return_value = mock_image
        mock_image_open.return_value = mock_image
        
        # Mock processor output with 'to' method
        mock_inputs = MagicMock()
        mock_inputs.to.return_value = mock_inputs
        mock_processor.return_value = mock_inputs
        
        mock_model.generate.return_value = MagicMock()
        mock_processor.batch_decode.return_value = ["A beautiful sunset"]
        
        # Test
        captioner = BLIP2Captioner(device="cpu")
        captioner.load_model()
        captions = captioner.generate_caption(Path("test.jpg"))
        
        assert len(captions) == 1
        assert captions[0]["text"] == "A beautiful sunset"
        assert 0.0 <= captions[0]["confidence"] <= 1.0
    
    @patch('vision.torch.cuda.empty_cache')
    def test_cleanup(self, mock_empty_cache):
        """Test cleanup."""
        captioner = BLIP2Captioner(device="cuda")
        captioner.model = MagicMock()
        captioner.processor = MagicMock()
        
        captioner.cleanup()
        
        assert captioner.model is None
        assert captioner.processor is None
        mock_empty_cache.assert_called_once()


class TestCaptionKeyframes:
    """Test keyframe captioning function."""
    
    @patch('vision.BLIP2Captioner')
    def test_caption_keyframes_success(self, mock_captioner_class, tmp_path):
        """Test successful keyframe captioning."""
        # Create test keyframe file
        keyframe_path = tmp_path / "test.jpg"
        keyframe_path.touch()
        
        # Mock captioner
        mock_captioner = MagicMock()
        mock_captioner_class.return_value = mock_captioner
        
        mock_captioner.generate_caption.return_value = [
            {"text": "A dog playing", "confidence": 0.9}
        ]
        
        # Test data
        scene_data = [
            {
                "shot_id": "s001",
                "keyframe_path": keyframe_path,
            }
        ]
        
        result = caption_keyframes(scene_data, device="cuda")
        
        assert len(result) == 1
        assert "captions" in result[0]
        assert len(result[0]["captions"]) == 1
        assert result[0]["captions"][0]["text"] == "A dog playing"
        mock_captioner.cleanup.assert_called_once()
    
    @patch('vision.BLIP2Captioner')
    def test_caption_keyframes_missing_file(self, mock_captioner_class):
        """Test with missing keyframe file."""
        mock_captioner = MagicMock()
        mock_captioner_class.return_value = mock_captioner
        
        scene_data = [
            {
                "shot_id": "s001",
                "keyframe_path": Path("/nonexistent/frame.jpg"),
            }
        ]
        
        result = caption_keyframes(scene_data)
        
        assert len(result) == 1
        assert result[0]["captions"] == []
