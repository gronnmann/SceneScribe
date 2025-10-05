"""Unit tests for OCR functionality."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ocr import add_ocr_to_scenes, extract_text_from_image


class TestExtractTextFromImage:
    """Test text extraction from images."""
    
    @patch('ocr.pytesseract.image_to_string')
    @patch('ocr.Image.open')
    def test_extract_text_success(self, mock_image_open, mock_pytesseract):
        """Test successful text extraction."""
        mock_image = MagicMock()
        mock_image_open.return_value = mock_image
        mock_pytesseract.return_value = "  Hello World  "
        
        text = extract_text_from_image(Path("test.jpg"), lang="eng")
        
        assert text == "Hello World"
        mock_pytesseract.assert_called_once_with(mock_image, lang="eng")
    
    @patch('ocr.pytesseract.image_to_string')
    @patch('ocr.Image.open')
    def test_extract_text_empty(self, mock_image_open, mock_pytesseract):
        """Test extraction with no text."""
        mock_image = MagicMock()
        mock_image_open.return_value = mock_image
        mock_pytesseract.return_value = ""
        
        text = extract_text_from_image(Path("test.jpg"))
        
        assert text == ""
    
    @patch('ocr.Image.open')
    def test_extract_text_error(self, mock_image_open):
        """Test extraction with error."""
        mock_image_open.side_effect = Exception("Image error")
        
        text = extract_text_from_image(Path("test.jpg"))
        
        assert text == ""


class TestAddOcrToScenes:
    """Test OCR addition to scenes."""
    
    @patch('ocr.extract_text_from_image')
    def test_add_ocr_to_scenes_success(self, mock_extract, tmp_path):
        """Test successful OCR addition."""
        # Create test keyframe
        keyframe_path = tmp_path / "frame.jpg"
        keyframe_path.touch()
        
        mock_extract.return_value = "Special Offer"
        
        scene_data = [
            {
                "shot_id": "s001",
                "keyframe_path": keyframe_path,
            },
            {
                "shot_id": "s002",
                "keyframe_path": keyframe_path,
            }
        ]
        
        result = add_ocr_to_scenes(scene_data, languages="eng")
        
        assert len(result) == 2
        assert result[0]["ocr_text"] == "Special Offer"
        assert result[1]["ocr_text"] == "Special Offer"
        assert mock_extract.call_count == 2
    
    @patch('ocr.extract_text_from_image')
    def test_add_ocr_to_scenes_missing_file(self, mock_extract):
        """Test OCR with missing keyframe."""
        scene_data = [
            {
                "shot_id": "s001",
                "keyframe_path": Path("/nonexistent/frame.jpg"),
            }
        ]
        
        result = add_ocr_to_scenes(scene_data)
        
        assert len(result) == 1
        assert result[0]["ocr_text"] == ""
        mock_extract.assert_not_called()
    
    @patch('ocr.extract_text_from_image')
    def test_add_ocr_multilingual(self, mock_extract, tmp_path):
        """Test OCR with multiple languages."""
        keyframe_path = tmp_path / "frame.jpg"
        keyframe_path.touch()
        
        mock_extract.return_value = "Hei verden"
        
        scene_data = [{"shot_id": "s001", "keyframe_path": keyframe_path}]
        
        result = add_ocr_to_scenes(scene_data, languages="eng+nor")
        
        mock_extract.assert_called_once_with(keyframe_path, lang="eng+nor")
        assert result[0]["ocr_text"] == "Hei verden"
