"""Unit tests for data fusion and JSON export."""

import json
from datetime import datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from fusion import (
    Caption,
    ProcessingMetadata,
    Shot,
    Transcript,
    VideoMetadata,
    Word,
    create_video_metadata,
    fuse_and_export,
    save_metadata_json,
)


class TestPydanticModels:
    """Test Pydantic data models."""
    
    def test_word_model(self):
        """Test Word model."""
        word = Word(word="hello", start_s=0.0, end_s=0.5)
        assert word.word == "hello"
        assert word.start_s == 0.0
        assert word.end_s == 0.5
    
    def test_word_model_validation_error(self):
        """Test Word model validation."""
        with pytest.raises(ValidationError):
            Word(word="hello")  # Missing required fields
    
    def test_transcript_model(self):
        """Test Transcript model."""
        words = [
            Word(word="hello", start_s=0.0, end_s=0.5),
            Word(word="world", start_s=0.5, end_s=1.0),
        ]
        transcript = Transcript(text="hello world", words=words)
        assert transcript.text == "hello world"
        assert len(transcript.words) == 2
    
    def test_caption_model(self):
        """Test Caption model."""
        caption = Caption(text="A beautiful sunset", confidence=0.95)
        assert caption.text == "A beautiful sunset"
        assert caption.confidence == 0.95
    
    def test_caption_model_confidence_validation(self):
        """Test Caption confidence bounds."""
        with pytest.raises(ValidationError):
            Caption(text="Test", confidence=1.5)  # > 1.0
        
        with pytest.raises(ValidationError):
            Caption(text="Test", confidence=-0.1)  # < 0.0
    
    def test_shot_model(self):
        """Test Shot model."""
        shot = Shot(
            shot_id="s001",
            start_s=0.0,
            end_s=5.0,
            keyframe="frames/test_s001.jpg",
            captions=[Caption(text="Test", confidence=0.9)],
            ocr_text="Hello",
            objects=[]
        )
        assert shot.shot_id == "s001"
        assert shot.start_s == 0.0
        assert len(shot.captions) == 1
        assert shot.ocr_text == "Hello"
    
    def test_processing_metadata_model(self):
        """Test ProcessingMetadata model."""
        metadata = ProcessingMetadata(
            created_at="2025-10-04T12:00:00Z",
            models={"asr": "whisperx", "caption": "blip2"},
            version="0.1.0"
        )
        assert metadata.version == "0.1.0"
        assert "asr" in metadata.models
    
    def test_video_metadata_model(self):
        """Test VideoMetadata model."""
        metadata = VideoMetadata(
            video_id="test_001",
            filename="test_001.mp4",
            duration_s=30.5,
            language="en",
            transcript=Transcript(text="hello", words=[]),
            shots=[],
            processing=ProcessingMetadata(
                created_at="2025-10-04T12:00:00Z",
                models={},
                version="0.1.0"
            )
        )
        assert metadata.video_id == "test_001"
        assert metadata.duration_s == 30.5
        assert metadata.language == "en"


class TestCreateVideoMetadata:
    """Test video metadata creation."""
    
    def test_create_video_metadata_basic(self):
        """Test basic metadata creation."""
        transcript_data = {
            "text": "Hello world",
            "words": [
                {"word": "Hello", "start_s": 0.0, "end_s": 0.5},
                {"word": "world", "start_s": 0.5, "end_s": 1.0}
            ],
            "language": "en"
        }
        
        scene_data = [
            {
                "shot_id": "s001",
                "start_s": 0.0,
                "end_s": 5.0,
                "keyframe": "frames/test_s001.jpg",
                "captions": [{"text": "A scene", "confidence": 0.9}],
                "ocr_text": "Text"
            }
        ]
        
        metadata = create_video_metadata(
            video_id="test",
            filename="test.mp4",
            duration=30.0,
            transcript_data=transcript_data,
            scene_data=scene_data
        )
        
        assert metadata.video_id == "test"
        assert metadata.filename == "test.mp4"
        assert metadata.duration_s == 30.0
        assert metadata.language == "en"
        assert len(metadata.transcript.words) == 2
        assert len(metadata.shots) == 1
    
    def test_create_video_metadata_custom_models(self):
        """Test metadata with custom model names."""
        transcript_data = {"text": "", "words": [], "language": "no"}
        scene_data = []
        
        models = {
            "asr": "custom-whisper",
            "caption": "custom-blip",
            "ocr": "custom-ocr",
            "object_detector": "custom-detector"
        }
        
        metadata = create_video_metadata(
            video_id="test",
            filename="test.mp4",
            duration=10.0,
            transcript_data=transcript_data,
            scene_data=scene_data,
            models_used=models
        )
        
        assert metadata.processing.models["asr"] == "custom-whisper"
        assert metadata.processing.models["caption"] == "custom-blip"


class TestSaveMetadataJson:
    """Test JSON file saving."""
    
    def test_save_metadata_json(self, tmp_path):
        """Test saving metadata to JSON file."""
        metadata = VideoMetadata(
            video_id="test",
            filename="test.mp4",
            duration_s=30.0,
            language="en",
            transcript=Transcript(text="hello", words=[]),
            shots=[],
            processing=ProcessingMetadata(
                created_at="2025-10-04T12:00:00Z",
                models={},
                version="0.1.0"
            )
        )
        
        output_path = tmp_path / "test.json"
        save_metadata_json(metadata, output_path)
        
        assert output_path.exists()
        
        # Read and validate JSON
        with open(output_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        assert data["video_id"] == "test"
        assert data["filename"] == "test.mp4"
        assert data["duration_s"] == 30.0
    
    def test_save_metadata_json_creates_directory(self, tmp_path):
        """Test that parent directories are created."""
        output_path = tmp_path / "nested" / "dir" / "test.json"
        
        metadata = VideoMetadata(
            video_id="test",
            filename="test.mp4",
            duration_s=10.0,
            language="en",
            transcript=Transcript(text="", words=[]),
            shots=[],
            processing=ProcessingMetadata(
                created_at="2025-10-04T12:00:00Z",
                models={},
                version="0.1.0"
            )
        )
        
        save_metadata_json(metadata, output_path)
        
        assert output_path.exists()
        assert output_path.parent.exists()


class TestFuseAndExport:
    """Test high-level fusion and export."""
    
    def test_fuse_and_export(self, tmp_path):
        """Test complete fusion and export process."""
        transcript_data = {
            "text": "Test transcript",
            "words": [],
            "language": "en"
        }
        
        scene_data = [
            {
                "shot_id": "s001",
                "start_s": 0.0,
                "end_s": 5.0,
                "keyframe": "frames/test_s001.jpg",
                "captions": [],
                "ocr_text": ""
            }
        ]
        
        output_path = tmp_path / "output.json"
        
        result_path = fuse_and_export(
            video_id="test_video",
            filename="test_video.mp4",
            duration=30.0,
            transcript_data=transcript_data,
            scene_data=scene_data,
            output_path=output_path
        )
        
        assert result_path == output_path
        assert output_path.exists()
        
        # Validate JSON content
        with open(output_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        assert data["video_id"] == "test_video"
        assert data["transcript"]["text"] == "Test transcript"
        assert len(data["shots"]) == 1
