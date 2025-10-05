"""Data fusion: merge ASR, vision, and OCR into structured JSON output."""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Optional

from loguru import logger
from pydantic import BaseModel, Field


class Word(BaseModel):
    """Word-level transcription data."""
    
    word: str = Field(description="Transcribed word")
    start_s: float = Field(description="Start time in seconds")
    end_s: float = Field(description="End time in seconds")


class Transcript(BaseModel):
    """Complete transcript data."""
    
    text: str = Field(description="Full transcript text")
    words: list[Word] = Field(default_factory=list, description="Word-level timestamps")


class Caption(BaseModel):
    """Image caption data."""
    
    text: str = Field(description="Caption text")
    confidence: float = Field(description="Confidence score", ge=0.0, le=1.0)


class DetectedObject(BaseModel):
    """Detected object data."""
    
    label: str = Field(description="Object label")
    confidence: float = Field(description="Confidence score", ge=0.0, le=1.0)


class Shot(BaseModel):
    """Scene/shot data."""
    
    shot_id: str = Field(description="Shot identifier")
    start_s: float = Field(description="Start time in seconds")
    end_s: float = Field(description="End time in seconds")
    keyframe: str = Field(description="Relative path to keyframe image")
    captions: list[Caption] = Field(default_factory=list, description="Image captions")
    ocr_text: str = Field(default="", description="OCR extracted text")
    objects: list[DetectedObject] = Field(default_factory=list, description="Detected objects")


class ProcessingMetadata(BaseModel):
    """Processing metadata."""
    
    created_at: str = Field(description="ISO timestamp of creation")
    models: dict[str, str] = Field(description="Model names used")
    version: str = Field(default="0.1.0", description="Pipeline version")


class VideoMetadata(BaseModel):
    """Complete video metadata output."""
    
    video_id: str = Field(description="Video identifier")
    filename: str = Field(description="Original filename")
    duration_s: float = Field(description="Video duration in seconds")
    language: str = Field(description="Detected language code")
    transcript: Transcript = Field(description="Transcription data")
    shots: list[Shot] = Field(description="Scene/shot data")
    processing: ProcessingMetadata = Field(description="Processing metadata")


def create_video_metadata(
    video_id: str,
    filename: str,
    duration: float,
    transcript_data: dict[str, any],
    scene_data: list[dict[str, any]],
    models_used: Optional[dict[str, str]] = None,
) -> VideoMetadata:
    """
    Create structured video metadata from pipeline outputs.
    
    Args:
        video_id: Video identifier
        filename: Original video filename
        duration: Video duration in seconds
        transcript_data: Transcription result
        scene_data: Scene data with captions and OCR
        models_used: Dictionary of model names used
        
    Returns:
        VideoMetadata object
    """
    # Default models
    if models_used is None:
        models_used = {
            "asr": "whisperx-large-v2",
            "caption": "blip2-opt-2.7b",
            "object_detector": "none",
            "ocr": "tesseract",
        }
    
    # Build transcript
    transcript = Transcript(
        text=transcript_data.get("text", ""),
        words=[Word(**word) for word in transcript_data.get("words", [])],
    )
    
    # Build shots
    shots = []
    for scene in scene_data:
        shot = Shot(
            shot_id=scene.get("shot_id", ""),
            start_s=scene.get("start_s", 0.0),
            end_s=scene.get("end_s", 0.0),
            keyframe=scene.get("keyframe", ""),
            captions=[Caption(**cap) for cap in scene.get("captions", [])],
            ocr_text=scene.get("ocr_text", ""),
            objects=[],  # Object detection not implemented yet
        )
        shots.append(shot)
    
    # Build processing metadata
    processing = ProcessingMetadata(
        created_at=datetime.now(UTC).isoformat().replace('+00:00', 'Z'),
        models=models_used,
        version="0.1.0",
    )
    
    # Build complete metadata
    metadata = VideoMetadata(
        video_id=video_id,
        filename=filename,
        duration_s=round(duration, 2),
        language=transcript_data.get("language", "unknown"),
        transcript=transcript,
        shots=shots,
        processing=processing,
    )
    
    return metadata


def save_metadata_json(
    metadata: VideoMetadata,
    output_path: Path,
) -> None:
    """
    Save video metadata to JSON file.
    
    Args:
        metadata: VideoMetadata object
        output_path: Path for output JSON file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            metadata.model_dump(),
            f,
            indent=2,
            ensure_ascii=False,
        )
    
    logger.info(f"Saved metadata JSON: {output_path}")


def fuse_and_export(
    video_id: str,
    filename: str,
    duration: float,
    transcript_data: dict[str, any],
    scene_data: list[dict[str, any]],
    output_path: Path,
    models_used: Optional[dict[str, str]] = None,
) -> Path:
    """
    High-level function to fuse data and export JSON.
    
    Args:
        video_id: Video identifier
        filename: Original video filename
        duration: Video duration
        transcript_data: Transcription result
        scene_data: Scene data with captions and OCR
        output_path: Path for output JSON
        models_used: Models used in processing
        
    Returns:
        Path to saved JSON file
    """
    logger.info(f"Fusing data for video: {video_id}")
    
    metadata = create_video_metadata(
        video_id=video_id,
        filename=filename,
        duration=duration,
        transcript_data=transcript_data,
        scene_data=scene_data,
        models_used=models_used,
    )
    
    save_metadata_json(metadata, output_path)
    
    return output_path
