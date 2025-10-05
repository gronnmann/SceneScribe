"""Pytest configuration and fixtures."""

import pytest


@pytest.fixture
def sample_video_path(tmp_path):
    """Create a sample video path for testing."""
    video_path = tmp_path / "test_video.mp4"
    video_path.touch()
    return video_path


@pytest.fixture
def sample_audio_path(tmp_path):
    """Create a sample audio path for testing."""
    audio_path = tmp_path / "test_audio.wav"
    audio_path.touch()
    return audio_path


@pytest.fixture
def sample_scene_data():
    """Sample scene data for testing."""
    return [
        {
            "shot_id": "s001",
            "start_s": 0.0,
            "end_s": 5.0,
            "keyframe": "frames/test_s001.jpg",
            "captions": [],
            "ocr_text": ""
        },
        {
            "shot_id": "s002",
            "start_s": 5.0,
            "end_s": 10.0,
            "keyframe": "frames/test_s002.jpg",
            "captions": [],
            "ocr_text": ""
        }
    ]


@pytest.fixture
def sample_transcript_data():
    """Sample transcript data for testing."""
    return {
        "text": "Hello world, this is a test.",
        "words": [
            {"word": "Hello", "start_s": 0.0, "end_s": 0.5},
            {"word": "world", "start_s": 0.5, "end_s": 1.0},
            {"word": "this", "start_s": 1.5, "end_s": 1.8},
            {"word": "is", "start_s": 1.8, "end_s": 2.0},
            {"word": "a", "start_s": 2.0, "end_s": 2.1},
            {"word": "test", "start_s": 2.1, "end_s": 2.5}
        ],
        "language": "en"
    }
