"""Unit tests for video preprocessing functions."""

from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

from video_preprocess import (
    detect_scenes,
    extract_audio,
    extract_keyframe,
    extract_keyframes_from_scenes,
    get_video_duration,
)


class TestExtractAudio:
    """Test audio extraction."""
    
    @patch('video_preprocess.subprocess.run')
    @patch('video_preprocess.ensure_directory')
    def test_extract_audio_success(self, mock_ensure_dir, mock_subprocess, tmp_path):
        """Test successful audio extraction."""
        video_path = tmp_path / "test.mp4"
        audio_path = tmp_path / "test.wav"
        
        mock_subprocess.return_value = MagicMock(returncode=0)
        
        result = extract_audio(video_path, audio_path)
        
        assert result == audio_path
        mock_subprocess.assert_called_once()
        assert "ffmpeg" in mock_subprocess.call_args[0][0]
    
    @patch('video_preprocess.subprocess.run')
    @patch('video_preprocess.ensure_directory')
    def test_extract_audio_custom_sample_rate(self, mock_ensure_dir, mock_subprocess, tmp_path):
        """Test audio extraction with custom sample rate."""
        video_path = tmp_path / "test.mp4"
        audio_path = tmp_path / "test.wav"
        
        mock_subprocess.return_value = MagicMock(returncode=0)
        
        extract_audio(video_path, audio_path, sample_rate=22050)
        
        cmd = mock_subprocess.call_args[0][0]
        assert "22050" in cmd


class TestGetVideoDuration:
    """Test video duration calculation."""
    
    @patch('cv2.VideoCapture')
    def test_get_video_duration_success(self, mock_cv2):
        """Test successful duration calculation."""
        mock_cap = MagicMock()
        mock_cap.get.side_effect = [30.0, 900]  # fps, frame_count
        mock_cv2.return_value = mock_cap
        
        duration = get_video_duration(Path("test.mp4"))
        
        assert duration == 30.0  # 900 frames / 30 fps
        mock_cap.release.assert_called_once()
    
    @patch('cv2.VideoCapture')
    def test_get_video_duration_zero_fps(self, mock_cv2):
        """Test duration with zero FPS."""
        mock_cap = MagicMock()
        mock_cap.get.side_effect = [0.0, 900]
        mock_cv2.return_value = mock_cap
        
        duration = get_video_duration(Path("test.mp4"))
        
        assert duration == 0.0


class TestDetectScenes:
    """Test scene detection."""
    
    @patch('video_preprocess.open_video')
    @patch('video_preprocess.SceneManager')
    def test_detect_scenes_multiple_scenes(self, mock_scene_manager, mock_open_video):
        """Test detection with multiple scenes."""
        mock_video = MagicMock()
        mock_open_video.return_value = mock_video
        
        mock_manager = MagicMock()
        mock_scene_manager.return_value = mock_manager
        
        # Mock scene list with timecodes
        mock_tc1_start = MagicMock()
        mock_tc1_start.get_seconds.return_value = 0.0
        mock_tc1_end = MagicMock()
        mock_tc1_end.get_seconds.return_value = 5.5
        
        mock_tc2_start = MagicMock()
        mock_tc2_start.get_seconds.return_value = 5.5
        mock_tc2_end = MagicMock()
        mock_tc2_end.get_seconds.return_value = 10.0
        
        mock_manager.get_scene_list.return_value = [
            (mock_tc1_start, mock_tc1_end),
            (mock_tc2_start, mock_tc2_end),
        ]
        
        scenes = detect_scenes(Path("test.mp4"))
        
        assert len(scenes) == 2
        assert scenes[0] == (0.0, 5.5)
        assert scenes[1] == (5.5, 10.0)
    
    @patch('video_preprocess.open_video')
    @patch('video_preprocess.SceneManager')
    def test_detect_scenes_custom_threshold(self, mock_scene_manager, mock_open_video):
        """Test detection with custom threshold."""
        mock_video = MagicMock()
        mock_open_video.return_value = mock_video
        
        mock_manager = MagicMock()
        mock_scene_manager.return_value = mock_manager
        mock_manager.get_scene_list.return_value = []
        
        detect_scenes(Path("test.mp4"), threshold=25.0)
        
        mock_manager.add_detector.assert_called_once()


class TestExtractKeyframe:
    """Test keyframe extraction."""
    
    @patch('video_preprocess.subprocess.run')
    @patch('video_preprocess.ensure_directory')
    def test_extract_keyframe_success(self, mock_ensure_dir, mock_subprocess, tmp_path):
        """Test successful keyframe extraction."""
        video_path = tmp_path / "test.mp4"
        output_path = tmp_path / "frame.jpg"
        
        mock_subprocess.return_value = MagicMock(returncode=0)
        
        result = extract_keyframe(video_path, 5.0, output_path)
        
        assert result == output_path
        cmd = mock_subprocess.call_args[0][0]
        assert "ffmpeg" in cmd
        assert "5.0" in cmd
    
    @patch('video_preprocess.subprocess.run')
    @patch('video_preprocess.ensure_directory')
    def test_extract_keyframe_failure(self, mock_ensure_dir, mock_subprocess, tmp_path):
        """Test failed keyframe extraction."""
        video_path = tmp_path / "test.mp4"
        output_path = tmp_path / "frame.jpg"
        
        mock_subprocess.side_effect = Exception("FFmpeg error")
        
        with pytest.raises(Exception, match="FFmpeg error"):
            extract_keyframe(video_path, 5.0, output_path)


class TestExtractKeyframesFromScenes:
    """Test keyframe extraction from scenes."""
    
    @patch('video_preprocess.extract_keyframe')
    @patch('video_preprocess.ensure_directory')
    def test_extract_keyframes_from_scenes(self, mock_ensure_dir, mock_extract, tmp_path):
        """Test extracting keyframes from multiple scenes."""
        video_path = tmp_path / "test.mp4"
        output_dir = tmp_path / "frames"
        
        scenes = [
            (0.0, 5.0),
            (5.0, 10.0),
            (10.0, 15.0),
        ]
        
        mock_extract.side_effect = [
            tmp_path / "frames" / "test_s001.jpg",
            tmp_path / "frames" / "test_s002.jpg",
            tmp_path / "frames" / "test_s003.jpg",
        ]
        
        scene_data = extract_keyframes_from_scenes(
            video_path, scenes, output_dir, "test"
        )
        
        assert len(scene_data) == 3
        assert scene_data[0]["shot_id"] == "s001"
        assert scene_data[0]["start_s"] == 0.0
        assert scene_data[0]["end_s"] == 5.0
        assert scene_data[1]["shot_id"] == "s002"
        assert scene_data[2]["shot_id"] == "s003"
