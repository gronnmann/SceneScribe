"""Integration tests for the complete pipeline."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from main import main, process_video


class TestProcessVideo:
    """Test complete video processing pipeline."""
    
    @patch('main.fuse_and_export')
    @patch('main.add_ocr_to_scenes')
    @patch('main.caption_keyframes')
    @patch('main.transcribe_audio')
    @patch('main.extract_keyframes_from_scenes')
    @patch('main.detect_scenes')
    @patch('main.extract_audio')
    @patch('main.get_video_duration')
    @patch('main.validate_video_file')
    def test_process_video_complete_pipeline(
        self,
        mock_validate,
        mock_duration,
        mock_extract_audio,
        mock_detect_scenes,
        mock_extract_keyframes,
        mock_transcribe,
        mock_caption,
        mock_ocr,
        mock_fuse,
        tmp_path
    ):
        """Test complete video processing pipeline."""
        # Setup mocks
        mock_validate.return_value = True
        mock_duration.return_value = 30.0
        
        audio_path = tmp_path / "temp" / "test.wav"
        audio_path.parent.mkdir(parents=True)
        audio_path.touch()
        mock_extract_audio.return_value = audio_path
        
        mock_detect_scenes.return_value = [(0.0, 10.0), (10.0, 20.0)]
        
        mock_extract_keyframes.return_value = [
            {
                "shot_id": "s001",
                "start_s": 0.0,
                "end_s": 10.0,
                "keyframe": "frames/test/test_s001.jpg",
                "keyframe_path": tmp_path / "frames" / "test_s001.jpg"
            }
        ]
        
        mock_transcribe.return_value = {
            "text": "Hello world",
            "words": [],
            "language": "en"
        }
        
        mock_caption.return_value = [
            {
                "shot_id": "s001",
                "captions": [{"text": "A scene", "confidence": 0.9}]
            }
        ]
        
        mock_ocr.return_value = [
            {
                "shot_id": "s001",
                "ocr_text": "Some text"
            }
        ]
        
        mock_fuse.return_value = tmp_path / "output.json"
        
        # Test
        video_path = tmp_path / "test.mp4"
        video_path.touch()
        
        output_dir = tmp_path / "outputs" / "json"
        frames_dir = tmp_path / "outputs" / "frames"
        
        process_video(
            video_path=video_path,
            output_dir=output_dir,
            frames_dir=frames_dir,
            device="cpu",
            language=None,
            scene_threshold=27.0,
            num_captions=1,
            skip_ocr=False,
            keep_frames=False,
            keep_audio=False,
        )
        
        # Verify all steps were called
        mock_validate.assert_called_once()
        mock_duration.assert_called_once()
        mock_extract_audio.assert_called_once()
        mock_detect_scenes.assert_called_once()
        mock_extract_keyframes.assert_called_once()
        mock_transcribe.assert_called_once()
        mock_caption.assert_called_once()
        mock_ocr.assert_called_once()
        mock_fuse.assert_called_once()
    
    @patch('main.validate_video_file')
    def test_process_video_invalid_file(self, mock_validate, tmp_path):
        """Test processing with invalid video file."""
        mock_validate.return_value = False
        
        video_path = tmp_path / "invalid.mp4"
        
        with pytest.raises(ValueError):
            process_video(
                video_path=video_path,
                output_dir=tmp_path,
                frames_dir=tmp_path,
                device="cpu",
                language=None,
                scene_threshold=27.0,
                num_captions=1,
                skip_ocr=False,
            )


class TestMainCLI:
    """Test main CLI entry point."""
    
    @patch('main.process_video')
    @patch('main.get_device')
    @patch('main.setup_logger')
    def test_main_single_file(self, mock_logger, mock_device, mock_process, tmp_path):
        """Test main with single video file."""
        from click.testing import CliRunner
        
        mock_device.return_value = "cpu"
        
        # Create test video
        video_path = tmp_path / "test.mp4"
        video_path.touch()
        
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "--input", str(video_path),
                "--output", str(tmp_path / "output"),
            ]
        )
        
        assert result.exit_code == 0
        mock_process.assert_called_once()
    
    @patch('main.process_video')
    @patch('main.get_device')
    @patch('main.setup_logger')
    def test_main_directory(self, mock_logger, mock_device, mock_process, tmp_path):
        """Test main with directory of videos."""
        from click.testing import CliRunner
        
        mock_device.return_value = "cpu"
        
        # Create test videos
        video_dir = tmp_path / "videos"
        video_dir.mkdir()
        (video_dir / "test1.mp4").touch()
        (video_dir / "test2.mp4").touch()
        
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "--input", str(video_dir),
                "--output", str(tmp_path / "output"),
            ]
        )
        
        assert result.exit_code == 0
        assert mock_process.call_count == 2
    
    def test_main_no_videos(self, tmp_path):
        """Test main with empty directory."""
        from click.testing import CliRunner
        
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "--input", str(empty_dir),
                "--output", str(tmp_path / "output"),
            ]
        )
        
        assert result.exit_code == 0  # Should handle gracefully
