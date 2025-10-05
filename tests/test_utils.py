"""Unit tests for utility functions."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from utils import (
    cleanup_gpu_memory,
    ensure_directory,
    get_device,
    get_video_id,
    setup_logger,
    validate_video_file,
)


class TestSetupLogger:
    """Test logger setup."""
    
    def test_setup_logger_info_level(self):
        """Test logger setup with INFO level."""
        setup_logger("INFO")
        # If no exception, test passes
        assert True
    
    def test_setup_logger_debug_level(self):
        """Test logger setup with DEBUG level."""
        setup_logger("DEBUG")
        assert True


class TestGetDevice:
    """Test device detection."""
    
    @patch('torch.cuda.is_available')
    def test_get_device_cuda_available(self, mock_cuda):
        """Test device detection when CUDA is available."""
        mock_cuda.return_value = True
        with patch('torch.cuda.get_device_name', return_value='RTX 3090'):
            device = get_device()
            assert device == "cuda"
    
    @patch('torch.cuda.is_available')
    def test_get_device_cuda_not_available(self, mock_cuda):
        """Test device detection when CUDA is not available."""
        mock_cuda.return_value = False
        device = get_device()
        assert device == "cpu"


class TestEnsureDirectory:
    """Test directory creation."""
    
    def test_ensure_directory_creates_new(self, tmp_path):
        """Test creating a new directory."""
        test_dir = tmp_path / "test" / "nested" / "dir"
        ensure_directory(test_dir)
        assert test_dir.exists()
        assert test_dir.is_dir()
    
    def test_ensure_directory_existing(self, tmp_path):
        """Test with existing directory."""
        test_dir = tmp_path / "existing"
        test_dir.mkdir()
        ensure_directory(test_dir)
        assert test_dir.exists()


class TestGetVideoId:
    """Test video ID extraction."""
    
    def test_get_video_id_mp4(self):
        """Test extracting ID from MP4 file."""
        video_path = Path("data/videos/test_video.mp4")
        video_id = get_video_id(video_path)
        assert video_id == "test_video"
    
    def test_get_video_id_no_extension(self):
        """Test extracting ID from file without extension."""
        video_path = Path("data/videos/test_video")
        video_id = get_video_id(video_path)
        assert video_id == "test_video"
    
    def test_get_video_id_with_path(self):
        """Test extracting ID from full path."""
        video_path = Path("/home/user/videos/my_video.avi")
        video_id = get_video_id(video_path)
        assert video_id == "my_video"


class TestValidateVideoFile:
    """Test video file validation."""
    
    def test_validate_video_file_valid_mp4(self, tmp_path):
        """Test validation with valid MP4 file."""
        video_file = tmp_path / "test.mp4"
        video_file.touch()
        assert validate_video_file(video_file) is True
    
    def test_validate_video_file_valid_avi(self, tmp_path):
        """Test validation with valid AVI file."""
        video_file = tmp_path / "test.avi"
        video_file.touch()
        assert validate_video_file(video_file) is True
    
    def test_validate_video_file_invalid_extension(self, tmp_path):
        """Test validation with invalid extension."""
        video_file = tmp_path / "test.txt"
        video_file.touch()
        assert validate_video_file(video_file) is False
    
    def test_validate_video_file_not_exists(self):
        """Test validation with non-existent file."""
        video_file = Path("/nonexistent/video.mp4")
        assert validate_video_file(video_file) is False


class TestCleanupGpuMemory:
    """Test GPU memory cleanup."""
    
    @patch('torch.cuda.is_available')
    @patch('torch.cuda.empty_cache')
    def test_cleanup_gpu_memory_cuda_available(self, mock_empty_cache, mock_cuda):
        """Test GPU cleanup when CUDA is available."""
        mock_cuda.return_value = True
        cleanup_gpu_memory()
        mock_empty_cache.assert_called_once()
    
    @patch('torch.cuda.is_available')
    @patch('torch.cuda.empty_cache')
    def test_cleanup_gpu_memory_cuda_not_available(self, mock_empty_cache, mock_cuda):
        """Test GPU cleanup when CUDA is not available."""
        mock_cuda.return_value = False
        cleanup_gpu_memory()
        mock_empty_cache.assert_not_called()
