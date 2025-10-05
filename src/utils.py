"""Utility functions for logging, GPU management, and file handling."""

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

import torch
from loguru import logger


def setup_logger(log_level: str = "INFO") -> None:
    """
    Configure loguru logger with custom formatting.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=log_level,
    )
    logger.info(f"Logger initialized with level: {log_level}")


def get_device() -> str:
    """
    Detect and return the best available compute device.
    
    Returns:
        Device string ('cuda' or 'cpu')
    """
    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"CUDA device detected: {gpu_name}")
    else:
        device = "cpu"
        logger.warning("No CUDA device found. Using CPU (this will be slow).")
    
    return device


def ensure_directory(path: Path) -> None:
    """
    Create directory if it doesn't exist.
    
    Args:
        path: Path to directory
    """
    path.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Ensured directory exists: {path}")


def get_video_id(video_path: Path) -> str:
    """
    Extract video ID from filename (without extension).
    
    Args:
        video_path: Path to video file
        
    Returns:
        Video ID string
    """
    return video_path.stem


def validate_video_file(video_path: Path) -> bool:
    """
    Check if video file exists and has valid extension.
    
    Args:
        video_path: Path to video file
        
    Returns:
        True if valid, False otherwise
    """
    valid_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    
    if not video_path.exists():
        logger.error(f"Video file not found: {video_path}")
        return False
    
    if video_path.suffix.lower() not in valid_extensions:
        logger.error(f"Invalid video extension: {video_path.suffix}")
        return False
    
    return True


def cleanup_gpu_memory() -> None:
    """Clear GPU cache and run garbage collection."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.debug("GPU memory cache cleared")


def check_ffmpeg() -> bool:
    """Check if FFmpeg is installed and accessible."""
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        logger.debug(f"FFmpeg found at: {ffmpeg_path}")
        return True
    else:
        logger.error("FFmpeg not found in system PATH")
        logger.error("Please install FFmpeg:")
        logger.error("  Windows: Download from https://ffmpeg.org/download.html")
        logger.error("           or use: winget install FFmpeg")
        logger.error("           or use: choco install ffmpeg")
        logger.error("  Linux: sudo apt-get install ffmpeg")
        logger.error("  macOS: brew install ffmpeg")
        return False


def check_tesseract() -> bool:
    """Check if Tesseract OCR is installed and accessible."""
    tesseract_path = shutil.which("tesseract")
    if tesseract_path:
        logger.debug(f"Tesseract found at: {tesseract_path}")
        return True
    else:
        logger.warning("Tesseract OCR not found in system PATH")
        logger.warning("OCR features will not work. To install:")
        logger.warning("  Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki")
        logger.warning("           or use: winget install UB-Mannheim.TesseractOCR")
        logger.warning("  Linux: sudo apt-get install tesseract-ocr")
        logger.warning("  macOS: brew install tesseract")
        return False


def check_dependencies(skip_ocr: bool = False) -> bool:
    """Check all required system dependencies."""
    logger.info("Checking system dependencies...")
    
    all_ok = True
    
    # Check FFmpeg (required)
    if not check_ffmpeg():
        all_ok = False
    
    # Check Tesseract (optional if --skip-ocr)
    if not skip_ocr:
        if not check_tesseract():
            logger.warning("Consider using --skip-ocr flag if you don't need OCR")
    
    return all_ok
