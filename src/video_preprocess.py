"""Video preprocessing: normalization, scene detection, and keyframe extraction."""

import subprocess
from pathlib import Path
from typing import Optional

import cv2
from loguru import logger
from scenedetect import ContentDetector, SceneManager, open_video

from utils import ensure_directory


def extract_audio(
    video_path: Path,
    output_path: Path,
    sample_rate: int = 16000,
) -> Path:
    """
    Extract audio from video as mono WAV file using ffmpeg.
    
    Args:
        video_path: Path to input video
        output_path: Path for output audio file
        sample_rate: Audio sample rate in Hz
        
    Returns:
        Path to extracted audio file
    """
    ensure_directory(output_path.parent)
    
    cmd = [
        "ffmpeg",
        "-i", str(video_path),
        "-vn",  # No video
        "-acodec", "pcm_s16le",  # PCM 16-bit
        "-ar", str(sample_rate),  # Sample rate
        "-ac", "1",  # Mono
        "-y",  # Overwrite output
        str(output_path),
    ]
    
    logger.info(f"Extracting audio from {video_path.name} to {output_path.name}")
    
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            text=True,
        )
        logger.debug(f"Audio extraction successful: {output_path}")
        return output_path
    except FileNotFoundError:
        logger.error("FFmpeg executable not found!")
        logger.error("Please install FFmpeg and ensure it's in your system PATH")
        logger.error("See README.md or QUICKSTART.md for installation instructions")
        raise RuntimeError("FFmpeg not found. Please install FFmpeg to proceed.")
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg audio extraction failed: {e.stderr}")
        raise


def get_video_duration(video_path: Path) -> float:
    """
    Get video duration in seconds using OpenCV.
    
    Args:
        video_path: Path to video file
        
    Returns:
        Duration in seconds
    """
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    if fps > 0:
        duration = frame_count / fps
        logger.debug(f"Video duration: {duration:.2f}s ({frame_count} frames @ {fps} fps)")
        return duration
    else:
        logger.warning("Could not determine video duration")
        return 0.0


def detect_scenes(
    video_path: Path,
    threshold: float = 27.0,
) -> list[tuple[float, float]]:
    """
    Detect scene changes using PySceneDetect ContentDetector.
    
    Args:
        video_path: Path to video file
        threshold: Content detection threshold (lower = more sensitive)
        
    Returns:
        List of (start_time, end_time) tuples in seconds
    """
    logger.info(f"Detecting scenes in {video_path.name} (threshold={threshold})")
    
    video = open_video(str(video_path))
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))
    
    # Detect scenes
    scene_manager.detect_scenes(video)
    scene_list = scene_manager.get_scene_list()
    
    # Convert to time tuples
    scenes = []
    for scene in scene_list:
        start_time = scene[0].get_seconds()
        end_time = scene[1].get_seconds()
        scenes.append((start_time, end_time))
    
    logger.info(f"Detected {len(scenes)} scenes")
    return scenes


def extract_keyframe(
    video_path: Path,
    timestamp: float,
    output_path: Path,
) -> Optional[Path]:
    """
    Extract a single frame at specified timestamp using ffmpeg.
    
    Args:
        video_path: Path to video file
        timestamp: Time in seconds
        output_path: Path for output image
        
    Returns:
        Path to extracted frame or None on failure
    """
    ensure_directory(output_path.parent)
    
    cmd = [
        "ffmpeg",
        "-ss", str(timestamp),
        "-i", str(video_path),
        "-vframes", "1",
        "-q:v", "2",  # High quality
        "-y",
        str(output_path),
    ]
    
    try:
        subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            text=True,
        )
        logger.debug(f"Extracted keyframe at {timestamp:.2f}s: {output_path.name}")
        return output_path
    except FileNotFoundError:
        logger.error("FFmpeg executable not found!")
        raise RuntimeError("FFmpeg not found. Please install FFmpeg to proceed.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Keyframe extraction failed at {timestamp}s: {e.stderr}")
        return None


def extract_keyframes_from_scenes(
    video_path: Path,
    scenes: list[tuple[float, float]],
    output_dir: Path,
    video_id: str,
) -> list[dict[str, any]]:
    """
    Extract middle keyframe from each scene.
    
    Args:
        video_path: Path to video file
        scenes: List of (start_time, end_time) tuples
        output_dir: Directory for keyframe images
        video_id: Video identifier
        
    Returns:
        List of scene dictionaries with keyframe paths
    """
    ensure_directory(output_dir)
    
    scene_data = []
    
    for i, (start_time, end_time) in enumerate(scenes):
        shot_id = f"s{i+1:03d}"
        middle_time = (start_time + end_time) / 2
        
        keyframe_filename = f"{video_id}_{shot_id}.jpg"
        keyframe_path = output_dir / keyframe_filename
        
        extracted_path = extract_keyframe(video_path, middle_time, keyframe_path)
        
        if extracted_path:
            scene_data.append({
                "shot_id": shot_id,
                "start_s": round(start_time, 2),
                "end_s": round(end_time, 2),
                "keyframe": str(keyframe_path.relative_to(output_dir.parent)),
                "keyframe_path": keyframe_path,
            })
    
    logger.info(f"Extracted {len(scene_data)} keyframes")
    return scene_data
