"""Main CLI entry point for TikTok Video Intelligence Extractor."""

from pathlib import Path

import click
from loguru import logger

from asr import transcribe_audio
from fusion import fuse_and_export
from ocr import add_ocr_to_scenes
from utils import (
    check_dependencies,
    cleanup_gpu_memory,
    get_device,
    get_video_id,
    setup_logger,
    validate_video_file,
)
from video_preprocess import (
    detect_scenes,
    extract_audio,
    extract_keyframes_from_scenes,
    get_video_duration,
)
from vision import caption_keyframes


@click.command()
@click.option(
    "--input",
    "-i",
    "input_path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Input video file or directory containing videos",
)
@click.option(
    "--output",
    "-o",
    "output_dir",
    type=click.Path(path_type=Path),
    default=Path("outputs/json"),
    help="Output directory for JSON files",
)
@click.option(
    "--frames-dir",
    type=click.Path(path_type=Path),
    default=Path("outputs/frames"),
    help="Output directory for keyframes",
)
@click.option(
    "--language",
    "-l",
    type=str,
    default=None,
    help="Language code for ASR (e.g., 'en', 'no'). Auto-detect if not specified.",
)
@click.option(
    "--scene-threshold",
    type=float,
    default=27.0,
    help="Scene detection threshold (lower = more sensitive)",
)
@click.option(
    "--num-captions",
    type=int,
    default=1,
    help="Number of captions to generate per keyframe",
)
@click.option(
    "--skip-ocr",
    is_flag=True,
    help="Skip OCR text extraction",
)
@click.option(
    "--keep-frames",
    is_flag=True,
    help="Keep extracted keyframe images (default: delete after processing)",
)
@click.option(
    "--keep-audio",
    is_flag=True,
    help="Keep extracted audio files (default: delete after processing)",
)
@click.option(
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
    default="INFO",
    help="Logging level",
)
def main(
    input_path: Path,
    output_dir: Path,
    frames_dir: Path,
    language: str | None,
    scene_threshold: float,
    num_captions: int,
    skip_ocr: bool,
    keep_frames: bool,
    keep_audio: bool,
    log_level: str,
) -> None:
    """
    TikTok Video Intelligence Extractor.
    
    Process video files to extract audio transcription, visual descriptions,
    and export structured JSON metadata.
    """
    # Setup logging
    setup_logger(log_level.upper())
    
    # Check system dependencies
    if not check_dependencies(skip_ocr=skip_ocr):
        logger.error("Missing required dependencies. Please install them and try again.")
        return
    
    # Get device
    device = get_device()
    
    # Collect video files
    if input_path.is_file():
        video_files = [input_path]
    else:
        video_files = list(input_path.glob("*.mp4"))
        video_files.extend(input_path.glob("*.avi"))
        video_files.extend(input_path.glob("*.mov"))
        video_files.extend(input_path.glob("*.mkv"))
        video_files.extend(input_path.glob("*.webm"))
    
    if not video_files:
        logger.error(f"No video files found in {input_path}")
        return
    
    logger.info(f"Found {len(video_files)} video(s) to process")
    
    # Process each video
    for video_path in video_files:
        try:
            process_video(
                video_path=video_path,
                output_dir=output_dir,
                frames_dir=frames_dir,
                device=device,
                language=language,
                scene_threshold=scene_threshold,
                num_captions=num_captions,
                skip_ocr=skip_ocr,
                keep_frames=keep_frames,
                keep_audio=keep_audio,
            )
        except Exception as e:
            logger.error(f"Failed to process {video_path.name}: {e}")
            logger.exception(e)
    
    logger.info("All videos processed successfully!")


def process_video(
    video_path: Path,
    output_dir: Path,
    frames_dir: Path,
    device: str,
    language: str | None,
    scene_threshold: float,
    num_captions: int,
    skip_ocr: bool,
    keep_frames: bool = False,
    keep_audio: bool = False,
) -> None:
    """
    Process a single video through the complete pipeline.
    
    Args:
        video_path: Path to video file
        output_dir: Output directory for JSON
        frames_dir: Output directory for keyframes
        device: Compute device
        language: Language code or None
        scene_threshold: Scene detection threshold
        num_captions: Number of captions per keyframe
        skip_ocr: Whether to skip OCR
        keep_frames: Keep extracted keyframes after processing
        keep_audio: Keep extracted audio files after processing
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing video: {video_path.name}")
    logger.info(f"{'='*60}\n")
    
    # Validate video
    if not validate_video_file(video_path):
        raise ValueError(f"Invalid video file: {video_path}")
    
    # Get video ID
    video_id = get_video_id(video_path)
    
    # Create output directories
    video_frames_dir = frames_dir / video_id
    video_frames_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Get video duration
    logger.info("Step 1/6: Getting video duration...")
    duration = get_video_duration(video_path)
    
    # Step 2: Extract audio
    logger.info("Step 2/6: Extracting audio...")
    audio_path = output_dir.parent / "temp" / f"{video_id}.wav"
    audio_path = extract_audio(video_path, audio_path)
    
    # Step 3: Detect scenes and extract keyframes
    logger.info("Step 3/6: Detecting scenes and extracting keyframes...")
    scenes = detect_scenes(video_path, threshold=scene_threshold)
    scene_data = extract_keyframes_from_scenes(
        video_path,
        scenes,
        video_frames_dir,
        video_id,
    )
    
    # Step 4: Transcribe audio
    logger.info("Step 4/6: Transcribing audio with WhisperX...")
    transcript_data = transcribe_audio(audio_path, device=device, language=language)
    cleanup_gpu_memory()
    
    # Step 5: Generate captions
    logger.info("Step 5/6: Generating image captions with BLIP-2...")
    scene_data = caption_keyframes(scene_data, device=device, num_captions=num_captions)
    cleanup_gpu_memory()
    
    # Step 6: OCR (optional)
    if not skip_ocr:
        logger.info("Step 6/6: Extracting text with OCR...")
        ocr_languages = "eng+nor" if transcript_data["language"] in ["no", "nn", "nb"] else "eng"
        scene_data = add_ocr_to_scenes(scene_data, languages=ocr_languages)
    else:
        logger.info("Step 6/6: Skipping OCR (--skip-ocr flag set)")
        for scene in scene_data:
            scene["ocr_text"] = ""
    
    # Fuse and export
    logger.info("Fusing data and exporting JSON...")
    output_json_path = output_dir / f"{video_id}.json"
    
    models_used = {
        "asr": "whisperx-large-v2",
        "caption": "blip2-opt-2.7b",
        "object_detector": "none",
        "ocr": "tesseract" if not skip_ocr else "none",
    }
    
    fuse_and_export(
        video_id=video_id,
        filename=video_path.name,
        duration=duration,
        transcript_data=transcript_data,
        scene_data=scene_data,
        output_path=output_json_path,
        models_used=models_used,
    )
    
    # Cleanup temporary files
    logger.info("Cleaning up temporary files...")
    
    # Always cleanup temp audio unless explicitly kept
    if audio_path.exists() and not keep_audio:
        audio_path.unlink()
        logger.debug(f"Deleted temp audio: {audio_path}")
    elif audio_path.exists() and keep_audio:
        logger.info(f"Keeping audio file: {audio_path}")
    
    # Cleanup temp directory if empty
    temp_dir = output_dir.parent / "temp"
    if temp_dir.exists() and not any(temp_dir.iterdir()):
        temp_dir.rmdir()
        logger.debug(f"Removed empty temp directory: {temp_dir}")
    
    # Cleanup keyframes unless explicitly kept
    if not keep_frames and video_frames_dir.exists():
        import shutil
        shutil.rmtree(video_frames_dir)
        logger.debug(f"Deleted keyframes directory: {video_frames_dir}")
        logger.info(f"Keyframes cleaned up (use --keep-frames to preserve)")
    elif keep_frames:
        logger.info(f"Keeping keyframes: {video_frames_dir}")
    
    logger.info(f"✓ Successfully processed: {video_path.name}")
    logger.info(f"  - JSON output: {output_json_path}")
    if keep_frames:
        logger.info(f"  - Keyframes: {video_frames_dir}")
    if keep_audio:
        logger.info(f"  - Audio: {audio_path}")


if __name__ == "__main__":
    main()
