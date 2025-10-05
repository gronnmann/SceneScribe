"""OCR text extraction using Tesseract."""

from pathlib import Path

import pytesseract
from loguru import logger
from PIL import Image


def extract_text_from_image(image_path: Path, lang: str = "eng") -> str:
    """
    Extract text from image using Tesseract OCR.
    
    Args:
        image_path: Path to image file
        lang: Tesseract language code(s), e.g., 'eng', 'nor', 'eng+nor'
        
    Returns:
        Extracted text string
    """
    try:
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image, lang=lang)
        
        # Clean up text
        text = text.strip()
        
        if text:
            logger.debug(f"OCR extracted {len(text)} characters from {image_path.name}")
        
        return text
    
    except Exception as e:
        logger.error(f"OCR failed for {image_path.name}: {e}")
        return ""


def add_ocr_to_scenes(
    scene_data: list[dict[str, any]],
    languages: str = "eng+nor",
) -> list[dict[str, any]]:
    """
    Add OCR text extraction to all scenes.
    
    Args:
        scene_data: List of scene dictionaries with keyframe_path
        languages: Tesseract language codes
        
    Returns:
        Updated scene data with OCR text
    """
    logger.info(f"Running OCR on {len(scene_data)} keyframes (languages: {languages})")
    
    for scene in scene_data:
        keyframe_path = scene.get("keyframe_path")
        
        if keyframe_path and Path(keyframe_path).exists():
            ocr_text = extract_text_from_image(Path(keyframe_path), lang=languages)
            scene["ocr_text"] = ocr_text
        else:
            scene["ocr_text"] = ""
            logger.warning(f"Keyframe not found for OCR: {keyframe_path}")
    
    logger.info("OCR processing complete")
    return scene_data
