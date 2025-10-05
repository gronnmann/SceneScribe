"""Vision analysis: image captioning using BLIP-2."""

from pathlib import Path
from typing import Optional

import torch
from loguru import logger
from PIL import Image
from transformers import Blip2ForConditionalGeneration, Blip2Processor


class BLIP2Captioner:
    """BLIP-2 image captioning model."""
    
    def __init__(
        self,
        device: str = "cuda",
        model_name: str = "Salesforce/blip2-opt-2.7b",
    ):
        """
        Initialize BLIP-2 captioner.
        
        Args:
            device: Device to use ('cuda' or 'cpu')
            model_name: Hugging Face model identifier
        """
        self.device = device
        self.model_name = model_name
        self.model: Optional[Blip2ForConditionalGeneration] = None
        self.processor: Optional[Blip2Processor] = None
        
        logger.info(f"BLIP2Captioner initialized: device={self.device}, model={self.model_name}")
    
    def load_model(self) -> None:
        """Load BLIP-2 model and processor."""
        logger.info(f"Loading BLIP-2 model: {self.model_name}")
        
        self.processor = Blip2Processor.from_pretrained(self.model_name)
        
        # Load model with appropriate dtype
        if self.device == "cuda":
            self.model = Blip2ForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
            ).to(self.device)
        else:
            self.model = Blip2ForConditionalGeneration.from_pretrained(
                self.model_name,
            ).to(self.device)
        
        logger.info("BLIP-2 model loaded successfully")
    
    def generate_caption(
        self,
        image_path: Path,
        max_length: int = 50,
        num_captions: int = 1,
    ) -> list[dict[str, any]]:
        """
        Generate caption(s) for an image.
        
        Args:
            image_path: Path to image file
            max_length: Maximum caption length
            num_captions: Number of captions to generate
            
        Returns:
            List of caption dictionaries with text and confidence
        """
        if self.model is None:
            self.load_model()
        
        # Load image
        image = Image.open(image_path).convert("RGB")
        
        # Process image
        inputs = self.processor(images=image, return_tensors="pt").to(
            self.device,
            torch.float16 if self.device == "cuda" else torch.float32,
        )
        
        # Generate caption(s)
        captions = []
        
        for i in range(num_captions):
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=5 if num_captions == 1 else 3,
                    do_sample=num_captions > 1,
                    top_p=0.9 if num_captions > 1 else None,
                )
            
            caption_text = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
            )[0].strip()
            
            captions.append({
                "text": caption_text,
                "confidence": 1.0 / (i + 1),  # Simple confidence approximation
            })
        
        logger.debug(f"Generated {len(captions)} caption(s) for {image_path.name}")
        return captions
    
    def cleanup(self) -> None:
        """Free GPU memory."""
        del self.model
        del self.processor
        
        self.model = None
        self.processor = None
        
        if self.device == "cuda":
            torch.cuda.empty_cache()
        
        logger.debug("BLIP-2 model unloaded")


def caption_keyframes(
    scene_data: list[dict[str, any]],
    device: str = "cuda",
    num_captions: int = 1,
) -> list[dict[str, any]]:
    """
    Generate captions for all keyframes in scene data.
    
    Args:
        scene_data: List of scene dictionaries with keyframe_path
        device: Device to use
        num_captions: Number of captions per image
        
    Returns:
        Updated scene data with captions
    """
    captioner = BLIP2Captioner(device=device)
    captioner.load_model()
    
    logger.info(f"Generating captions for {len(scene_data)} keyframes...")
    
    for scene in scene_data:
        keyframe_path = scene.get("keyframe_path")
        
        if keyframe_path and Path(keyframe_path).exists():
            captions = captioner.generate_caption(
                Path(keyframe_path),
                num_captions=num_captions,
            )
            scene["captions"] = captions
        else:
            scene["captions"] = []
            logger.warning(f"Keyframe not found: {keyframe_path}")
    
    captioner.cleanup()
    logger.info("Captioning complete")
    
    return scene_data
