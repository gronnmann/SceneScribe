"""Verify system dependencies and installation."""

import subprocess
import sys
from pathlib import Path

import click
from loguru import logger


def check_command(command: str, name: str, version_flag: str = "--version") -> bool:
    """Check if a command is available in PATH."""
    try:
        result = subprocess.run(
            [command, version_flag],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            version_line = result.stdout.split('\n')[0] or result.stderr.split('\n')[0]
            logger.info(f"✓ {name}: {version_line}")
            return True
        else:
            logger.error(f"✗ {name}: Command failed")
            return False
    except FileNotFoundError:
        logger.error(f"✗ {name}: Not found in PATH")
        return False
    except subprocess.TimeoutExpired:
        logger.error(f"✗ {name}: Command timeout")
        return False
    except Exception as e:
        logger.error(f"✗ {name}: {e}")
        return False


def check_python_package(package: str, import_name: str = None) -> bool:
    """Check if a Python package is installed."""
    if import_name is None:
        import_name = package
    
    try:
        __import__(import_name)
        logger.info(f"✓ Python package '{package}' is installed")
        return True
    except ImportError:
        logger.error(f"✗ Python package '{package}' is NOT installed")
        return False


def check_cuda() -> bool:
    """Check CUDA availability."""
    try:
        import torch
        
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            cuda_version = torch.version.cuda
            logger.info(f"✓ CUDA available: {device_name} (CUDA {cuda_version})")
            
            # Check for cuDNN
            if torch.backends.cudnn.is_available():
                logger.info(f"✓ cuDNN available: version {torch.backends.cudnn.version()}")
            else:
                logger.warning("⚠ cuDNN not available (may cause issues with WhisperX)")
            
            return True
        else:
            logger.warning("⚠ CUDA not available (will use CPU - slower)")
            return False
    except ImportError:
        logger.error("✗ PyTorch not installed")
        return False


@click.command()
def verify():
    """Verify all system dependencies for VideoAdTranscriber."""
    logger.remove()
    logger.add(
        sys.stderr,
        format="<level>{message}</level>",
        colorize=True,
    )
    
    logger.info("\n" + "="*60)
    logger.info("VideoAdTranscriber - Dependency Verification")
    logger.info("="*60 + "\n")
    
    checks = {
        "system": [],
        "python": [],
        "optional": [],
    }
    
    # System commands
    logger.info("System Dependencies:")
    logger.info("-" * 40)
    checks["system"].append(check_command("python", "Python", "--version"))
    checks["system"].append(check_command("ffmpeg", "FFmpeg", "-version"))
    checks["system"].append(check_command("tesseract", "Tesseract", "--version"))
    
    # Python packages
    logger.info("\nPython Packages:")
    logger.info("-" * 40)
    checks["python"].append(check_python_package("torch"))
    checks["python"].append(check_python_package("whisperx"))
    checks["python"].append(check_python_package("transformers"))
    checks["python"].append(check_python_package("scenedetect"))
    checks["python"].append(check_python_package("cv2", "cv2"))
    checks["python"].append(check_python_package("pytesseract"))
    checks["python"].append(check_python_package("pydantic"))
    checks["python"].append(check_python_package("click"))
    checks["python"].append(check_python_package("loguru"))
    
    # CUDA
    logger.info("\nGPU Support:")
    logger.info("-" * 40)
    checks["optional"].append(check_cuda())
    
    # Directory structure
    logger.info("\nDirectory Structure:")
    logger.info("-" * 40)
    
    base_dir = Path(__file__).parent.parent
    required_dirs = [
        base_dir / "src",
        base_dir / "data" / "videos",
        base_dir / "outputs" / "json",
        base_dir / "outputs" / "frames",
    ]
    
    dir_check = True
    for dir_path in required_dirs:
        if dir_path.exists():
            logger.info(f"✓ Directory exists: {dir_path.relative_to(base_dir)}")
        else:
            logger.warning(f"⚠ Directory missing: {dir_path.relative_to(base_dir)} (will be created)")
            dir_check = False
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("Verification Summary")
    logger.info("="*60)
    
    system_ok = all(checks["system"])
    python_ok = all(checks["python"])
    cuda_ok = any(checks["optional"])
    
    if system_ok:
        logger.info("✓ All system dependencies are installed")
    else:
        logger.error("✗ Some system dependencies are missing")
        logger.info("\nTo install missing dependencies:")
        logger.info("  - FFmpeg: https://ffmpeg.org/download.html")
        logger.info("  - Tesseract: https://github.com/UB-Mannheim/tesseract/wiki")
    
    if python_ok:
        logger.info("✓ All Python packages are installed")
    else:
        logger.error("✗ Some Python packages are missing")
        logger.info("\nTo install missing packages:")
        logger.info("  pip install -r requirements.txt")
    
    if cuda_ok:
        logger.info("✓ GPU acceleration available")
    else:
        logger.warning("⚠ GPU acceleration not available (CPU mode will be slower)")
        logger.info("\nFor GPU support:")
        logger.info("  - Install CUDA Toolkit: https://developer.nvidia.com/cuda-downloads")
        logger.info("  - Install cuDNN: https://developer.nvidia.com/cudnn")
        logger.info("  - Install PyTorch with CUDA: pip install torch --index-url https://download.pytorch.org/whl/cu121")
    
    logger.info("\n" + "="*60 + "\n")
    
    if system_ok and python_ok:
        logger.info("✓ Setup complete! You can now run:")
        logger.info("  python src/main.py --input data/videos --output outputs/json")
        return 0
    else:
        logger.error("✗ Setup incomplete. Please install missing dependencies.")
        return 1


if __name__ == "__main__":
    sys.exit(verify())
