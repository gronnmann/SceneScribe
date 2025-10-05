# SceneScribe

A Python-based AI pipeline that analyzes videos (for example TikTok), transcribes speech, describes visual scenes, and exports structured JSON metadata for different applications.

Note this project is an attempt at vibe coding using the new Claude Sonnet 4.5. It appears to work well (hence I've uploaded it to GitHub), but be careful.

## 🎯 Features

- **Automatic Speech Recognition (ASR)**: WhisperX with Norwegian and English support
- **Scene Detection**: PySceneDetect for intelligent keyframe extraction
- **Image Captioning**: BLIP-2 for detailed visual descriptions
- **OCR**: Tesseract for on-screen text extraction
- **Structured Output**: Pydantic-validated JSON with timestamps and metadata
- **GPU Accelerated**: Full CUDA support for RTX 3090 and other NVIDIA GPUs

## 📦 Installation

### Option 1: Local Installation

**Prerequisites:**
- Python 3.10+
- CUDA 12.1+ (for GPU acceleration)
- FFmpeg
- Tesseract OCR

```bash
# Clone repository
git clone <repository-url>
cd VideoAdTranscriber

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Double check installation
python verify_setup.py
```

### Option 2: Docker

```bash
# Build Docker image
docker build -t video-ad-transcriber .

# Run container with GPU support
docker run --gpus all -v $(pwd)/data:/app/data -v $(pwd)/outputs:/app/outputs video-ad-transcriber python3 src/main.py --input data/videos --output outputs/json
```

## 🚀 Usage

### Basic Usage

Process a single video:

```bash
python src/main.py --input data/videos/video.mp4 --output outputs/json
```

Process all videos in a directory:

```bash
python src/main.py --input data/videos --output outputs/json
```

### Advanced Options

```bash
python src/main.py \
  --input data/videos \
  --output outputs/json \
  --frames-dir outputs/frames \
  --language no \
  --scene-threshold 25.0 \
  --num-captions 3 \
  --skip-ocr \
  --keep-frames \
  --keep-audio \
  --log-level DEBUG
```

### CLI Options

| Option | Description | Default |
|--------|-------------|---------|
| `--input`, `-i` | Input video file or directory | Required |
| `--output`, `-o` | Output directory for JSON files | `outputs/json` |
| `--frames-dir` | Output directory for keyframes | `outputs/frames` |
| `--language`, `-l` | Language code (e.g., 'en', 'no') | Auto-detect |
| `--scene-threshold` | Scene detection threshold (lower = more sensitive) | 27.0 |
| `--num-captions` | Number of captions per keyframe | 1 |
| `--skip-ocr` | Skip OCR text extraction | False |
| `--keep-frames` | Keep extracted keyframe images | False |
| `--keep-audio` | Keep extracted audio files | False |
| `--log-level` | Logging level (DEBUG, INFO, WARNING, ERROR) | INFO |

## 📁 Project Structure

```
VideoAdTranscriber/
├── data/
│   └── videos/              # Input videos (.mp4)
├── outputs/
│   ├── json/                # Output JSON files
│   └── frames/              # Extracted keyframes
├── src/
│   ├── __init__.py
│   ├── main.py              # CLI entrypoint
│   ├── video_preprocess.py  # Video processing & scene detection
│   ├── asr.py               # WhisperX transcription
│   ├── vision.py            # BLIP-2 captioning
│   ├── ocr.py               # Tesseract OCR
│   ├── fusion.py            # Data fusion & JSON export
│   └── utils.py             # Utilities
├── requirements.txt
├── Dockerfile
└── README.md
```

## 📄 Output Format

Each video produces a JSON file with the following structure:

```json
{
  "video_id": "video_001",
  "filename": "video_001.mp4",
  "duration_s": 32.5,
  "language": "no",
  "transcript": {
    "text": "Hei alle sammen, i dag viser jeg deg hvordan...",
    "words": [
      {"word": "Hei", "start_s": 0.12, "end_s": 0.35},
      {"word": "alle", "start_s": 0.36, "end_s": 0.56}
    ]
  },
  "shots": [
    {
      "shot_id": "s001",
      "start_s": 0.0,
      "end_s": 5.1,
      "keyframe": "frames/video_001/video_001_s001.jpg",
      "captions": [
        {
          "text": "A woman smiling at the camera holding a coffee mug",
          "confidence": 0.89
        }
      ],
      "ocr_text": "",
      "objects": []
    }
  ],
  "processing": {
    "created_at": "2025-10-04T18:30:00Z",
    "models": {
      "asr": "whisperx-large-v2",
      "caption": "blip2-opt-2.7b",
      "object_detector": "none",
      "ocr": "tesseract"
    },
    "version": "0.1.0"
  }
}
```

## 🧠 Pipeline Stages

1. **Video Preprocessing**: Extract audio, detect scenes, extract keyframes
2. **Audio Transcription**: WhisperX with word-level timestamps
3. **Visual Analysis**: BLIP-2 image captioning
4. **OCR**: Tesseract text extraction
5. **Data Fusion**: Merge into structured JSON
6. **Export**: Save metadata and keyframes

## ⚙️ System Requirements

### Minimum Requirements
- CPU: 4+ cores
- RAM: 16 GB
- Storage: 10 GB + space for videos/outputs

### Recommended (GPU Acceleration)
- GPU: NVIDIA RTX 3090 or equivalent (24GB VRAM)
- CUDA: 12.1+
- RAM: 32 GB
- Storage: 50 GB SSD

## 🛠️ Development

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=src tests/
```

### Code Quality

```bash
# Format code
black src/

# Lint
ruff check src/

# Type checking
mypy src/
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Submit a pull request

## 📝 License

This project is open-source and available under the MIT License.

## 🙏 Acknowledgments

- **WhisperX**: Fast speech recognition with word-level timestamps
- **PySceneDetect**: Scene detection and analysis
- **BLIP-2**: Image captioning
- **Tesseract**: OCR engine

## 📧 Contact

For questions or issues, please open a GitHub issue.

---