# 🧠 Copilot Instructions: TikTok Video Intelligence Extractor

This document defines how GitHub Copilot (or any code generation AI) should behave when generating the code for this project.

---

## 🎯 Mission
Create a **Python-based pipeline** that analyzes TikTok videos, transcribes the speech, describes scenes, and exports a JSON summary of each video.

The goal is **accuracy, modularity, and reproducibility** — every module should be clear, testable, and built to run on a **CUDA-enabled GPU (RTX 3090)**.

---

## 🧰 Project Context
This project already has a specification file named `PROJECT.md`.  
Copilot should refer to that document as the *single source of truth* for structure and functionality.

The project must be fully **open-source**, using **publicly available models and libraries** only.

---

## 🧱 Folder Structure to Follow
tiktok_ai_extractor/
├── data/videos/ # Input .mp4 files
├── outputs/json/ # Output JSONs
├── outputs/frames/ # Extracted keyframes
├── src/
│ ├── main.py # CLI entrypoint
│ ├── video_preprocess.py # ffmpeg + PySceneDetect utilities
│ ├── asr.py # WhisperX transcription
│ ├── vision.py # BLIP-2 captioning + GroundingDINO
│ ├── ocr.py # OCR integration
│ ├── fusion.py # Merge results into JSON
│ └── utils.py # Helpers
├── requirements.txt
├── Dockerfile
└── README.md


---

## 🧩 Coding Standards

### General
- Use **Python 3.10+** syntax.
- Include **type hints** and **docstrings** (Google style).
- Use | instead of typing.Union and list[xxx] instead of List[xxx] etc
- Follow **PEP 8** formatting.
- Use clear, functional decomposition (small, reusable functions).
- Keep imports organized by standard → third-party → local.

### Logging & Errors
- Use loguru instead of `print()`.
- Provide meaningful error handling with custom exceptions where appropriate.

### Dependencies
- Use `torch`, `transformers`, `openai-whisper`, `ffmpeg-python`, `PySceneDetect`, `pandas`, and `opencv-python`.
- Avoid proprietary or cloud-based APIs — only open-source models (e.g., via Hugging Face).
- Use `pydantic` for the json schema.
- Use `click` for the cli interface.

### GPU Usage
- Auto-detect CUDA with `torch.cuda.is_available()`.
- Use `device = "cuda" if torch.cuda.is_available() else "cpu"` consistently.

---

## ⚙️ Behavior Rules for Copilot

1. **Never create placeholder code** like `# TODO`.  
   Always implement a minimal but functional version.
2. **Follow modular boundaries:**  
   - `main.py` orchestrates the pipeline.  
   - Each module performs a single stage.
3. **Do not use hardcoded paths** — always use `pathlib.Path` relative paths.
4. **Ensure JSON outputs** match the example in `PROJECT.md`.
5. **Use functions, not Jupyter notebooks** or inline scripts.
6. **Comment your reasoning only in docstrings** — avoid verbose inline comments.
7. **Optimize for readability**, not cleverness.

---

## 🧩 Expected Modules Overview

| Module | Responsibility |
|---------|----------------|
| `video_preprocess.py` | Normalize video, detect scenes, extract keyframes |
| `asr.py` | Transcribe audio with WhisperX, detect language |
| `vision.py` | Generate image captions, optionally detect objects |
| `ocr.py` | Extract text from keyframes using Tesseract/PaddleOCR |
| `fusion.py` | Merge ASR + visual + OCR data into final JSON |
| `utils.py` | Logging, file handling, GPU utilities |
| `main.py` | Orchestrate the full pipeline via CLI or batch mode |

---

## 🧠 Output Example
All outputs must follow the JSON format provided in `PROJECT_DESCRIPTION.md`.  
Ensure that timestamps, keyframe references, and language codes are preserved.

---

## 🧩 Development Goals
- Modular and scalable architecture
- Full GPU utilization (WhisperX + BLIP-2)
- Reliable JSON output per video
- Multi-language support (Norwegian + English)
- Ready for later integration with LLM-based creative systems

---

## ✅ Done Definition
A minimal version is **“done”** when:
- The pipeline can process at least one `.mp4` file end-to-end.
- JSON output is successfully created in `/outputs/json/`.
- No manual path configuration is required.
- All modules run with GPU acceleration (if available).

---

**Author:** Internal AI Tools Project  
**Date:** 2025-10-04  
**Version:** 1.0
