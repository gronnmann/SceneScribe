# 🧠 Copilot Instructions: TikTok Video Intelligence Extractor

This document defines how GitHub Copilot (or any code generation AI) should behave when generating the code for this project.

---

## 🎯 Mission
Create a **Python-based pipeline** that analyzes TikTok videos, transcribes the speech, describes scenes, and exports a JSON summary of each video.

The goal is **accuracy, modularity, and reproducibility** — every module should be clear, testable, and built to run on a **CUDA-enabled GPU (RTX 3090)**.

---

## 🧰 Project Context
This project already has a explaination and specification shown in `README.md`
Copilot should refer to that document as the *single source of truth* for structure and functionality.

The project must be fully **open-source**, using **publicly available models and libraries** only.

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

## 🧩 Development Goals
- Modular and scalable architecture
- Full GPU utilization (WhisperX + BLIP-2)
- Reliable JSON output per video
- Multi-language support (Norwegian + English)
- Ready for later integration with LLM-based creative systems