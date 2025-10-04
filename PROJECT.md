# TikTok Video Intelligence Extractor

## 🎯 Project Title

**TikTok Video Intelligence Extractor**  
_(Working name — can be changed later)_

---

## 🧩 Project Overview

The **TikTok Video Intelligence Extractor** is a Python-based pipeline that processes TikTok video files, analyzes their **audio** and **visual** content using **open-source AI models**, and exports detailed **JSON files** describing each video.

These JSONs will later be used as structured input for large language models (LLMs) to **generate similar ads**, **analyze creative style**, or perform **content intelligence**.

---

## 🧠 Core Objectives

1. **Ingest** a directory of TikTok video files (`.mp4`).
2. **Transcribe** speech using Whisper-based ASR (supports Norwegian and English).
3. **Detect scenes** and extract representative keyframes.
4. **Describe** visuals per scene using captioning and object detection models.
5. Optionally perform **OCR** and **speaker diarization**.
6. **Export** all metadata as a structured JSON file per video.

---

## 🧰 Tech Stack

| Area                            | Tool / Library                                                           | Purpose                                                        |
| ------------------------------- | ------------------------------------------------------------------------ | -------------------------------------------------------------- |
| **Language**                    | Python 3.10+                                                             | Primary language                                               |
| **Video Processing**            | `ffmpeg`, `PySceneDetect`                                                | Normalization, scene detection, keyframe extraction            |
| **Audio Processing**            | `ffmpeg`, `librosa`                                                      | Extract audio for transcription                                |
| **Speech-to-Text (ASR)**        | `WhisperX` with `openai/whisper-large-v2` and `NbAiLab/nb-whisper-large` | High-accuracy multilingual transcription (English + Norwegian) |
| **Diarization (optional)**      | `pyannote.audio`                                                         | Identify speakers in dialogues                                 |
| **Image Captioning**            | `BLIP-2` (Hugging Face Transformers)                                     | Generate captions and descriptions of keyframes                |
| **Object Detection (optional)** | `GroundingDINO`                                                          | Zero-shot object detection                                     |
| **OCR (optional)**              | `Tesseract` or `PaddleOCR`                                               | Extract visible text from frames                               |
| **Data Handling**               | `pandas`, `json`, `pathlib`                                              | Organize and export structured outputs                         |
| **Orchestration**               | CLI-based Python workflow (expandable to Prefect / Airflow)              | Manage pipeline stages                                         |
| **Environment**                 | Docker + CUDA (PyTorch GPU support)                                      | GPU acceleration (NVIDIA RTX 3090)                             |

---

## ⚙️ System Architecture

### 1. Input

- A folder containing `.mp4` videos.
- Each file named arbitrarily (video ID or TikTok URL slug optional).

### 2. Preprocessing

- Normalize format and audio using `ffmpeg`.
- Extract 16 kHz mono `.wav` audio.
- Detect scenes using **PySceneDetect** and extract keyframes (JPGs).

### 3. Audio Analysis

- Run **WhisperX** transcription with language auto-detection.
- Use:
  - **Norwegian model:** `NbAiLab/nb-whisper-large`
  - **English model:** `openai/whisper-large-v2`
- Output includes:
  - Full transcript text
  - Word-level timestamps
  - (Optional) speaker diarization

### 4. Visual Analysis

- For each keyframe:
  - Generate 1–3 **captions** using BLIP-2.
  - Detect **objects** with GroundingDINO (optional).
  - Extract **on-screen text** with OCR (optional).

### 5. Data Fusion

- Merge transcript, scene, and image metadata into a unified JSON.
- Link audio timestamps with shot time intervals.

### 6. Output

- For each input video: `outputs/{video_name}.json`
- Optional: keyframes stored in `outputs/{video_name}/frames/`.

---

## 📄 Example JSON Output

```json
{
  "video_id": "tiktok_001",
  "filename": "video_001.mp4",
  "duration_s": 32.5,
  "language": "no",
  "transcript": {
    "text": "Hei alle sammen, i dag viser jeg deg hvordan...",
    "words": [
      { "word": "Hei", "start_s": 0.12, "end_s": 0.35 },
      { "word": "alle", "start_s": 0.36, "end_s": 0.56 }
    ]
  },
  "shots": [
    {
      "shot_id": "s001",
      "start_s": 0.0,
      "end_s": 5.1,
      "keyframe": "frames/video_001_s001.jpg",
      "captions": [
        {
          "text": "A woman smiling at the camera holding a coffee mug",
          "confidence": 0.89
        }
      ],
      "ocr_text": "",
      "objects": [
        { "label": "person", "confidence": 0.97 },
        { "label": "cup", "confidence": 0.83 }
      ]
    }
  ],
  "processing": {
    "created_at": "2025-10-04T18:30:00Z",
    "models": {
      "asr": "nb-whisper-large-v2",
      "caption": "blip2-large",
      "object_detector": "grounding-dino",
      "ocr": "tesseract"
    },
    "version": "0.1.0"
  }
}
```

# Folder Strtucture
```
tiktok_ai_extractor/
├── data/
│   └── videos/             # Input videos (.mp4)
├── outputs/
│   ├── json/               # Output JSONs
│   └── frames/             # Extracted keyframes
├── src/
│   ├── __init__.py
│   ├── main.py             # CLI entrypoint
│   ├── video_preprocess.py # ffmpeg + PySceneDetect utilities
│   ├── asr.py              # WhisperX transcription
│   ├── vision.py           # BLIP-2 captioning + GroundingDINO
│   ├── ocr.py              # Tesseract integration
│   ├── fusion.py           # Combine results → JSON
│   └── utils.py
├── requirements.txt
├── Dockerfile
├── README.md
└── agent_prompt.md         # CoPilot system prompt file
```