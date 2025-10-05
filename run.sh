#!/bin/bash
# Example script to run the pipeline

set -e

echo "==================================================="
echo "TikTok Video Intelligence Extractor - Run Script"
echo "==================================================="
echo ""

# Configuration
INPUT_DIR="data/videos"
OUTPUT_DIR="outputs/json"
FRAMES_DIR="outputs/frames"
LANGUAGE=""  # Auto-detect
SCENE_THRESHOLD=27.0
NUM_CAPTIONS=1
LOG_LEVEL="INFO"

# Check if input directory has videos
if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Input directory $INPUT_DIR does not exist"
    exit 1
fi

VIDEO_COUNT=$(find "$INPUT_DIR" -type f \( -name "*.mp4" -o -name "*.avi" -o -name "*.mov" -o -name "*.mkv" -o -name "*.webm" \) | wc -l)

if [ "$VIDEO_COUNT" -eq 0 ]; then
    echo "Error: No video files found in $INPUT_DIR"
    echo "Please add video files (.mp4, .avi, .mov, .mkv, .webm) to the directory"
    exit 1
fi

echo "Found $VIDEO_COUNT video(s) to process"
echo ""

# Create output directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$FRAMES_DIR"

# Run pipeline
echo "Starting pipeline..."
echo ""

python src/main.py \
    --input "$INPUT_DIR" \
    --output "$OUTPUT_DIR" \
    --frames-dir "$FRAMES_DIR" \
    --scene-threshold "$SCENE_THRESHOLD" \
    --num-captions "$NUM_CAPTIONS" \
    --log-level "$LOG_LEVEL"

echo ""
echo "==================================================="
echo "Processing complete!"
echo "==================================================="
echo ""
echo "Output JSON files: $OUTPUT_DIR"
echo "Extracted keyframes: $FRAMES_DIR"
echo ""
