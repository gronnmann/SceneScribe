# Video Input Directory

Place your video files here for processing.

## Supported Formats

- `.mp4` (recommended)
- `.avi`
- `.mov`
- `.mkv`
- `.webm`

## Example

```
data/videos/
├── tiktok_ad_001.mp4
├── tiktok_ad_002.mp4
└── tiktok_ad_003.mp4
```

## Notes

- Files can have any name (alphanumeric recommended)
- The filename (without extension) will be used as the `video_id`
- Ensure videos have audio tracks for transcription
- Recommended: videos under 5 minutes for faster processing

## Processing

To process all videos in this directory:

```bash
python src/main.py --input data/videos --output outputs/json
```

To process a single video:

```bash
python src/main.py --input data/videos/my_video.mp4 --output outputs/json
```
