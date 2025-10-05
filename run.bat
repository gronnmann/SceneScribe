@echo off
REM Windows batch script to run the pipeline

setlocal EnableDelayedExpansion

echo ===================================================
echo TikTok Video Intelligence Extractor - Run Script
echo ===================================================
echo.

REM Configuration
set INPUT_DIR=data\videos
set OUTPUT_DIR=outputs\json
set FRAMES_DIR=outputs\frames
set SCENE_THRESHOLD=27.0
set NUM_CAPTIONS=1
set LOG_LEVEL=INFO

REM Check if input directory exists
if not exist "%INPUT_DIR%" (
    echo Error: Input directory %INPUT_DIR% does not exist
    exit /b 1
)

REM Count video files
set VIDEO_COUNT=0
for %%f in ("%INPUT_DIR%\*.mp4" "%INPUT_DIR%\*.avi" "%INPUT_DIR%\*.mov" "%INPUT_DIR%\*.mkv" "%INPUT_DIR%\*.webm") do (
    set /a VIDEO_COUNT+=1
)

if %VIDEO_COUNT%==0 (
    echo Error: No video files found in %INPUT_DIR%
    echo Please add video files ^(.mp4, .avi, .mov, .mkv, .webm^) to the directory
    exit /b 1
)

echo Found %VIDEO_COUNT% video^(s^) to process
echo.

REM Create output directories
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"
if not exist "%FRAMES_DIR%" mkdir "%FRAMES_DIR%"

REM Run pipeline
echo Starting pipeline...
echo.

python src\main.py ^
    --input "%INPUT_DIR%" ^
    --output "%OUTPUT_DIR%" ^
    --frames-dir "%FRAMES_DIR%" ^
    --scene-threshold %SCENE_THRESHOLD% ^
    --num-captions %NUM_CAPTIONS% ^
    --log-level %LOG_LEVEL%

echo.
echo ===================================================
echo Processing complete!
echo ===================================================
echo.
echo Output JSON files: %OUTPUT_DIR%
echo Extracted keyframes: %FRAMES_DIR%
echo.

endlocal
