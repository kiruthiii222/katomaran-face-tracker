# Katomaran Intelligent Face Tracker

A modular Python prototype for face detection, face-embedding-based visitor re-identification, frame tracking, entry/exit event logging, and unique visitor counting.

> **Privacy note:** Face embeddings and cropped face images are biometric data. Run this only on video you are authorized to process, notify people where required, restrict access to generated data, and delete it when no longer needed. This demo is not a security-grade identity system.

## Features

- YOLO face detection (requires a face-trained YOLO weights file; generic COCO YOLO weights detect people, not faces).
- InsightFace embeddings for comparing detected faces to previously registered visitors.
- Automatic registration of new visitor IDs in SQLite.
- IoU-based short-term tracking across detection cycles.
- One entry event when a track first appears and one exit event when it disappears or the stream ends.
- UTC timestamps, cropped face images, SQLite event metadata, and `logs/events.log`.
- Configurable detection interval via `config.json`.
- Annotated MP4 output and on-screen visitor count.
- Video file, webcam index, or RTSP URL input.

## Architecture

```mermaid
flowchart TD
    A[Video file / Webcam / RTSP] --> B[YOLO Face Detector]
    B --> C[Face Crop]
    C --> D[InsightFace Embedding]
    D --> E{Match registered embedding?}
    E -- Yes --> F[Existing face_id]
    E -- No --> G[Register new visitor in SQLite]
    G --> H[New face_id]
    F --> I[IoU Tracker]
    H --> I
    I --> J[Entry / Exit Event Logger]
    J --> K[SQLite visitors + events]
    J --> L[Timestamped cropped images]
    J --> M[events.log]
    I --> N[Annotated video + visitor count]
```

## Requirements

- Python 3.10 or 3.11 recommended.
- Windows, macOS, or Linux.
- 8 GB RAM recommended for a CPU demo; GPU is optional. InsightFace uses ONNX Runtime CPU provider in this starter implementation.
- A YOLO face model weights file. Do not use the standard `yolov8n.pt` COCO model as a face detector.
- The sample video supplied with the hackathon task. Download it from the Drive folder shared in the task document and save it as `sample_video.mp4` in this project root.

## Setup (Windows PowerShell)

```powershell
cd katomaran-face-tracker
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, use Command Prompt and run `.venv\\Scripts\\activate.bat` instead.

## Model setup

1. Create a `models` folder in the project root.
2. Obtain a compatible YOLO face-detection weights file from a reputable model repository, after reviewing its license and model source.
3. Save it as `models/yolov8n-face.pt`, or update `face_model_path` in `config.json` to your actual file path.
4. InsightFace's `buffalo_l` model pack is typically downloaded automatically on first run. The first run therefore needs internet access. Model downloads can take time.

The YOLO face detection model weights are included in this repository at `models/yolov8n-face.pt`. The project uses Ultralytics YOLO for face detection and InsightFace for face embeddings and recognition. The model is loaded locally during execution.


## Run

Place the provided video at `sample_video.mp4`, then run:

```powershell
python main.py
```

To use another video:

```powershell
python main.py --source "C:\\videos\\entrance.mp4"
```

To use a webcam:

```powershell
python main.py --source 0
```

To use an authorized RTSP camera stream:

```powershell
python main.py --source "rtsp://username:password@camera-host:554/stream"
```

Press `q` to stop the preview. Do not commit RTSP credentials or private camera URLs to GitHub.

## Configuration

`config.json` example:

```json
{
  "video_source": "sample_video.mp4",
  "face_model_path": "models/yolov8n-face.pt",
  "yolo_confidence": 0.45,
  "detection_skip_frames": 2,
  "recognition_threshold": 0.45,
  "max_frames_lost": 15,
  "min_face_size": 40,
  "save_annotated_video": true,
  "output_video_path": "output/annotated_video.mp4",
  "database_path": "database/visitors.db",
  "logs_dir": "logs",
  "rtsp_reconnect_seconds": 3,
  "show_window": true
}
```

- `detection_skip_frames: 1` means detect on every frame; `2` means every other frame.
- `recognition_threshold` is a cosine-similarity threshold. Tune it against the supplied video; a lower threshold can merge different people, while a higher threshold can create duplicate identities.
- `max_frames_lost` is measured in detection cycles, not raw frames.
- Set `show_window` to `false` for headless execution.

## Generated output

- `database/visitors.db`: `visitors` and `events` tables.
- `logs/events.log`: registration, embedding, tracking, entry, exit, and run lifecycle messages.
- `logs/entries/YYYY-MM-DD/`: cropped face images captured at entry.
- `logs/exits/YYYY-MM-DD/`: cropped face images captured at exit.
- `output/annotated_video.mp4`: video with track IDs and unique visitor count.

You must run the program against the supplied video before submission and include real output files or screenshots in `sample_results/`. Do not create fake logs or claim untested results.

To inspect database rows using Python:

```python
import sqlite3
conn = sqlite3.connect("database/visitors.db")
print("Visitors:", conn.execute("SELECT face_id, first_seen FROM visitors").fetchall())
print("Events:", conn.execute("SELECT face_id, event_type, timestamp, image_path FROM events").fetchall())
conn.close()
```

## Assumptions and limitations

- A new visitor identity is assigned when no stored embedding meets the configured similarity threshold.
- The visitor count means the number of unique face identities registered in this database, not the number of entry events.
- Entry and exit are inferred from a track appearing and then disappearing for more than `max_frames_lost` detection cycles. At end-of-file, all active tracks are marked as exited.
- The starter tracker uses greedy IoU matching. It may lose a track during long occlusions, crowded scenes, or abrupt movement. ByteTrack/DeepSORT and temporal smoothing are possible improvements.
- This starter version opens a provided RTSP stream but does not automatically reconnect after a network interruption.
- Lighting, face size, camera angle, model quality, and threshold choice affect recognition accuracy. Similarity scores are not proof of a person's real-world identity.
- Repeated appearances after a track has ended can create a new entry/exit event while reusing the same `face_id`, which is intended for repeated visits.
- CPU load depends on input resolution, frame rate, detection interval, and model size. On CPU, face detection and InsightFace embedding extraction are the main compute costs. GPU use is not enabled in this starter code.

## AI planning document

See [`docs/AI_PLANNING.md`](docs/AI_PLANNING.md) for the planned workflow, modules, validation plan, compute estimate, and example prompts used to develop and review the code.

## Demo video (mandatory)

Record a Loom or YouTube video showing:
1. The repository and project architecture.
2. Installing dependencies and setting the model/video paths.
3. Running `python main.py` on the sample video.
4. Face IDs and unique visitor count in the preview.
5. `events.log`, cropped entry/exit images, and database rows.
6. The configurable `detection_skip_frames` value.

Replace this line with your real, accessible demo link before submission:

`Demo video: ADD_YOUR_LOOM_OR_YOUTUBE_URL_HERE`

## Suggested GitHub upload

```bash
git init
git add .
git commit -m "Build intelligent face tracker hackathon prototype"
git branch -M main
git remote add origin https://github.com/kiruthiii222/katomaran-face-tracker
git push -u origin main
```

Do not commit private videos, face images, database files, credentials, or model weights unless you have permission to share them. For the hackathon submission, provide representative real output evidence separately if the sample video or biometric data cannot be made public.

This project is a part of a hackathon run by https://katomaran.com
