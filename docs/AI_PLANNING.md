# AI Planning and Development Notes

## 1. Problem statement
Build a video-processing prototype that detects faces, assigns persistent visitor identities using facial embeddings, tracks people in a camera view, records entry/exit events, and calculates unique registered visitors.

## 2. Implementation plan
1. Create a modular Python application with configuration separated from code.
2. Read frames from a sample video, webcam, or RTSP URL using OpenCV.
3. Run a face-trained YOLO detector every `detection_skip_frames` frames.
4. Crop detected faces and use InsightFace to create normalized embeddings.
5. Compare each embedding with saved embeddings using cosine similarity.
6. Register a new SQLite visitor row when no identity crosses the threshold.
7. Associate detections with active tracks using Intersection over Union (IoU).
8. Log one entry when a track is created and one exit when it disappears or the stream ends.
9. Save cropped images by event type and UTC date; save event metadata in SQLite and messages in `events.log`.
10. Draw bounding boxes and visitor count on the output video.
11. Run the provided sample video and inspect logs, images, database rows, and annotated video.
12. Record an explanatory demo and update the README with the real URL.

## 3. Modules
- `main.py`: orchestration, video input, frame processing, overlay and output.
- `src/detector.py`: YOLO face detection.
- `src/face_recognizer.py`: InsightFace embeddings and identity matching.
- `src/tracker.py`: IoU-based track association and lost-track handling.
- `src/database.py`: SQLite schema and visitor/event persistence.
- `src/event_logger.py`: timestamped image saving and mandatory text log.
- `src/visitor_counter.py`: unique visitor count from database records.
- `config.json`: paths and tunable parameters.

## 4. Data model
### visitors
- `face_id`: primary key, e.g. `visitor_00001`.
- `first_seen`: UTC ISO-8601 timestamp.
- `embedding`: normalized embedding serialized as JSON text.

### events
- `event_id`: auto-incrementing primary key.
- `face_id`: registered identity associated with the event.
- `event_type`: `entry` or `exit`.
- `timestamp`: UTC ISO-8601 timestamp.
- `image_path`: path to the event's cropped face image.
- `track_id`: local track instance identifier.

## 5. Assumptions
- Input video is readable by OpenCV.
- The YOLO weights file is face-trained and compatible with Ultralytics.
- The InsightFace model pack can be downloaded or is already cached.
- A visitor's appearance is sufficiently similar across frames for embedding matching.
- A track ending after a configurable number of missed detection cycles is treated as an exit.
- A visitor returning later may create a new track and a new entry/exit pair but should reuse the same `face_id` if the embedding matches.

## 6. Validation plan
- [ ] Verify video opens and frames are read.
- [ ] Verify face detections draw correct boxes.
- [ ] Verify first-seen face creates a visitor row.
- [ ] Verify repeat frames do not create a new visitor row.
- [ ] Verify each new track produces one entry event.
- [ ] Verify a lost track produces one exit event.
- [ ] Verify entry and exit images exist and paths are recorded in SQLite.
- [ ] Verify `events.log` contains registration, tracking, entry, and exit messages.
- [ ] Verify unique visitor count equals the number of rows in `visitors`.
- [ ] Run with detection skip values of 1 and 2.
- [ ] Inspect annotated output video and record a demo.

## 7. Compute estimate (qualitative)
The implementation uses CPU execution for InsightFace and Ultralytics' configured device selection. Actual utilization depends on CPU, resolution, frame rate, number of faces, and model size. The highest costs are YOLO inference on detection frames and InsightFace embedding extraction per face. Increasing `detection_skip_frames` generally reduces detector work but can make tracks less stable and delay event transitions. Measure actual CPU/RAM usage on the target machine rather than claiming a benchmark that was not performed. GPU acceleration is not explicitly configured in this starter version.

## 8. AI-assisted development prompts (examples)
- "Design a modular Python architecture for YOLO face detection, InsightFace embeddings, SQLite visitor registration, and timestamped entry/exit events."
- "Review this IoU tracker for duplicate entry/exit events and explain its limitations under occlusion."
- "Check that the SQLite schema stores visitor metadata and one row per entry/exit event."
- "Create a test checklist for verifying face re-identification, local image storage, and database consistency."

AI-generated code must be reviewed and understood before the interview. The developer should be able to explain embedding similarity, the detection skip parameter, IoU matching, the database schema, and the known limitations.
