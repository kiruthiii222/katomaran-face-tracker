import argparse
import json
import logging
import time
from pathlib import Path
import cv2

from src.database import Database
from src.detector import FaceDetector
from src.face_recognizer import FaceRecognizer
from src.tracker import IoUTracker
from src.event_logger import EventLogger
from src.visitor_counter import VisitorCounter


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def crop_face(frame, bbox):
    x1, y1, x2, y2 = bbox
    return frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)].copy()


def run(config, source_override=None):
    source = source_override if source_override is not None else config["video_source"]
    db = Database(config["database_path"])
    event_logger = EventLogger(config["logs_dir"], db)
    detector = FaceDetector(config["face_model_path"], config.get("yolo_confidence", 0.45))
    recognizer = FaceRecognizer(config.get("recognition_threshold", 0.45), config.get("min_face_size", 40))
    tracker = IoUTracker(config.get("max_frames_lost", 15))
    counter = VisitorCounter(db)
    known_visitors = db.all_visitors()

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        db.close()
        raise RuntimeError(f"Could not open video/camera source: {source}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0 or fps > 120:
        fps = 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    writer = None
    if config.get("save_annotated_video", True):
        out_path = Path(config.get("output_video_path", "output/annotated_video.mp4"))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    skip = max(1, int(config.get("detection_skip_frames", 1)))
    frame_number = 0
    last_detections = []
    event_logger.info(f"START source={source}; detection_skip_frames={skip}")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_number += 1
            # Run face detection only once per configured interval. Tracks persist between cycles.
            if (frame_number - 1) % skip == 0:
                raw_detections = detector.detect(frame)
                detections = []
                for item in raw_detections:
                    crop = crop_face(frame, item["bbox"])
                    embedding = recognizer.embedding(crop)
                    if embedding is None:
                        continue
                    face_id, similarity = recognizer.identify(embedding, known_visitors)
                    if face_id is None:
                        face_id = f"visitor_{len(known_visitors) + 1:05d}"
                        db.register_visitor(face_id, embedding)
                        known_visitors.append({"face_id": face_id, "embedding": embedding})
                        event_logger.info(f"EMBEDDING_GENERATED face_id={face_id}; similarity={similarity:.4f}")
                        event_logger.info(f"VISITOR_REGISTERED face_id={face_id}")
                    detections.append({"bbox": item["bbox"], "face_id": face_id, "crop": crop, "confidence": item["confidence"]})
                last_detections = detections
                assignments, ended_tracks = tracker.update(detections, frame_number)
                for det_index, track_id in assignments:
                    # New tracks are identified by their first-seen frame; only log entry once.
                    track = tracker.tracks.get(track_id)
                    if track is not None and track.last_seen_frame == frame_number and not getattr(track, "entry_logged", False):
                        event_logger.record_event(track.face_id, "entry", track_id, track.last_crop)
                        track.entry_logged = True
                        event_logger.info(f"TRACKING_STARTED track_id={track_id} face_id={track.face_id}")
                for track in ended_tracks:
                    event_logger.record_event(track.face_id, "exit", track.track_id, track.last_crop)
                    event_logger.info(f"TRACKING_ENDED track_id={track.track_id} face_id={track.face_id}")

            # Draw current active tracks; bounding boxes are refreshed on detection frames.
            for track in tracker.tracks.values():
                x1, y1, x2, y2 = track.bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 2)
                cv2.putText(frame, f"{track.face_id} / T{track.track_id}", (x1, max(20, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 0), 2)
            cv2.putText(frame, f"Unique visitors: {counter.unique_visitors}", (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
            if writer is not None:
                writer.write(frame)
            if config.get("show_window", True):
                cv2.imshow("Katomaran Face Tracker (press q to quit)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        event_logger.info("INTERRUPTED_BY_USER")
    finally:
        # If the stream ends while tracks are active, emit exactly one exit per active track.
        for track in list(tracker.tracks.values()):
            event_logger.record_event(track.face_id, "exit", track.track_id, track.last_crop)
            event_logger.info(f"TRACKING_ENDED track_id={track.track_id} face_id={track.face_id} reason=stream_end")
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
        event_logger.info(f"STOP unique_visitors={counter.unique_visitors}; total_events={db.event_count()}")
        print(f"Unique visitors registered: {counter.unique_visitors}")
        print(f"Total entry/exit events: {db.event_count()}")
        db.close()


def main():
    parser = argparse.ArgumentParser(description="Katomaran Intelligent Face Tracker")
    parser.add_argument("--config", default="config.json", help="Path to config JSON")
    parser.add_argument("--source", default=None, help="Override video source with file path, camera index, or RTSP URL")
    args = parser.parse_args()
    config = load_config(args.config)
    source = args.source
    if source is not None and source.isdigit():
        source = int(source)
    run(config, source)


if __name__ == "__main__":
    main()
