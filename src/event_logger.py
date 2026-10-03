import logging
from pathlib import Path
from datetime import datetime, timezone
import cv2


class EventLogger:
    def __init__(self, logs_dir, database):
        self.logs_dir = Path(logs_dir)
        self.database = database
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger("katomaran_face_tracker")
        self.logger.setLevel(logging.INFO)
        self.logger.handlers.clear()
        handler = logging.FileHandler(self.logs_dir / "events.log", encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        self.logger.addHandler(handler)

    @staticmethod
    def timestamp():
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    def record_event(self, face_id, event_type, track_id, face_crop):
        ts = self.timestamp()
        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        folder = self.logs_dir / ("entries" if event_type == "entry" else "exits") / day
        folder.mkdir(parents=True, exist_ok=True)
        safe_ts = ts.replace(":", "-")
        image_path = folder / f"{face_id}_track{track_id}_{safe_ts}.jpg"
        saved_path = None
        if face_crop is not None and face_crop.size:
            if cv2.imwrite(str(image_path), face_crop):
                saved_path = image_path
        self.database.add_event(face_id, event_type, ts, saved_path, track_id)
        self.logger.info("FACE_%s face_id=%s track_id=%s timestamp=%s image=%s", event_type.upper(), face_id, track_id, ts, saved_path)
        return ts, saved_path

    def info(self, message):
        self.logger.info(message)

    def error(self, message):
        self.logger.exception(message)
