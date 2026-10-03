import sqlite3
from pathlib import Path
from datetime import datetime, timezone
import json
import numpy as np


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Database:
    def __init__(self, db_path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self):
        self.conn.executescript("""
        PRAGMA journal_mode=WAL;
        CREATE TABLE IF NOT EXISTS visitors (
            face_id TEXT PRIMARY KEY,
            first_seen TEXT NOT NULL,
            embedding TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS events (
            event_id INTEGER PRIMARY KEY AUTOINCREMENT,
            face_id TEXT NOT NULL,
            event_type TEXT NOT NULL CHECK(event_type IN ('entry','exit')),
            timestamp TEXT NOT NULL,
            image_path TEXT,
            track_id INTEGER,
            FOREIGN KEY(face_id) REFERENCES visitors(face_id)
        );
        """)
        self.conn.commit()

    def all_visitors(self):
        rows = self.conn.execute("SELECT face_id, first_seen, embedding FROM visitors").fetchall()
        result = []
        for row in rows:
            try:
                emb = np.asarray(json.loads(row["embedding"]), dtype=np.float32)
                result.append({"face_id": row["face_id"], "first_seen": row["first_seen"], "embedding": emb})
            except (ValueError, TypeError, json.JSONDecodeError):
                continue
        return result

    def register_visitor(self, face_id, embedding):
        self.conn.execute(
            "INSERT INTO visitors(face_id, first_seen, embedding) VALUES (?, ?, ?)",
            (face_id, utc_now(), json.dumps(np.asarray(embedding, dtype=float).tolist()))
        )
        self.conn.commit()

    def add_event(self, face_id, event_type, timestamp, image_path, track_id):
        self.conn.execute(
            "INSERT INTO events(face_id, event_type, timestamp, image_path, track_id) VALUES (?, ?, ?, ?, ?)",
            (face_id, event_type, timestamp, str(image_path) if image_path else None, track_id)
        )
        self.conn.commit()

    def unique_visitor_count(self):
        return self.conn.execute("SELECT COUNT(*) FROM visitors").fetchone()[0]

    def event_count(self):
        return self.conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]

    def close(self):
        self.conn.close()
