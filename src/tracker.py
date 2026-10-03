from dataclasses import dataclass
import numpy as np


def iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    x1, y1 = max(ax1, bx1), max(ay1, by1)
    x2, y2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union else 0.0


@dataclass
class Track:
    track_id: int
    bbox: tuple
    face_id: str
    last_crop: object
    last_seen_frame: int
    missed: int = 0
    entry_logged: bool = False


class IoUTracker:
    """Lightweight IoU tracker. A track ends after max_frames_lost missed detection cycles."""
    def __init__(self, max_frames_lost=15, iou_threshold=0.25):
        self.max_frames_lost = int(max_frames_lost)
        self.iou_threshold = float(iou_threshold)
        self.tracks = {}
        self.next_id = 1

    def update(self, detections, frame_number):
        unmatched_tracks = set(self.tracks.keys())
        assignments = []
        # Greedy highest-IoU matching; adequate for a compact demo, not a multi-camera tracker.
        pairs = []
        for det_index, det in enumerate(detections):
            for track_id, track in self.tracks.items():
                score = iou(det["bbox"], track.bbox)
                if score >= self.iou_threshold:
                    pairs.append((score, det_index, track_id))
        used_dets, used_tracks = set(), set()
        for score, det_index, track_id in sorted(pairs, reverse=True):
            if det_index in used_dets or track_id in used_tracks:
                continue
            used_dets.add(det_index)
            used_tracks.add(track_id)
            unmatched_tracks.discard(track_id)
            assignments.append((det_index, track_id))
        for track_id in unmatched_tracks:
            self.tracks[track_id].missed += 1
        for det_index, det in enumerate(detections):
            if det_index not in used_dets:
                track_id = self.next_id
                self.next_id += 1
                self.tracks[track_id] = Track(track_id, det["bbox"], det["face_id"], det.get("crop"), frame_number)
                assignments.append((det_index, track_id))
            else:
                track_id = next(tid for di, tid in assignments if di == det_index)
                track = self.tracks[track_id]
                track.bbox = det["bbox"]
                track.face_id = det["face_id"]
                track.last_crop = det.get("crop") if det.get("crop") is not None else track.last_crop
                track.last_seen_frame = frame_number
                track.missed = 0
        ended = []
        for track_id, track in list(self.tracks.items()):
            if track.missed > self.max_frames_lost:
                ended.append(track)
                del self.tracks[track_id]
        return assignments, ended
