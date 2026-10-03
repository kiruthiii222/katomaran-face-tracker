from pathlib import Path
import cv2


class FaceDetector:
    """YOLO-based face detector. The configured weights must be face-trained weights."""
    def __init__(self, model_path, confidence=0.45):
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Face model not found: {model_path}. Download a YOLO face-detection model "
                "and place it at this path; see README.md. A generic YOLO COCO model detects people, not faces."
            )
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("Install requirements first: pip install -r requirements.txt") from exc
        self.model = YOLO(str(model_path))
        self.confidence = float(confidence)

    def detect(self, frame):
        result = self.model.predict(frame, conf=self.confidence, verbose=False)[0]
        detections = []
        if result.boxes is None:
            return detections
        for box in result.boxes:
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            h, w = frame.shape[:2]
            x1, x2 = max(0, x1), min(w, x2)
            y1, y2 = max(0, y1), min(h, y2)
            if x2 > x1 and y2 > y1:
                detections.append({"bbox": (x1, y1, x2, y2), "confidence": float(box.conf[0])})
        return detections
