import numpy as np


class FaceRecognizer:
    def __init__(self, recognition_threshold=0.45, min_face_size=40):
        try:
            from insightface.app import FaceAnalysis
        except ImportError as exc:
            raise RuntimeError("Install InsightFace and ONNX Runtime with pip install -r requirements.txt") from exc
        # InsightFace downloads its default recognition/detection models on first run.
        self.app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        self.app.prepare(ctx_id=-1, det_size=(320, 320))
        self.threshold = float(recognition_threshold)
        self.min_face_size = int(min_face_size)

    def embedding(self, face_crop):
        if face_crop is None or face_crop.size == 0:
            return None
        h, w = face_crop.shape[:2]
        if min(h, w) < self.min_face_size:
            return None
        faces = self.app.get(face_crop)
        if not faces:
            return None
        # Use the highest-detection-score face if the crop contains more than one.
        face = max(faces, key=lambda f: float(getattr(f, "det_score", 0.0)))
        emb = np.asarray(face.normed_embedding, dtype=np.float32)
        norm = np.linalg.norm(emb)
        return emb / norm if norm else None

    def identify(self, embedding, visitors):
        if embedding is None or not visitors:
            return None, -1.0
        best_id, best_score = None, -1.0
        for visitor in visitors:
            known = visitor["embedding"]
            if known.shape != embedding.shape:
                continue
            score = float(np.dot(embedding, known) / ((np.linalg.norm(embedding) * np.linalg.norm(known)) + 1e-12))
            if score > best_score:
                best_id, best_score = visitor["face_id"], score
        if best_score >= self.threshold:
            return best_id, best_score
        return None, best_score
