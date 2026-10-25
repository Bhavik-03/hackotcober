"""
detector.py
Pothole detector. Tries to pull a pothole-finetuned YOLOv8 model from
Hugging Face; falls back to stock YOLOv8n (COCO) in "demo mode" so the
pipeline never blocks on a download during the hackathon.

Usage:
    from detector import detect_potholes
    results = detect_potholes("data/sample.jpg")
"""
import os
from pathlib import Path

MODEL_DIR = Path(__file__).parent / "models"
MODEL_DIR.mkdir(exist_ok=True)
POTHOLE_MODEL_PATH = MODEL_DIR / "pothole_yolov8.pt"

# Public, pre-trained pothole model on Hugging Face (fallback chain tried in order).
HF_CANDIDATES = [
    ("Samdutse/pothole-yolov8", "best.pt"),
    ("vinothvikas1987/pothole-detection-yolov8", "best.pt"),
]


def _try_download_pothole_model() -> str | None:
    """Try each HF candidate; return a local path if one downloads, else None."""
    if POTHOLE_MODEL_PATH.exists():
        return str(POTHOLE_MODEL_PATH)
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        return None

    for repo_id, filename in HF_CANDIDATES:
        try:
            path = hf_hub_download(repo_id=repo_id, filename=filename)
            return path
        except Exception as e:  # noqa: BLE001 - we just want a graceful fallback
            print(f"[detector] could not fetch {repo_id}: {e}")
            continue
    return None


_model = None
_mode = "unknown"


def _get_model():
    global _model, _mode
    if _model is not None:
        return _model, _mode

    from ultralytics import YOLO

    hf_path = _try_download_pothole_model()
    if hf_path:
        _model = YOLO(hf_path)
        _mode = "pothole-finetuned"
    else:
        # Fallback: stock COCO model. We treat any detected object as a
        # stand-in "road anomaly" for demo purposes and say so explicitly
        # in the output, rather than pretending it's pothole-specific.
        _model = YOLO("yolov8n.pt")
        _mode = "demo-fallback-coco"

    print(f"[detector] running in mode: {_mode}")
    return _model, _mode


def detect_potholes(image_path: str, conf: float = 0.25) -> dict:
    """Run detection on an image and return a plain dict (JSON-friendly)."""
    model, mode = _get_model()
    results = model.predict(source=image_path, conf=conf, verbose=False)
    r = results[0]

    detections = []
    for box in r.boxes:
        xyxy = box.xyxy[0].tolist()
        detections.append({
            "label": "pothole" if mode == "pothole-finetuned" else model.names[int(box.cls[0])],
            "confidence": round(float(box.conf[0]), 3),
            "bbox_xyxy": [round(v, 1) for v in xyxy],
        })

    annotated_path = str(Path(image_path).with_name(Path(image_path).stem + "_annotated.jpg"))
    r.save(filename=annotated_path)

    return {
        "image_path": image_path,
        "annotated_path": annotated_path,
        "mode": mode,
        "num_detections": len(detections),
        "detections": detections,
    }


def detect_potholes_video(video_path: str, sample_every_n_frames: int = 15, conf: float = 0.25) -> dict:
    """Sample every Nth frame of a video, run the pothole model, aggregate counts.

    No tracking/dedup across frames: total_detections is a plain sum, so the
    same pothole seen in several sampled frames is counted each time.
    """
    import cv2

    model, mode = _get_model()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    step = max(1, int(sample_every_n_frames))
    per_frame_counts = []
    total_detections = 0
    best_conf = -1.0
    best_annotated = None  # BGR ndarray of the best frame
    best_detections = []
    best_frame_idx = None

    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % step == 0:
                r = model.predict(source=frame, conf=conf, verbose=False)[0]
                per_frame_counts.append(len(r.boxes))
                total_detections += len(r.boxes)
                if len(r.boxes):
                    top = float(r.boxes.conf.max())
                    if top > best_conf:
                        best_conf = top
                        best_frame_idx = frame_idx
                        best_annotated = r.plot()
                        best_detections = [{
                            "label": "pothole" if mode == "pothole-finetuned" else model.names[int(b.cls[0])],
                            "confidence": round(float(b.conf[0]), 3),
                            "bbox_xyxy": [round(v, 1) for v in b.xyxy[0].tolist()],
                        } for b in r.boxes]
            frame_idx += 1
    finally:
        cap.release()

    annotated_path = None
    if best_annotated is not None:
        annotated_path = str(Path(video_path).with_name(Path(video_path).stem + "_best_frame.jpg"))
        cv2.imwrite(annotated_path, best_annotated)

    return {
        "source": "video",
        "video_path": video_path,
        "annotated_path": annotated_path,
        "mode": mode,
        "frames_sampled": len(per_frame_counts),
        "sample_every_n_frames": step,
        "num_detections": total_detections,
        "per_frame_counts": per_frame_counts,
        "best_frame_index": best_frame_idx,
        "best_confidence": round(best_conf, 3) if best_conf >= 0 else None,
        "detections": best_detections,  # detections in the best frame only
    }


if __name__ == "__main__":
    import sys, json
    img = sys.argv[1] if len(sys.argv) > 1 else "data/sample.jpg"
    print(json.dumps(detect_potholes(img), indent=2))
