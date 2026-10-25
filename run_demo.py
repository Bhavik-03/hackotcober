"""
run_demo.py
Fallback runner that does NOT need a Temporal server running — use this
if the Temporal dev server isn't up during judging. Same two steps,
called directly in sequence.

    python run_demo.py data/sample.jpg
"""
import sys
import json

from detector import detect_potholes
from gemma_report import generate_incident_report


def main(image_path: str):
    print(f"[1/2] Running detector on {image_path} ...")
    detection = detect_potholes(image_path)
    print(json.dumps(detection, indent=2))

    print("\n[2/2] Generating incident report via Gemma (Jarvislabs) ...")
    report = generate_incident_report(detection)
    print("\n--- INCIDENT REPORT ---")
    print(report)

    return {"detection": detection, "report": report}


if __name__ == "__main__":
    img = sys.argv[1] if len(sys.argv) > 1 else "data/sample.jpg"
    main(img)
