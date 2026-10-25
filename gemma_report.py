"""
gemma_report.py
Turns raw detector output into a plain-English incident report using
Gemma, served through Jarvislabs Token Factory (OpenAI-compatible API).

Set your key first:
    export JARVISLABS_API_KEY="sk-..."
"""
import os
import json
import requests

# Confirmed from the Jarvislabs Token Factory playground's "View code" panel.
BASE_URL = "https://models.jarvislabs.net/v1"
API_KEY = os.environ.get("JARVISLABS_API_KEY")

# Confirmed model id from the playground dropdown (Gemma 4 26B A4B IT).
GEMMA_MODEL = "gemma-4-26b-a4b-it"


def generate_incident_report(detection_result: dict) -> str:
    """Call Gemma with the detector's JSON output, get back a short report."""
    if not API_KEY:
        return (
            "[No JARVISLABS_API_KEY set] "
            f"Detected {detection_result['num_detections']} potential issue(s). "
            "Set JARVISLABS_API_KEY to generate a full report."
        )

    if detection_result.get("source") == "video":
        context = (
            "You are given aggregated object detection output from scanned "
            "dashcam footage: frames were sampled at intervals, 'num_detections' is "
            "the total across all sampled frames (the same pothole may be counted "
            "in several frames), 'per_frame_counts' is the trend over the footage, and "
            "'detections' are from the single best frame."
        )
    else:
        context = "You are given raw object detection output from a camera mounted on a vehicle."

    prompt = f"""You are a road-maintenance assistant. {context} Write a short incident
report for a municipal maintenance team.

Detection data (JSON):
{json.dumps(detection_result, indent=2)}

Respond in this format, 4 lines max:
Severity: <Low/Medium/High>
Summary: <one sentence, plain English, what was found>
Likely cause: <one short phrase>
Recommended action: <one short phrase, with a rough timeframe>
"""

    resp = requests.post(
        f"{BASE_URL}/chat/completions",
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": GEMMA_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.4,
            # This Gemma deployment "thinks" before answering (see the
            # reasoning/reasoning_content fields) and spends tokens on
            # that first. 200 wasn't enough for it to finish reasoning
            # AND write the answer, so content came back null. Give it
            # more room.
            "max_tokens": 1024,
        },
        timeout=60,
    )
    resp.raise_for_status()
    message = resp.json()["choices"][0]["message"]
    content = message.get("content")
    if content:
        return content.strip()
    # Fallback: some responses only populate reasoning_content even once
    # finished. Use it rather than crash, so the demo never dies here.
    reasoning = message.get("reasoning_content") or message.get("reasoning")
    if reasoning:
        return reasoning.strip()
    return "[Gemma returned an empty response]"


if __name__ == "__main__":
    sample = {
        "num_detections": 1,
        "detections": [{"label": "pothole", "confidence": 0.87, "bbox_xyxy": [120, 340, 260, 410]}],
        "mode": "pothole-finetuned",
    }
    print(generate_incident_report(sample))