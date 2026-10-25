# Vision Incident Pipeline 🚧

Upload a road photo → YOLO flags the problem → **Gemma** (served via **Jarvislabs**
Token Factory) explains it in plain English → **Temporal** keeps the pipeline
reliable (auto-retries if a step fails) → **Kepler** shows the efficiency story.

## 1. Setup (5 min)

```bash
pip install -r requirements.txt
export JARVISLABS_API_KEY="your-jarvislabs-token-factory-key"
```

Drop a road photo at `data/sample.jpg` (any photo with a pothole/road damage works).

## 2. Fastest demo path — no Temporal server needed

```bash
python run_demo.py data/sample.jpg
```

or the dashboard:

```bash
streamlit run app.py
```

## 3. Full pipeline with Temporal (the "durable workflow" story)

Terminal 1 — start a local Temporal dev server (one-time install: `brew install temporal` or see temporal.io):
```bash
temporal server start-dev
```

Terminal 2 — start the worker:
```bash
python workflow.py
```

Terminal 3 — kick off a run:
```bash
python starter.py data/sample.jpg
```

Open the Temporal Web UI (printed on server start, usually http://localhost:8233)
to **show judges the workflow graph live** — this is your strongest visual for
"why Temporal, not just a script": if the Gemma call fails or rate-limits, you'll
see it auto-retry in the UI instead of the whole run dying.

## How each sponsor tool is used

| Tool | Role in this project |
|---|---|
| **YOLO (CV model)** | Detects the problem in the image, draws the bounding box |
| **Gemma** (open-weight) | Turns the raw detection JSON into a human-readable incident report |
| **Jarvislabs Token Factory** | Serves Gemma — `gemma_report.py` calls it over an OpenAI-compatible API |
| **Temporal** | Orchestrates `detect → report` as a durable, auto-retrying workflow (`workflow.py`) |
| **Kepler** | Cluster-level power/energy exporter (eBPF) — not an in-app API. For the demo, run Kepler per the Hack Day instructions and show its metrics panel side-by-side to back the "efficient inference" angle (see the bonus-points sheet: install from the RubixKube releases page and sign in via the Hack Day pass link). |

## Files

- `detector.py` — YOLO detection (tries a pothole-finetuned HF model, falls back to stock COCO weights for demo safety)
- `gemma_report.py` — Gemma call via Jarvislabs Token Factory
- `workflow.py` / `starter.py` — Temporal workflow, activities, worker, and starter
- `run_demo.py` — no-Temporal-server fallback (detect → report, straight line)
- `app.py` — Streamlit dashboard

## If something breaks mid-demo

- No Jarvislabs key set → `gemma_report.py` prints a placeholder instead of crashing, so detection still demos fine.
- No Temporal server → use `run_demo.py` or `app.py`, same core logic, no orchestration layer shown.
- HF pothole model unreachable → detector silently falls back to stock YOLOv8n and labels its output `demo-fallback-coco` so you're never caught claiming something false on stage.
