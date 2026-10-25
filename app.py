"""
app.py — Streamlit demo dashboard.
    streamlit run app.py
Upload a road image or video, see the detection + Gemma's plain-English report.
"""
import re

import streamlit as st
from PIL import Image

from detector import detect_potholes, detect_potholes_video
from gemma_report import generate_incident_report

st.set_page_config(page_title="Vision Incident Pipeline", page_icon="🚧", layout="centered")

CORAL = "#FF6B5B"
SEVERITY_COLOR = {"high": CORAL, "medium": "#F5B740", "low": "#5FA777"}

st.markdown(f"""
<style>
.stApp {{
    background: radial-gradient(1200px 600px at 15% -10%, #12302d 0%, #0b1a1a 45%, #080f10 100%);
    color: #F4F1EA;
}}
.block-container {{ padding-top: 2.5rem; max-width: 820px; }}
header[data-testid="stHeader"] {{ background: transparent; }}

h1 {{
    font-size: 3.2rem !important; font-weight: 900 !important;
    letter-spacing: -0.04em !important; line-height: 1.02 !important;
    color: #F4F1EA !important;
}}
h2, h3 {{ font-weight: 800 !important; letter-spacing: -0.02em !important; color: #F4F1EA !important; }}

.tagline {{ font-size: 1.1rem; color: #A9B9B5; margin: 0.4rem 0 0.9rem; }}
.stApp .tagline b {{ color: {CORAL}; }}
.stApp .pill {{
    display: inline-block; padding: 0.2rem 0.7rem; margin: 0 0.4rem 0.4rem 0;
    border: 1px solid rgba(244,241,234,0.22); border-radius: 999px;
    background: rgba(8,15,16,0.6); color: #CFE0DB; font-size: 0.78rem; letter-spacing: 0.02em;
}}

/* readable text on the dark base (Streamlit's own theme is light) */
.stApp, .stApp p, .stApp label, .stApp li, .stApp span,
div[data-testid="stWidgetLabel"] p, div[data-testid="stCaptionContainer"],
div[data-testid="stFileUploaderDropzoneInstructions"] span, div[data-testid="stFileUploaderDropzoneInstructions"] small {{
    color: #E4EAE7;
}}
div[data-testid="stWidgetLabel"] p {{ font-weight: 700; }}
header[data-testid="stHeader"] * {{ color: #A9B9B5 !important; }}
[data-testid="stFileUploaderDropzone"] {{
    background: rgba(255,255,255,0.04); border: 1px dashed rgba(244,241,234,0.3); border-radius: 14px;
}}
[data-testid="stFileUploaderDropzone"] button {{
    background: transparent; color: {CORAL}; border: 1px solid {CORAL}; border-radius: 999px; font-weight: 700;
}}
div[data-testid="stExpander"] {{ border: 1px solid rgba(244,241,234,0.14); border-radius: 14px; background: rgba(255,255,255,0.025); }}
div[data-testid="stCode"], div[data-testid="stCode"] pre {{ background: rgba(0,0,0,0.35) !important; }}
div[data-testid="stCode"] code, div[data-testid="stCode"] span {{ color: #E4EAE7 !important; }}

/* control cards */
div[data-testid="stVerticalBlockBorderWrapper"] {{
    border: 1px solid rgba(244,241,234,0.14) !important; border-radius: 16px !important;
    background: rgba(255,255,255,0.025); padding: 0.4rem 0.6rem;
}}

/* coral accents: slider */
div[data-baseweb="slider"] div[role="slider"] {{
    background-color: {CORAL} !important; border-color: {CORAL} !important; box-shadow: none !important;
}}
div[data-baseweb="slider"] div[role="slider"] div {{ color: {CORAL} !important; font-weight: 700; }}
div[data-testid="stSlider"] [data-baseweb="slider"] > div > div:first-child {{
    background: linear-gradient(to right, {CORAL} var(--p, 0%), rgba(244,241,234,0.18) var(--p, 0%)) !important;
}}
/* coral accents: radio */
label[data-testid="stRadioOption"] > div:first-of-type > div:first-child {{
    background: transparent !important; border: 2px solid rgba(244,241,234,0.4) !important;
}}
label[data-testid="stRadioOption"]:has(input:checked) > div:first-of-type > div:first-child {{
    background: {CORAL} !important; border-color: {CORAL} !important;
}}
label[data-testid="stRadioOption"]:has(input:checked) > div:first-of-type > div:first-child > div {{
    background: #140c0b !important;
}}
[data-testid="stFileUploaderFile"], [data-testid="stFileUploaderFileData"] {{ background: transparent !important; }}
[data-testid="stFileUploaderFile"] * {{ color: #E4EAE7 !important; }}
/* key numbers */
div[data-testid="stMetricValue"] {{ color: {CORAL}; font-weight: 900; letter-spacing: -0.03em; }}
div[data-testid="stMetricLabel"] {{ color: #A9B9B5; }}
.stButton > button, .stDownloadButton > button {{
    background: {CORAL}; color: #1a0f0d; border: 0; border-radius: 999px; font-weight: 800;
}}

/* severity pill */
.stApp .sev {{
    display: inline-block; padding: 0.35rem 1.1rem; border-radius: 999px; margin: 0.2rem 0 0.6rem;
    font-size: 1.05rem; font-weight: 900; letter-spacing: 0.04em; text-transform: uppercase; color: #140c0b;
}}
</style>
""", unsafe_allow_html=True)


def show_report(report: str):
    """Render the Gemma report with a colour-coded severity pill."""
    m = re.search(r"Severity:\s*\**\s*(High|Medium|Low)", report, re.IGNORECASE)
    if m:
        level = m.group(1).capitalize()
        st.markdown(
            f'<span class="sev" style="background:{SEVERITY_COLOR[level.lower()]}">{level} severity</span>',
            unsafe_allow_html=True,
        )
    st.code(report, language=None)


st.title("🚧 Vision Incident Pipeline")
st.markdown(
    '<div class="tagline">Turn road footage into maintenance-ready incident reports <b>in seconds.</b></div>'
    + "".join(f'<span class="pill">{t}</span>' for t in
              ["🔍 YOLOv8", "🧠 Gemma", "☁️ Jarvislabs", "⏱️ Temporal"]),
    unsafe_allow_html=True,
)

with st.container(border=True):
    conf_threshold = st.slider(
        "Confidence threshold", 0.05, 0.95, 0.5, 0.05,
        help="Only detections at or above this confidence are drawn and counted.",
    )
with st.container(border=True):
    input_type = st.radio("Input type", ["Image", "Video"], horizontal=True)

detection = None
if input_type == "Image":
    uploaded = st.file_uploader("Upload a road image", type=["jpg", "jpeg", "png"])
    if uploaded:
        img_path = f"data/{uploaded.name}"
        with open(img_path, "wb") as f:
            f.write(uploaded.getbuffer())

        st.image(Image.open(img_path), caption="Input", use_container_width=True)

        with st.spinner("Detecting..."):
            detection = detect_potholes(img_path, conf=conf_threshold)

        st.image(detection["annotated_path"], caption="Detected", use_container_width=True)
        st.subheader(f"{detection['num_detections']} detection(s)")
else:
    uploaded = st.file_uploader("Upload road footage", type=["mp4", "mov", "avi", "mkv"])
    every_n = st.slider("Sample every N frames", 5, 60, 15)
    if uploaded:
        vid_path = f"data/{uploaded.name}"
        with open(vid_path, "wb") as f:
            f.write(uploaded.getbuffer())

        st.video(vid_path)

        with st.spinner("Scanning footage..."):
            detection = detect_potholes_video(vid_path, sample_every_n_frames=every_n, conf=conf_threshold)

        if detection["annotated_path"]:
            st.image(detection["annotated_path"],
                     caption=f"Best frame (#{detection['best_frame_index']}, "
                             f"confidence {detection['best_confidence']})",
                     use_container_width=True)
        else:
            st.warning("No potholes detected in the sampled frames.")
        c1, c2 = st.columns(2)
        c1.metric("Frames sampled", detection["frames_sampled"])
        c2.metric("Total detections", detection["num_detections"])
        st.caption("Detections per sampled frame")
        st.line_chart(detection["per_frame_counts"])

if detection:
    if detection["mode"] != "pothole-finetuned":
        st.info("Running in demo-fallback mode (stock COCO weights) — swap in a pothole-finetuned model for production.")

    with st.spinner("Asking Gemma for an incident report..."):
        report = generate_incident_report(detection)

    st.subheader("📋 Incident Report (Gemma)")
    show_report(report)

    with st.expander("⚡ Efficiency note (Kepler)"):
        st.write(
            "Kepler (RubixKube) measures actual compute energy at the node/pod "
            "level via eBPF — it's a cluster-level exporter, not an in-app API. "
            "For the demo, show the Kepler dashboard/Grafana panel side-by-side "
            "with this app to make the 'efficient inference' point live."
        )
else:
    st.write(f"👆 Upload {'an image' if input_type == 'Image' else 'a video'} to run the pipeline.")
