"""
PCB Defect Detector — Streamlit Web Application
Two-stage pipeline: YOLO detection → CNN classification

Features
--------
• Dark-themed premium UI with custom CSS (glassmorphism + gradient).
• Four tabs: **Detector**, **Gallery**, **Model Performance**, **About**.
• Per-image Before/After slider (self-contained HTML/JS, no extra deps).
• Per-detection GradCAM heatmap panel using a cached, per-CNN GradCAM instance.
• Batch upload with grid / side-by-side / compact gallery layouts.
• Pre-loaded demo samples (drag-and-drop or one-click load).
"""

from __future__ import annotations

import json
import tempfile
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from before_after_slider import before_after_slider
from gradcam_ui import render_gradcam_for_detection

# ──────────────────────────────────────────────────────────────────────────────
# Page configuration (must be the first Streamlit call)
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PCB Defect Detector",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

PROJECT_ROOT = Path(__file__).resolve().parent

CNN_OPTIONS = {
    "ResNet18": "runs/stage2/resnet18/best.pt",
    "ResNet50": "runs/stage2/resnet50/best.pt",
    "EfficientNet-B2": "runs/stage2/efficientnet_b2/best.pt",
}

DEFAULT_YOLO_PATH = "runs/detect/v8m_768_adamw_aug/weights/best.pt"

DEFECT_COLORS = {
    "missing_hole": "#f87171",
    "mouse_bite": "#fb923c",
    "open_circuit": "#facc15",
    "short": "#34d399",
    "spur": "#60a5fa",
    "spurious_copper": "#a78bfa",
}

DEMO_INPUT_DIR = PROJECT_ROOT / "demo_input"
GRADCAM_DIR = PROJECT_ROOT / "runs" / "gradcam"
PERF_DIR = PROJECT_ROOT / "runs" / "system_eval"
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def _defect_css_class(label: str) -> str:
    key = label.lower().replace(" ", "_")
    return key if key in DEFECT_COLORS else "other"


# ──────────────────────────────────────────────────────────────────────────────
# Custom CSS — premium dark-compatible styling
# ──────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

:root {
    --violet: #a78bfa;
    --blue:   #60a5fa;
    --green:  #34d399;
    --bg-0:   #0f0c29;
    --bg-1:   #1a1733;
    --line:   rgba(255,255,255,0.07);
    --text-1: rgba(255,255,255,0.85);
    --text-2: rgba(255,255,255,0.55);
    --text-3: rgba(255,255,255,0.35);
}

/* ── Header gradient banner ── */
.header-banner {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    border-radius: 16px;
    padding: 2.2rem 2rem 1.8rem 2rem;
    margin-bottom: 1.4rem;
    text-align: center;
    position: relative;
    overflow: hidden;
    border: 1px solid var(--line);
    box-shadow: 0 8px 32px rgba(0,0,0,0.35);
}
.header-banner::before {
    content: '';
    position: absolute;
    top: -50%; left: -50%;
    width: 200%; height: 200%;
    background:
      radial-gradient(circle at 30% 40%, rgba(100,100,255,0.08) 0%, transparent 60%),
      radial-gradient(circle at 70% 60%, rgba(0,200,200,0.06) 0%, transparent 60%);
    animation: shimmer 8s ease-in-out infinite alternate;
}
@keyframes shimmer {
    0% { transform: translateX(-5%) translateY(-5%); }
    100% { transform: translateX(5%) translateY(5%); }
}
.header-banner h1 {
    margin: 0;
    font-size: 2.2rem;
    font-weight: 800;
    background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    position: relative;
    letter-spacing: -0.03em;
}
.header-banner p {
    margin: .6rem 0 0 0;
    color: var(--text-2);
    font-size: .95rem;
    position: relative;
}
.header-banner.compact { padding: 1.2rem 1.6rem; }
.header-banner.compact h1 { font-size: 1.5rem; }

/* ── Card ── */
.card {
    background: rgba(255,255,255,0.03);
    border: 1px solid var(--line);
    border-radius: 14px;
    padding: 1.4rem;
    margin-bottom: 1.1rem;
    backdrop-filter: blur(12px);
    box-shadow: 0 4px 20px rgba(0,0,0,0.20);
    transition: box-shadow 0.3s ease, border-color 0.3s ease, transform 0.2s ease;
}
.card:hover {
    border-color: rgba(167,139,250,0.30);
    box-shadow: 0 6px 28px rgba(0,0,0,0.30);
}

/* ── Metric cards ── */
.metric-row {
    display: flex;
    gap: 1rem;
    margin-bottom: 1.4rem;
    flex-wrap: wrap;
}
.metric-card {
    flex: 1 1 180px;
    background: linear-gradient(145deg, rgba(255,255,255,0.04), rgba(255,255,255,0.01));
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 1.2rem 1rem;
    text-align: center;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.metric-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 24px rgba(0,0,0,0.25);
}
.metric-value {
    font-size: 1.9rem;
    font-weight: 700;
    background: linear-gradient(90deg, #a78bfa, #60a5fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.2;
}
.metric-label {
    font-size: .78rem;
    color: var(--text-2);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: .3rem;
}

/* ── Defect-type badges ── */
.defect-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 6px;
    font-size: .78rem;
    font-weight: 600;
    letter-spacing: 0.02em;
    margin: 2px;
    border: 1px solid rgba(255,255,255,0.05);
}
.defect-missing_hole   { background: rgba(239,68,68,0.18);  color: #f87171; }
.defect-mouse_bite     { background: rgba(251,146,60,0.18); color: #fb923c; }
.defect-open_circuit   { background: rgba(250,204,21,0.18); color: #facc15; }
.defect-short          { background: rgba(52,211,153,0.18); color: #34d399; }
.defect-spur           { background: rgba(96,165,250,0.18); color: #60a5fa; }
.defect-spurious_copper{ background: rgba(167,139,250,0.18);color: #a78bfa; }
.defect-other          { background: rgba(148,163,184,0.18);color: #94a3b8; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: rgba(15,12,41,0.60);
    backdrop-filter: blur(18px);
}
section[data-testid="stSidebar"] .stMarkdown h2 {
    font-size: 1.1rem;
    font-weight: 700;
    color: var(--violet);
    margin-bottom: .2rem;
    letter-spacing: -0.01em;
}

/* ── Streamlit element tweaks ── */
.stDataFrame { border-radius: 10px; overflow: hidden; }
div[data-testid="stFileUploader"] {
    border: 2px dashed rgba(167,139,250,0.25) !important;
    border-radius: 12px !important;
    transition: border-color 0.3s ease;
}
div[data-testid="stFileUploader"]:hover {
    border-color: rgba(167,139,250,0.50) !important;
}
div[data-testid="stFileUploaderDropzone"] {
    background: rgba(167,139,250,0.04) !important;
}

/* ── Buttons ── */
.stDownloadButton > button, .stButton > button {
    background: linear-gradient(135deg, #302b63, #24243e) !important;
    border: 1px solid rgba(167,139,250,0.30) !important;
    border-radius: 8px !important;
    color: var(--text-1) !important;
    font-weight: 600 !important;
    transition: all 0.25s ease !important;
}
.stDownloadButton > button:hover, .stButton > button:hover {
    border-color: rgba(167,139,250,0.60) !important;
    box-shadow: 0 4px 16px rgba(167,139,250,0.20) !important;
    transform: translateY(-1px) !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 6px;
    background: rgba(15,12,41,0.40);
    padding: 6px;
    border-radius: 12px;
    border: 1px solid var(--line);
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border-radius: 8px !important;
    color: var(--text-2) !important;
    padding: 8px 18px !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(167,139,250,0.25), rgba(96,165,250,0.18)) !important;
    color: var(--text-1) !important;
    box-shadow: 0 2px 8px rgba(167,139,250,0.25) !important;
}

/* ── Status indicator ── */
.status-dot {
    display: inline-block;
    width: 8px; height: 8px;
    border-radius: 50%;
    margin-right: 6px;
    animation: pulse 2s ease-in-out infinite;
}
.status-dot.ready { background: #34d399; }
.status-dot.error { background: #f87171; }
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: .45; }
}

/* ── Section separator ── */
.section-sep {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(167,139,250,0.25), transparent);
    margin: 1.6rem 0;
}

/* ── Demo sample button ── */
.demo-pill {
    display: inline-block;
    padding: 6px 12px;
    margin: 3px;
    background: rgba(167,139,250,0.10);
    border: 1px solid rgba(167,139,250,0.25);
    border-radius: 20px;
    color: var(--text-1);
    font-size: .82rem;
    cursor: pointer;
    transition: all 0.2s ease;
}
.demo-pill:hover {
    background: rgba(167,139,250,0.25);
    transform: translateY(-1px);
}

/* ── Gallery grid ── */
.gallery-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid var(--line);
    border-radius: 12px;
    padding: 0.8rem;
    margin-bottom: 0.9rem;
    transition: all 0.25s ease;
}
.gallery-card:hover {
    border-color: rgba(167,139,250,0.35);
    transform: translateY(-2px);
    box-shadow: 0 6px 22px rgba(0,0,0,0.30);
}
.gallery-card img {
    border-radius: 8px;
    width: 100%;
    display: block;
}
.gallery-title {
    font-weight: 700;
    font-size: .95rem;
    color: var(--text-1);
    margin-bottom: .2rem;
}
.gallery-sub {
    color: var(--text-2);
    font-size: .78rem;
    margin-bottom: .5rem;
}

/* ── Empty state ── */
.empty-state {
    text-align: center;
    padding: 3rem 2rem;
    background: rgba(255,255,255,0.02);
    border: 1px dashed rgba(167,139,250,0.20);
    border-radius: 14px;
}
.empty-state .icon {
    font-size: 2.6rem;
    margin-bottom: .5rem;
}
.empty-state .title {
    font-size: 1.05rem;
    color: var(--text-1);
    margin-bottom: .3rem;
}
.empty-state .subtitle {
    font-size: .82rem;
    color: var(--text-3);
}

@media (prefers-reduced-motion: reduce) {
    .header-banner::before,
    .status-dot { animation: none; }
    .card, .gallery-card, .metric-card { transition: none; }
}
</style>
""",
    unsafe_allow_html=True,
)


# ──────────────────────────────────────────────────────────────────────────────
# Model loading with caching
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_pipeline(yolo_path: str, cnn_path: str, imgsz: int, conf: float, iou: float):
    from stage12_yolo_cnn_system import Stage12Pipeline

    yolo_abs = PROJECT_ROOT / yolo_path
    cnn_abs = PROJECT_ROOT / cnn_path

    if not yolo_abs.exists():
        raise FileNotFoundError(f"YOLO weights not found:\n`{yolo_abs}`")
    if not cnn_abs.exists():
        raise FileNotFoundError(f"CNN checkpoint not found:\n`{cnn_abs}`")

    return Stage12Pipeline(
        yolo_path=str(yolo_abs),
        cnn_checkpoint=str(cnn_abs),
        yolo_imgsz=imgsz,
        yolo_conf=conf,
        yolo_iou=iou,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔬 PCB Defect Detector")
    st.caption(
        "Two-stage pipeline: **YOLO** object detection → **CNN** defect "
        "classification. Upload PCB images to find & classify manufacturing defects."
    )
    st.markdown('<div class="section-sep"></div>', unsafe_allow_html=True)

    st.markdown("## 🧠 CNN Model")
    cnn_choice = st.selectbox(
        "Architecture",
        options=list(CNN_OPTIONS.keys()),
        index=0,
        help="Stage-2 classifier backbone. ResNet18 is fastest; EfficientNet-B2 is most accurate.",
        key="cnn_choice",
    )
    cnn_path = st.text_input(
        "Checkpoint path",
        value=CNN_OPTIONS[cnn_choice],
        help="Relative to project root.",
        key="cnn_path",
    )

    st.markdown('<div class="section-sep"></div>', unsafe_allow_html=True)

    st.markdown("## ⚙️ YOLO Settings")
    yolo_path = st.text_input(
        "YOLO weights",
        value=DEFAULT_YOLO_PATH,
        help="Relative to project root.",
        key="yolo_path",
    )
    yolo_conf = st.slider(
        "Confidence threshold", 0.05, 0.95, 0.25, 0.05,
        help="Minimum detection confidence.", key="yolo_conf",
    )
    yolo_iou = st.slider(
        "IoU threshold (NMS)", 0.10, 0.95, 0.70, 0.05,
        help="Non-maximum-suppression IoU overlap threshold.", key="yolo_iou",
    )
    yolo_imgsz = st.selectbox(
        "Image size", [640, 768, 1024], index=1,
        help="Input resolution for YOLO inference.", key="yolo_imgsz",
    )

    st.markdown('<div class="section-sep"></div>', unsafe_allow_html=True)

    st.markdown("## 🎨 Display")
    show_gradcam = st.checkbox(
        "Enable GradCAM heatmap per detection",
        value=True,
        help="Adds a GradCAM overlay on each YOLO crop (slightly slower).",
    )
    show_slider = st.checkbox(
        "Enable Before/After slider",
        value=True,
        help="Adds a draggable slider comparing original vs annotated image.",
    )

    st.markdown('<div class="section-sep"></div>', unsafe_allow_html=True)
    st.caption("Built with Streamlit · YOLO + CNN pipeline")


# ──────────────────────────────────────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
<div class="header-banner">
    <h1>PCB Defect Detector</h1>
    <p>Upload PCB images below and let the two-stage AI pipeline find manufacturing defects in seconds.</p>
</div>
""",
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────────────────────
# Model load + status
# ──────────────────────────────────────────────────────────────────────────────
pipeline = None
model_ready = False
try:
    with st.spinner("Loading models… this may take a moment on first run."):
        pipeline = load_pipeline(yolo_path, cnn_path, yolo_imgsz, yolo_conf, yolo_iou)
    model_ready = True
except FileNotFoundError as exc:
    st.error(f"**Model file missing**\n\n{exc}")
    st.info("Make sure you have trained models in the expected paths. Adjust paths in the sidebar.")
except Exception as exc:
    st.error(f"**Failed to load models**\n\n{exc}")

if model_ready:
    st.markdown(
        '<span class="status-dot ready"></span> Pipeline loaded — ready for inference',
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        '<span class="status-dot error"></span> Pipeline not loaded — check sidebar settings',
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Helpers used across tabs
# ──────────────────────────────────────────────────────────────────────────────
def _save_upload_to_temp(uploaded_file) -> str:
    suffix = Path(uploaded_file.name).suffix
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.getbuffer())
    tmp.close()
    return tmp.name


def _run_pipeline(image_path: str) -> dict:
    """Run inference + annotation for one image. Returns a result dict."""
    from stage12_yolo_cnn_system import annotate_predictions

    result = pipeline.predict_image(image_path)
    annotated_bgr = annotate_predictions(image_path, result)
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    # Read original for slider / gallery rendering
    original_rgb = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

    # Pre-compute crops for GradCAM (avoids re-reading the image)
    crops: list[np.ndarray] = []
    for pred in result.get("predictions", []):
        x1, y1, x2, y2 = pred["bbox"]
        crop = cv2.imread(image_path)[y1:y2, x1:x2]
        if crop is not None and crop.size > 0:
            crops.append(crop)
    result["_annotated_rgb"] = annotated_rgb
    result["_original_rgb"] = original_rgb
    result["_crops"] = crops
    return result


def _list_demo_samples() -> list[Path]:
    if not DEMO_INPUT_DIR.exists():
        return []
    return sorted(p for p in DEMO_INPUT_DIR.iterdir() if p.suffix.lower() in SUPPORTED_EXTS)


def _aggregate_predictions(results: list[dict]) -> tuple[list[str], list[float], Counter]:
    labels: list[str] = []
    confs: list[float] = []
    counter: Counter = Counter()
    for r in results:
        for p in r.get("predictions", []):
            labels.append(p["stage2_label"])
            confs.append(p["combined_confidence"])
            counter[p["stage2_label"]] += 1
    return labels, confs, counter


def _empty_state(icon: str, title: str, subtitle: str) -> str:
    return f"""
<div class="empty-state">
    <div class="icon">{icon}</div>
    <div class="title">{title}</div>
    <div class="subtitle">{subtitle}</div>
</div>
"""


# ──────────────────────────────────────────────────────────────────────────────
# Tabs
# ──────────────────────────────────────────────────────────────────────────────
tab_detector, tab_gallery, tab_perf, tab_about = st.tabs(
    ["🔍 Detector", "🖼 Gallery", "📊 Model Performance", "ℹ️ About"]
)


# ──────────────────────────────────────────────────────────────────────────────
# TAB 1 — Detector (single-image deep view)
# ──────────────────────────────────────────────────────────────────────────────
with tab_detector:
    st.markdown(
        '<p style="color:rgba(255,255,255,0.55); margin-top:-0.5rem;">'
        'Detailed inspection with Before/After slider and GradCAM per detection.'
        '</p>',
        unsafe_allow_html=True,
    )

    # Demo samples row
    demo_samples = _list_demo_samples()
    if demo_samples and model_ready:
        st.markdown("**Try a sample:**")
        cols = st.columns(min(5, len(demo_samples)))
        for i, sample in enumerate(demo_samples[:5]):
            with cols[i]:
                if st.button(
                    f"📷 {sample.stem[:24]}",
                    key=f"demo_load_{i}",
                    width="stretch",
                    help=f"Load {sample.name}",
                ):
                    st.session_state["detector_upload"] = str(sample)
        st.markdown('<div class="section-sep"></div>', unsafe_allow_html=True)

    detector_file = st.file_uploader(
        "Upload a PCB image for detailed inspection",
        type=["jpg", "jpeg", "png", "bmp"],
        accept_multiple_files=False,
        key="detector_uploader",
        help="Supports JPG, JPEG, PNG, and BMP formats.",
    )

    # Allow loading from a demo sample via session state
    demo_path = st.session_state.get("detector_upload")
    if demo_path and Path(demo_path).exists():
        image_path = demo_path
        uploaded_name = Path(demo_path).name
    elif detector_file is not None:
        image_path = _save_upload_to_temp(detector_file)
        uploaded_name = detector_file.name
    else:
        image_path = None
        uploaded_name = None

    if image_path and model_ready:
        try:
            with st.spinner("Running inference…"):
                result = _run_pipeline(image_path)
            st.markdown(f'<div class="card">', unsafe_allow_html=True)
            st.subheader(f"📄 {uploaded_name}")

            original_rgb = result["_original_rgb"]
            annotated_rgb = result["_annotated_rgb"]
            preds = result.get("predictions", [])

            if show_slider and preds:
                st.markdown("**Before / After**")
                before_after_slider(
                    original=original_rgb,
                    modified=annotated_rgb,
                    right_label="Detected",
                    height=420,
                    key=f"slider_{uploaded_name}",
                )
            else:
                col_o, col_a = st.columns(2)
                with col_o:
                    st.markdown("**Original**")
                    st.image(original_rgb, width="stretch")
                with col_a:
                    st.markdown("**Detections**")
                    st.image(annotated_rgb, width="stretch")

            if preds:
                # Results table
                rows = [
                    {
                        "Defect": p["stage2_label"],
                        "Stage 1": f"{p['stage1_confidence']:.3f}",
                        "Stage 2": f"{p['stage2_confidence']:.3f}",
                        "Combined": f"{p['combined_confidence']:.3f}",
                        "BBox (x1 y1 x2 y2)": str(p["bbox"]),
                    }
                    for p in preds
                ]
                st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

                # Defect badges
                badges = " ".join(
                    f'<span class="defect-badge defect-{_defect_css_class(r["Defect"])}">'
                    f'{r["Defect"]}</span>'
                    for r in rows
                )
                st.markdown(badges, unsafe_allow_html=True)

                # GradCAM per detection
                if show_gradcam:
                    st.markdown('<div class="section-sep"></div>', unsafe_allow_html=True)
                    st.markdown("**GradCAM — what the CNN looks at**")
                    crops = result["_crops"]
                    for idx, (pred, crop) in enumerate(zip(preds, crops)):
                        with st.expander(
                            f"🔬 #{idx + 1}  {pred['stage2_label']}  ·  "
                            f"combined {pred['combined_confidence']:.1%}",
                            expanded=(idx == 0),
                        ):
                            gradcam_out = render_gradcam_for_detection(
                                crop_bgr=crop,
                                image_stem=Path(image_path).stem,
                                bundle=pipeline.cnn_bundle,
                            )
                            if gradcam_out.get("mode") == "unsupported":
                                st.info("GradCAM not supported for this CNN architecture.")
                            else:
                                cols = st.columns(3)
                                with cols[0]:
                                    st.markdown(
                                        "<div class='gallery-sub'>Crop</div>",
                                        unsafe_allow_html=True,
                                    )
                                    st.image(
                                        cv2.cvtColor(crop, cv2.COLOR_BGR2RGB),
                                        width="stretch",
                                    )
                                with cols[1]:
                                    st.markdown(
                                        "<div class='gallery-sub'>Heatmap</div>",
                                        unsafe_allow_html=True,
                                    )
                                    if gradcam_out.get("cam") is not None:
                                        st.image(
                                            gradcam_out["cam"],
                                            width="stretch",
                                            clamp=True,
                                        )
                                    else:
                                        st.caption("Heatmap not extracted from cache.")
                                with cols[2]:
                                    st.markdown(
                                        "<div class='gallery-sub'>Overlay</div>",
                                        unsafe_allow_html=True,
                                    )
                                    if gradcam_out.get("overlay") is not None:
                                        st.image(
                                            gradcam_out["overlay"],
                                            width="stretch",
                                        )
            else:
                st.success("✅ No defects detected — board looks clean!")

            # JSON download
            public_result = {
                k: v for k, v in result.items() if not k.startswith("_")
            }
            st.download_button(
                "⬇ Download JSON",
                data=json.dumps(public_result, indent=2),
                file_name=f"{Path(uploaded_name).stem}_results.json",
                mime="application/json",
                key=f"dl_det_{uploaded_name}",
            )

            st.markdown("</div>", unsafe_allow_html=True)
        except Exception as exc:
            st.error(f"Error processing **{uploaded_name}**: {exc}")
    else:
        st.markdown(
            _empty_state(
                "📷",
                "Drag & drop a PCB image to start inspection",
                "Supports JPG · JPEG · PNG · BMP",
            ),
            unsafe_allow_html=True,
        )


# ──────────────────────────────────────────────────────────────────────────────
# TAB 2 — Gallery (batch upload)
# ──────────────────────────────────────────────────────────────────────────────
with tab_gallery:
    st.markdown(
        '<p style="color:rgba(255,255,255,0.55); margin-top:-0.5rem;">'
        'Upload multiple PCB images and inspect them side-by-side or in a grid.'
        '</p>',
        unsafe_allow_html=True,
    )

    gallery_files = st.file_uploader(
        "Drop PCB images here",
        type=["jpg", "jpeg", "png", "bmp"],
        accept_multiple_files=True,
        key="gallery_uploader",
    )

    # Layout selector
    layout = st.radio(
        "Gallery layout",
        options=["Grid (3 cols)", "Side-by-side", "Compact list"],
        horizontal=True,
        key="gallery_layout",
    )

    if gallery_files and model_ready:
        all_results: list[dict] = []
        all_labels: list[str] = []
        all_confs: list[float] = []

        progress = st.progress(0.0, text="Starting…")
        for idx, uploaded in enumerate(gallery_files):
            progress.progress(
                (idx) / len(gallery_files),
                text=f"Processing {uploaded.name} ({idx + 1}/{len(gallery_files)})",
            )
            tmp_path = _save_upload_to_temp(uploaded)
            try:
                result = _run_pipeline(tmp_path)
                result["_uploaded_name"] = uploaded.name
                all_results.append(result)
                for p in result.get("predictions", []):
                    all_labels.append(p["stage2_label"])
                    all_confs.append(p["combined_confidence"])
            except Exception as exc:
                st.error(f"Error processing **{uploaded.name}**: {exc}")
        progress.progress(1.0, text="Done")

        if not all_results:
            st.stop()

        # Render gallery
        if layout.startswith("Grid"):
            n_cols = 3
            for i in range(0, len(all_results), n_cols):
                row_results = all_results[i : i + n_cols]
                cols = st.columns(n_cols, gap="small")
                for col, res in zip(cols, row_results):
                    with col:
                        st.markdown('<div class="gallery-card">', unsafe_allow_html=True)
                        st.markdown(
                            f'<div class="gallery-title">{res["_uploaded_name"]}</div>',
                            unsafe_allow_html=True,
                        )
                        n_pred = len(res.get("predictions", []))
                        st.markdown(
                            f'<div class="gallery-sub">{n_pred} detection(s)</div>',
                            unsafe_allow_html=True,
                        )
                        st.image(res["_annotated_rgb"], width="stretch")
                        if n_pred:
                            badges = " ".join(
                                f'<span class="defect-badge defect-{_defect_css_class(p["stage2_label"])}">'
                                f'{p["stage2_label"]}</span>'
                                for p in res["predictions"]
                            )
                            st.markdown(badges, unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)
        elif layout.startswith("Side-by-side"):
            for res in all_results:
                st.markdown('<div class="gallery-card">', unsafe_allow_html=True)
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Original**")
                    st.image(res["_original_rgb"], width="stretch")
                with c2:
                    st.markdown("**Detected**")
                    st.image(res["_annotated_rgb"], width="stretch")
                st.markdown(
                    f"**{res['_uploaded_name']}** — {len(res['predictions'])} detection(s)",
                )
                if res["predictions"]:
                    badges = " ".join(
                        f'<span class="defect-badge defect-{_defect_css_class(p["stage2_label"])}">'
                        f'{p["stage2_label"]}</span>'
                        for p in res["predictions"]
                    )
                    st.markdown(badges, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
        else:  # Compact list
            for res in all_results:
                with st.expander(
                    f"📄 {res['_uploaded_name']}  ·  {len(res['predictions'])} detection(s)",
                    expanded=False,
                ):
                    c1, c2 = st.columns([1, 2])
                    with c1:
                        st.image(res["_annotated_rgb"], width="stretch")
                    with c2:
                        if res["predictions"]:
                            df = pd.DataFrame(
                                [
                                    {
                                        "Defect": p["stage2_label"],
                                        "S1": f"{p['stage1_confidence']:.3f}",
                                        "S2": f"{p['stage2_confidence']:.3f}",
                                        "Combined": f"{p['combined_confidence']:.3f}",
                                    }
                                    for p in res["predictions"]
                                ]
                            )
                            st.dataframe(df, width="stretch", hide_index=True)
                        else:
                            st.success("No defects detected.")

        # ── Aggregate summary ──
        st.markdown('<div class="section-sep"></div>', unsafe_allow_html=True)
        st.markdown(
            """
<div class="header-banner compact">
    <h1>Summary</h1>
    <p>Aggregated metrics across all uploaded images</p>
</div>
""",
            unsafe_allow_html=True,
        )
        n_images = len(all_results)
        total_defects = len(all_labels)
        avg_conf = float(np.mean(all_confs)) if all_confs else 0.0
        counter = Counter(all_labels)
        n_types = len(counter)

        st.markdown(
            f"""
<div class="metric-row">
    <div class="metric-card">
        <div class="metric-value">{n_images}</div>
        <div class="metric-label">Images</div>
    </div>
    <div class="metric-card">
        <div class="metric-value">{total_defects}</div>
        <div class="metric-label">Total Defects</div>
    </div>
    <div class="metric-card">
        <div class="metric-value">{avg_conf:.1%}</div>
        <div class="metric-label">Avg Combined Conf</div>
    </div>
    <div class="metric-card">
        <div class="metric-value">{n_types}</div>
        <div class="metric-label">Defect Types</div>
    </div>
</div>
""",
            unsafe_allow_html=True,
        )

        if counter:
            chart_df = (
                pd.DataFrame.from_dict(counter, orient="index", columns=["Count"])
                .sort_values("Count", ascending=False)
            )
            chart_df.index.name = "Defect Type"
            st.bar_chart(chart_df, width="stretch")

            # Cross-image pivot
            rows = []
            for res in all_results:
                name = res["_uploaded_name"]
                for p in res["predictions"]:
                    rows.append(
                        {
                            "Image": name,
                            "Defect": p["stage2_label"],
                            "Confidence": p["combined_confidence"],
                        }
                    )
            if rows:
                pivot_df = pd.DataFrame(rows)
                st.markdown("**Defects per image**")
                pivot = (
                    pivot_df.groupby(["Image", "Defect"])
                    .size()
                    .unstack(fill_value=0)
                )
                st.dataframe(pivot, width="stretch")

        # JSON download for all results
        clean = [{k: v for k, v in r.items() if not k.startswith("_")} for r in all_results]
        st.download_button(
            "⬇ Download All Results (JSON)",
            data=json.dumps(clean, indent=2),
            file_name="pcb_all_results.json",
            mime="application/json",
            key="dl_all",
        )

    elif gallery_files and not model_ready:
        st.warning("Please fix the model configuration in the sidebar before uploading images.")
    else:
        st.markdown(
            _empty_state(
                "🖼",
                "Upload multiple PCB images to build a gallery",
                "Switch between grid, side-by-side, or compact list layout above.",
            ),
            unsafe_allow_html=True,
        )


# ──────────────────────────────────────────────────────────────────────────────
# TAB 3 — Model Performance
# ──────────────────────────────────────────────────────────────────────────────
with tab_perf:
    st.markdown(
        '<p style="color:rgba(255,255,255,0.55); margin-top:-0.5rem;">'
        'Error-analysis artifacts generated by <code>error_analysis.py</code> '
        'and GradCAM figures from <code>gradcam_visualize.py</code>.'
        '</p>',
        unsafe_allow_html=True,
    )

    sub_tabs = st.tabs(
        ["📈 Error Analysis", "🔥 GradCAM samples", "📝 Reports"]
    )

    # Discover models available under runs/system_eval
    model_dirs = []
    if PERF_DIR.exists():
        for child in sorted(PERF_DIR.iterdir()):
            if child.is_dir() and (child / "error_analysis").exists():
                model_dirs.append(child.name)

    with sub_tabs[0]:
        if not model_dirs:
            st.info("No error-analysis artifacts found in `runs/system_eval/*/error_analysis/`.")
        else:
            model_choice = st.selectbox(
                "Model",
                options=model_dirs,
                key="perf_model",
            )
            analysis_dir = PERF_DIR / model_choice / "error_analysis"
            pngs = sorted(p for p in analysis_dir.glob("*.png"))

            if not pngs:
                st.info(f"No PNGs found in `{analysis_dir}`.")
            else:
                for png in pngs:
                    st.markdown(f"**{png.stem.replace('_', ' ').title()}**")
                    st.image(str(png), width="stretch")
                    st.markdown('<div class="section-sep"></div>', unsafe_allow_html=True)

    with sub_tabs[1]:
        if not GRADCAM_DIR.exists():
            st.info("`runs/gradcam/` not found.")
        else:
            gradcam_files = sorted(GRADCAM_DIR.glob("*.png"))
            if not gradcam_files:
                st.info("No GradCAM PNGs available. Run `python gradcam_visualize.py` first.")
            else:
                cols = st.columns(2)
                for i, png in enumerate(gradcam_files):
                    with cols[i % 2]:
                        st.markdown('<div class="gallery-card">', unsafe_allow_html=True)
                        st.markdown(
                            f'<div class="gallery-title">{png.stem}</div>',
                            unsafe_allow_html=True,
                        )
                        st.image(str(png), width="stretch")
                        st.markdown("</div>", unsafe_allow_html=True)

    with sub_tabs[2]:
        report_paths = []
        if PERF_DIR.exists():
            for d in PERF_DIR.iterdir():
                rp = d / "error_analysis" / "error_analysis_report.md"
                if rp.exists():
                    report_paths.append(rp)

        if not report_paths:
            st.info("No error-analysis reports found.")
        else:
            for rp in report_paths:
                st.markdown(f"### `{rp.parent.parent.name}` / error_analysis_report.md")
                st.markdown(rp.read_text(encoding="utf-8"))
                st.markdown('<div class="section-sep"></div>', unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# TAB 4 — About
# ──────────────────────────────────────────────────────────────────────────────
with tab_about:
    st.markdown(
        """
<div class="card">
    <h2 style="margin-top:0;">🔬 PCB Defect Detector</h2>
    <p style="color:rgba(255,255,255,0.65);">
        A two-stage deep-learning pipeline that locates and classifies manufacturing
        defects on printed circuit boards.
    </p>
    <h3>How it works</h3>
    <ol style="color:rgba(255,255,255,0.75); line-height:1.7;">
        <li><b>Stage 1 — YOLO detection:</b> finds candidate defect regions in the PCB image.</li>
        <li><b>Stage 2 — CNN classification:</b> classifies each crop into one of six defect categories.</li>
        <li><b>GradCAM (optional):</b> visualises which pixels drove the CNN's decision.</li>
    </ol>
    <h3>Defect classes</h3>
    <p>
        <span class="defect-badge defect-missing_hole">missing_hole</span>
        <span class="defect-badge defect-mouse_bite">mouse_bite</span>
        <span class="defect-badge defect-open_circuit">open_circuit</span>
        <span class="defect-badge defect-short">short</span>
        <span class="defect-badge defect-spur">spur</span>
        <span class="defect-badge defect-spurious_copper">spurious_copper</span>
    </p>
    <h3>Tips</h3>
    <ul style="color:rgba(255,255,255,0.65); line-height:1.7;">
        <li>Lower the <b>Confidence threshold</b> in the sidebar to catch subtle defects.</li>
        <li>Use the <b>Detector</b> tab for deep inspection (GradCAM + slider).</li>
        <li>Use the <b>Gallery</b> tab to compare many boards at once.</li>
    </ul>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
<div class="card">
    <h3 style="margin-top:0;">📁 Project layout</h3>
    <pre style="color:rgba(255,255,255,0.75); font-size:.82rem; line-height:1.5;">
app.py                      ← this Streamlit UI
before_after_slider.py      ← HTML/JS comparison widget
gradcam_ui.py               ← per-crop GradCAM helpers
stage12_yolo_cnn_system.py  ← Stage-1 + Stage-2 inference pipeline
gradcam_visualize.py        ← offline GradCAM CLI
stage2_cnn_utils.py         ← model loading & inference utilities
runs/
├── stage2/                 ← CNN checkpoints (ResNet18/50, EfficientNet-B2)
├── detect/                 ← YOLO checkpoints
├── gradcam/                ← pre-generated GradCAM PNGs
└── system_eval/            ← error-analysis artifacts
    </pre>
</div>
""",
        unsafe_allow_html=True,
    )
