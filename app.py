"""
PCB Defect Detector — Streamlit Web Application
Two-stage pipeline: YOLO detection  →  CNN classification
"""

import json
import tempfile
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# ──────────────────────────────────────────────────────────────
# Page configuration (must be first Streamlit call)
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PCB Defect Detector",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────
# Custom CSS — premium dark-compatible styling
# ──────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Header gradient banner ── */
.header-banner {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    border-radius: 16px;
    padding: 2.5rem 2rem;
    margin-bottom: 2rem;
    text-align: center;
    position: relative;
    overflow: hidden;
    border: 1px solid rgba(255,255,255,0.06);
    box-shadow: 0 8px 32px rgba(0,0,0,0.35);
}
.header-banner::before {
    content: '';
    position: absolute;
    top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: radial-gradient(circle at 30% 40%, rgba(100,100,255,0.08) 0%, transparent 60%),
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
    color: rgba(255,255,255,0.55);
    font-size: .95rem;
    position: relative;
}

/* ── Card container ── */
.card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px;
    padding: 1.5rem;
    margin-bottom: 1.2rem;
    backdrop-filter: blur(12px);
    box-shadow: 0 4px 20px rgba(0,0,0,0.20);
    transition: box-shadow 0.3s ease, border-color 0.3s ease;
}
.card:hover {
    border-color: rgba(167,139,250,0.25);
    box-shadow: 0 6px 28px rgba(0,0,0,0.30);
}

/* ── Metric cards ── */
.metric-row {
    display: flex;
    gap: 1rem;
    margin-bottom: 1.5rem;
}
.metric-card {
    flex: 1;
    background: linear-gradient(145deg, rgba(255,255,255,0.04), rgba(255,255,255,0.01));
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 1.4rem;
    text-align: center;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.metric-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 24px rgba(0,0,0,0.25);
}
.metric-value {
    font-size: 2rem;
    font-weight: 700;
    background: linear-gradient(90deg, #a78bfa, #60a5fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.metric-label {
    font-size: .82rem;
    color: rgba(255,255,255,0.50);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: .3rem;
}

/* ── Defect-type colour badges ── */
.defect-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 6px;
    font-size: .78rem;
    font-weight: 600;
    letter-spacing: 0.02em;
}
.defect-missing_hole   { background: rgba(239,68,68,0.18);  color: #f87171; }
.defect-mouse_bite     { background: rgba(251,146,60,0.18); color: #fb923c; }
.defect-open_circuit   { background: rgba(250,204,21,0.18); color: #facc15; }
.defect-short          { background: rgba(52,211,153,0.18); color: #34d399; }
.defect-spur           { background: rgba(96,165,250,0.18); color: #60a5fa; }
.defect-spurious_copper{ background: rgba(167,139,250,0.18);color: #a78bfa; }
/* generic fallback */
.defect-other          { background: rgba(148,163,184,0.18);color: #94a3b8; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: rgba(15,12,41,0.60);
    backdrop-filter: blur(18px);
}
section[data-testid="stSidebar"] .stMarkdown h2 {
    font-size: 1.1rem;
    font-weight: 700;
    color: #a78bfa;
    margin-bottom: .2rem;
    letter-spacing: -0.01em;
}

/* ── Streamlit element tweaks ── */
.stDataFrame { border-radius: 10px; overflow: hidden; }
div[data-testid="stFileUploader"] {
    border: 2px dashed rgba(167,139,250,0.25) !important;
    border-radius: 12px;
    transition: border-color 0.3s ease;
}
div[data-testid="stFileUploader"]:hover {
    border-color: rgba(167,139,250,0.50) !important;
}

/* ── Download button ── */
.stDownloadButton > button {
    background: linear-gradient(135deg, #302b63, #24243e) !important;
    border: 1px solid rgba(167,139,250,0.30) !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    transition: all 0.25s ease !important;
}
.stDownloadButton > button:hover {
    border-color: rgba(167,139,250,0.60) !important;
    box-shadow: 0 4px 16px rgba(167,139,250,0.20) !important;
}

/* ── Section separator ── */
.section-sep {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(167,139,250,0.25), transparent);
    margin: 2rem 0;
}

/* ── status indicator ── */
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
</style>
""",
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────
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


def _defect_css_class(label: str) -> str:
    key = label.lower().replace(" ", "_")
    return key if key in DEFECT_COLORS else "other"


# ──────────────────────────────────────────────────────────────
# Model loading with caching
# ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_pipeline(yolo_path: str, cnn_path: str, imgsz: int, conf: float, iou: float):
    """Load the two-stage pipeline. Cached so models stay in memory."""
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


# ──────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔬 PCB Defect Detector")
    st.caption(
        "Two-stage pipeline: **YOLO** object detection → **CNN** defect classification. "
        "Upload PCB images to find & classify manufacturing defects."
    )

    st.markdown('<div class="section-sep"></div>', unsafe_allow_html=True)

    # ── CNN model selector ──
    st.markdown("## 🧠 CNN Model")
    cnn_choice = st.selectbox(
        "Architecture",
        options=list(CNN_OPTIONS.keys()),
        index=0,
        help="Stage-2 classifier backbone. ResNet18 is fastest; EfficientNet-B2 is most accurate.",
    )
    cnn_path = st.text_input(
        "Checkpoint path",
        value=CNN_OPTIONS[cnn_choice],
        help="Relative to project root.",
    )

    st.markdown('<div class="section-sep"></div>', unsafe_allow_html=True)

    # ── YOLO settings ──
    st.markdown("## ⚙️ YOLO Settings")
    yolo_path = st.text_input(
        "YOLO weights",
        value=DEFAULT_YOLO_PATH,
        help="Relative to project root.",
    )
    yolo_conf = st.slider(
        "Confidence threshold",
        min_value=0.05,
        max_value=0.95,
        value=0.25,
        step=0.05,
        help="Minimum detection confidence.",
    )
    yolo_iou = st.slider(
        "IoU threshold (NMS)",
        min_value=0.10,
        max_value=0.95,
        value=0.70,
        step=0.05,
        help="Non-maximum-suppression IoU overlap threshold.",
    )
    yolo_imgsz = st.selectbox(
        "Image size",
        options=[640, 768, 1024],
        index=1,
        help="Input resolution for YOLO inference.",
    )

    st.markdown('<div class="section-sep"></div>', unsafe_allow_html=True)
    st.caption("Built with Streamlit · YOLO + CNN pipeline")


# ──────────────────────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────────────────────
st.markdown(
    """
<div class="header-banner">
    <h1>PCB Defect Detector</h1>
    <p>Upload PCB images below and let the two-stage AI pipeline find manufacturing defects in seconds.</p>
</div>
""",
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────
# Load models
# ──────────────────────────────────────────────────────────────
pipeline = None
model_ready = False

try:
    with st.spinner("Loading models… this may take a moment on first run."):
        pipeline = load_pipeline(yolo_path, cnn_path, yolo_imgsz, yolo_conf, yolo_iou)
    model_ready = True
except FileNotFoundError as exc:
    st.error(f"**Model file missing**\n\n{exc}")
    st.info(
        "Make sure you have trained models in the expected paths. "
        "You can adjust the paths in the sidebar."
    )
except Exception as exc:
    st.error(f"**Failed to load models**\n\n{exc}")

# Status indicator
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

st.markdown('<div class="section-sep"></div>', unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# File uploader
# ──────────────────────────────────────────────────────────────
uploaded_files = st.file_uploader(
    "Drop PCB images here",
    type=["jpg", "jpeg", "png", "bmp"],
    accept_multiple_files=True,
    help="Supports JPG, JPEG, PNG, and BMP formats.",
)

if uploaded_files and model_ready:
    from stage12_yolo_cnn_system import annotate_predictions

    all_predictions: list[dict] = []
    all_defect_labels: list[str] = []
    all_confidences: list[float] = []

    for idx, uploaded in enumerate(uploaded_files):
        st.markdown(f'<div class="card">', unsafe_allow_html=True)
        st.subheader(f"📄 {uploaded.name}")

        # Save to temp file (pipeline needs a file path)
        suffix = Path(uploaded.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.getbuffer())
            tmp_path = tmp.name

        try:
            # Run inference
            result = pipeline.predict_image(tmp_path)
            annotated_bgr = annotate_predictions(tmp_path, result)
            annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

            # Original & annotated side-by-side
            col_orig, col_ann = st.columns(2)
            with col_orig:
                st.markdown("**Original**")
                original_img = Image.open(tmp_path)
                st.image(original_img, use_container_width=True)

            with col_ann:
                st.markdown("**Detections**")
                st.image(annotated_rgb, use_container_width=True)

            preds = result.get("predictions", [])
            if preds:
                # Build results table
                rows = []
                for p in preds:
                    label = p["stage2_label"]
                    css_cls = _defect_css_class(label)
                    all_defect_labels.append(label)
                    all_confidences.append(p["combined_confidence"])

                    rows.append(
                        {
                            "Defect Type": label,
                            "Stage 1 Conf": f"{p['stage1_confidence']:.3f}",
                            "Stage 2 Conf": f"{p['stage2_confidence']:.3f}",
                            "Combined Conf": f"{p['combined_confidence']:.3f}",
                            "Bbox (x1 y1 x2 y2)": f"{p['bbox']}",
                        }
                    )

                df = pd.DataFrame(rows)
                st.dataframe(
                    df,
                    use_container_width=True,
                    hide_index=True,
                )

                # Defect-type badges
                badge_html = " ".join(
                    f'<span class="defect-badge defect-{_defect_css_class(r["Defect Type"])}">'
                    f'{r["Defect Type"]}</span>'
                    for r in rows
                )
                st.markdown(badge_html, unsafe_allow_html=True)
            else:
                st.success("✅ No defects detected — board looks clean!")

            all_predictions.append(result)

            # Download JSON
            result_json = json.dumps(result, indent=2)
            st.download_button(
                label="⬇ Download JSON",
                data=result_json,
                file_name=f"{Path(uploaded.name).stem}_results.json",
                mime="application/json",
                key=f"dl_{idx}",
            )

        except Exception as exc:
            st.error(f"Error processing **{uploaded.name}**: {exc}")

        st.markdown("</div>", unsafe_allow_html=True)

    # ──────────────────────────────────────────────────────────
    # Summary statistics
    # ──────────────────────────────────────────────────────────
    if all_predictions:
        st.markdown('<div class="section-sep"></div>', unsafe_allow_html=True)
        st.markdown(
            """
<div class="header-banner" style="padding:1.6rem 2rem;">
    <h1 style="font-size:1.5rem;">Summary Statistics</h1>
    <p>Aggregated results across all uploaded images</p>
</div>
""",
            unsafe_allow_html=True,
        )

        total_defects = len(all_defect_labels)
        avg_conf = float(np.mean(all_confidences)) if all_confidences else 0.0
        n_images = len(all_predictions)
        defect_counter = Counter(all_defect_labels)

        # Metric cards
        st.markdown(
            f"""
<div class="metric-row">
    <div class="metric-card">
        <div class="metric-value">{n_images}</div>
        <div class="metric-label">Images Analysed</div>
    </div>
    <div class="metric-card">
        <div class="metric-value">{total_defects}</div>
        <div class="metric-label">Total Defects</div>
    </div>
    <div class="metric-card">
        <div class="metric-value">{avg_conf:.1%}</div>
        <div class="metric-label">Avg Combined Confidence</div>
    </div>
    <div class="metric-card">
        <div class="metric-value">{len(defect_counter)}</div>
        <div class="metric-label">Defect Types</div>
    </div>
</div>
""",
            unsafe_allow_html=True,
        )

        # Bar chart — defects by type
        if defect_counter:
            chart_df = (
                pd.DataFrame.from_dict(defect_counter, orient="index", columns=["Count"])
                .sort_values("Count", ascending=False)
            )
            chart_df.index.name = "Defect Type"
            st.bar_chart(chart_df, use_container_width=True)

        # Full JSON download
        full_json = json.dumps(all_predictions, indent=2)
        st.download_button(
            label="⬇ Download All Results (JSON)",
            data=full_json,
            file_name="pcb_all_results.json",
            mime="application/json",
            key="dl_all",
        )

elif uploaded_files and not model_ready:
    st.warning("Please fix the model configuration in the sidebar before uploading images.")
elif not uploaded_files:
    # Placeholder when no files are uploaded
    st.markdown(
        """
<div class="card" style="text-align:center; padding:3rem 2rem;">
    <p style="font-size:2.5rem; margin:0;">📷</p>
    <p style="font-size:1.05rem; color:rgba(255,255,255,0.55); margin:.8rem 0 .3rem 0;">
        Drag & drop PCB images above to start inspection
    </p>
    <p style="font-size:.82rem; color:rgba(255,255,255,0.35);">
        Supports JPG · JPEG · PNG · BMP — multiple files at once
    </p>
</div>
""",
        unsafe_allow_html=True,
    )
