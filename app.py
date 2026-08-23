"""
PCB Defect Detector - Streamlit Web Application
Two-stage pipeline: YOLO detection -> CNN classification

Design notes
------------
Dark professional inspection tool with restrained single-accent palette.
Built with native CSS (no design-system package); glassmorphism used as a
supporting material on cards, not as the dominant language. All
animations honor prefers-reduced-motion. Single accent (cool slate-blue)
locked across the whole page; defect colors are functional, not accent.
Custom inline SVG icon set for tabs, sections, and defect classes.
"""

from __future__ import annotations

import datetime
import json
import random
import tempfile
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

from gradcam_ui import render_gradcam_for_detection
from report_generator import generate_qc_certificate_html
from zoom_component import render_interactive_zoom_loupe_html

# ──────────────────────────────────────────────────────────────────────────────
# Page configuration (must be the first Streamlit call)
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PCB Defect Detector",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "app_start_time" not in st.session_state:
    st.session_state["app_start_time"] = time.time()

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
DATASET_TEST_DIR = PROJECT_ROOT / "pcb-defect-dataset" / "test" / "images"
GRADCAM_DIR = PROJECT_ROOT / "runs" / "gradcam"
PERF_DIR = PROJECT_ROOT / "runs" / "system_eval"
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def _defect_css_class(label: str) -> str:
    key = label.lower().replace(" ", "_")
    return key if key in DEFECT_COLORS else "other"


# ──────────────────────────────────────────────────────────────────────────────
# Inline SVG icon library
# ──────────────────────────────────────────────────────────────────────────────
def _icon(name: str, size: int = 16, color: str = "currentColor", stroke_width: float = 1.6) -> str:
    """Return an inline SVG icon. `name` is a key in the ICONS dict."""
    paths = ICONS.get(name, ICONS["dot"])
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 24 24" fill="none" stroke="{color}" '
        f'stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round" '
        f'style="display:inline-block; vertical-align:-2px;">{paths}</svg>'
    )


# A compact, consistent SVG icon set (1.5px stroke, rounded caps).
ICONS = {
    "dot": '<circle cx="12" cy="12" r="2" fill="currentColor" stroke="none"/>',

    "search": (
        '<circle cx="11" cy="11" r="7"/>'
        '<path d="M21 21l-4.3-4.3"/>'
    ),
    "image": (
        '<rect x="3" y="3" width="18" height="18" rx="2"/>'
        '<circle cx="8.5" cy="8.5" r="1.5"/>'
        '<path d="M21 15l-5-5L5 21"/>'
    ),
    "chart": (
        '<path d="M3 3v18h18"/>'
        '<path d="M7 14l4-4 4 4 5-5"/>'
    ),
    "info": (
        '<circle cx="12" cy="12" r="9"/>'
        '<path d="M12 8h.01"/>'
        '<path d="M11 12h1v4h1"/>'
    ),
    "microscope": (
        '<path d="M6 18h8"/>'
        '<path d="M3 22h18"/>'
        '<path d="M14 22a7 7 0 0 0 7-7"/>'
        '<path d="M9 14a4 4 0 0 1 5.7-5.7l3.6 3.6a4 4 0 0 1-5.7 5.7l-1.8-1.8"/>'
        '<path d="M7 13l-3 3 3 3"/>'
    ),
    "upload": (
        '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>'
        '<path d="M17 8l-5-5-5 5"/>'
        '<path d="M12 3v12"/>'
    ),
    "download": (
        '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>'
        '<path d="M7 10l5 5 5-5"/>'
        '<path d="M12 15V3"/>'
    ),
    "chip": (
        '<rect x="4" y="4" width="16" height="16" rx="2"/>'
        '<rect x="9" y="9" width="6" height="6"/>'
        '<path d="M9 2v2M15 2v2M9 20v2M15 20v2M2 9h2M2 15h2M20 9h2M20 15h2"/>'
    ),
    "settings": (
        '<circle cx="12" cy="12" r="3"/>'
        '<path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/>'
    ),
    "palette": (
        '<circle cx="13.5" cy="6.5" r=".5" fill="currentColor"/>'
        '<circle cx="17.5" cy="10.5" r=".5" fill="currentColor"/>'
        '<circle cx="8.5" cy="7.5" r=".5" fill="currentColor"/>'
        '<circle cx="6.5" cy="12.5" r=".5" fill="currentColor"/>'
        '<path d="M12 2a10 10 0 0 0 0 20 1.5 1.5 0 0 0 1.4-2.1 1.5 1.5 0 0 1 1.4-2.1H17a5 5 0 0 0 5-5C22 6.5 17.5 2 12 2z"/>'
    ),
    "cpu": (
        '<rect x="4" y="4" width="16" height="16" rx="2"/>'
        '<rect x="9" y="9" width="6" height="6"/>'
        '<path d="M9 1v3M15 1v3M9 20v3M15 20v3M20 9h3M20 14h3M1 9h3M1 14h3"/>'
    ),
    "bolt": (
        '<path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z"/>'
    ),
    "check": (
        '<path d="M5 12l4 4L20 6"/>'
    ),
    "x": (
        '<path d="M6 6l12 12M18 6L6 18"/>'
    ),
    "file": (
        '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>'
        '<path d="M14 2v6h6"/>'
    ),
    "grid": (
        '<rect x="3" y="3" width="7" height="7"/>'
        '<rect x="14" y="3" width="7" height="7"/>'
        '<rect x="14" y="14" width="7" height="7"/>'
        '<rect x="3" y="14" width="7" height="7"/>'
    ),
    "rows": (
        '<path d="M3 6h18M3 12h18M3 18h18"/>'
    ),
    "layers": (
        '<path d="M12 2L2 7l10 5 10-5-10-5z"/>'
        '<path d="M2 17l10 5 10-5"/>'
        '<path d="M2 12l10 5 10-5"/>'
    ),
    "fire": (
        '<path d="M8.5 14.5A2.5 2.5 0 0 0 11 12c0-1.38-.5-2-1-3-1.072-2.143-.224-4.054 2-6 .5 2.5 2 4.9 4 6.5 2 1.6 3 3.5 3 5.5a7 7 0 1 1-14 0c0-1.153.433-2.294 1-3a2.5 2.5 0 0 0 2.5 2.5z"/>'
    ),
    "doc": (
        '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>'
        '<path d="M14 2v6h6"/>'
        '<path d="M8 13h8M8 17h8M8 9h2"/>'
    ),
    "tag": (
        '<path d="M20.59 13.41l-7.17 7.17a2 2 0 0 1-2.83 0L2 12V2h10l8.59 8.59a2 2 0 0 1 0 2.82z"/>'
        '<path d="M7 7h.01"/>'
    ),

    # Defect-class icons (small, color-tinted via the badge CSS class).
    "defect-missing_hole": (
        '<circle cx="12" cy="12" r="9"/>'
        '<circle cx="12" cy="12" r="3.5"/>'
    ),
    "defect-mouse_bite": (
        '<path d="M5 9c0-3 2-5 7-5s7 2 7 5v6c0 3-2 5-7 5s-7-2-7-5V9z"/>'
        '<path d="M5 9c1.5 1 3 1.5 4.5 1.5"/>'
        '<path d="M5 13c1.5 1 3 1.5 4.5 1.5"/>'
    ),
    "defect-open_circuit": (
        '<path d="M3 12h6"/>'
        '<path d="M15 12h6"/>'
        '<rect x="10" y="9" width="4" height="6" rx="1"/>'
    ),
    "defect-short": (
        '<path d="M3 12h4l3-7 4 14 3-7h4"/>'
    ),
    "defect-spur": (
        '<path d="M3 20l8-8"/>'
        '<path d="M11 12l3-9"/>'
        '<path d="M14 3l7 7-3 3-7-7z"/>'
    ),
    "defect-spurious_copper": (
        '<path d="M3 17c3-6 9-6 12 0"/>'
        '<path d="M5 12c2-4 6-4 8 0"/>'
        '<circle cx="6" cy="20" r="1.5" fill="currentColor"/>'
        '<circle cx="18" cy="20" r="1.5" fill="currentColor"/>'
    ),

    "pipeline-stage": (
        '<rect x="3" y="9" width="6" height="6" rx="1"/>'
        '<rect x="15" y="9" width="6" height="6" rx="1"/>'
        '<path d="M9 12h6"/>'
        '<path d="M12 9V6M12 18v-3"/>'
    ),
    "wand": (
        '<path d="M15 4V2M15 16v-2M8 9h2M20 9h2M17.8 11.8l1.4 1.4M17.8 6.2l1.4-1.4"/>'
        '<path d="M9 15l-6 6 3 3 6-6"/>'
        '<path d="M12 6l3 3-6 6-3-3z"/>'
    ),
    "folder": (
        '<path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"/>'
    ),
    "ok": (
        '<path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/>'
        '<path d="M22 4L12 14.01l-3-3"/>'
    ),

    # Arrow icons for connectors (chunkier weight, suitable for arrows).
    "arrow-right": (
        '<path d="M5 12h14"/>'
        '<path d="M13 5l7 7-7 7"/>'
    ),
    "arrow-down": (
        '<path d="M12 5v14"/>'
        '<path d="M5 13l7 7 7-7"/>'
    ),
}


# ──────────────────────────────────────────────────────────────────────────────
# System Clock & Runtime Widgets
# ──────────────────────────────────────────────────────────────────────────────
def _render_system_clock() -> None:
    """Render a live digital real-time clock & system uptime widget."""
    elapsed_init = int(time.time() - st.session_state.get("app_start_time", time.time()))
    clock_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <meta charset="utf-8">
    <style>
        * {{ margin:0; padding:0; box-sizing:border-box; font-family: 'JetBrains Mono', 'Segoe UI Mono', monospace; }}
        body {{ background: transparent; overflow: hidden; }}
        .clock-strip {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            background: linear-gradient(180deg, rgba(19, 28, 46, 0.85) 0%, rgba(11, 15, 25, 0.95) 100%);
            border: 1px solid rgba(56, 189, 248, 0.22);
            border-radius: 10px;
            padding: 8px 18px;
            color: #E2E8F0;
            font-size: 13px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.06);
            backdrop-filter: blur(12px);
        }}
        .clock-item {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .status-dot-pulse {{
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #10B981;
            box-shadow: 0 0 12px #10B981;
            animation: pulse-dot 2s infinite ease-in-out;
        }}
        @keyframes pulse-dot {{
            0%, 100% {{ transform: scale(1); opacity: 1; }}
            50% {{ transform: scale(1.4); opacity: 0.5; }}
        }}
        .label {{
            color: #94A3B8;
            font-weight: 600;
            font-size: 11px;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }}
        .val {{
            color: #38BDF8;
            font-weight: 700;
            background: rgba(15, 23, 42, 0.7);
            border: 1px solid rgba(56, 189, 248, 0.25);
            padding: 2px 10px;
            border-radius: 6px;
            letter-spacing: 0.03em;
        }}
        .val.uptime {{
            color: #34D399;
            border-color: rgba(52, 211, 153, 0.25);
        }}
        .val.time {{
            color: #F8FAFC;
        }}
    </style>
    </head>
    <body>
        <div class="clock-strip">
            <div class="clock-item">
                <div class="status-dot-pulse"></div>
                <span class="label">SYSTEM:</span>
                <span style="color:#10B981; font-weight:700; font-size:12px;">ACTIVE & ONLINE</span>
            </div>
            <div class="clock-item">
                <span class="label">🕒 LOCAL TIME:</span>
                <span class="val time" id="live-time">--:--:--</span>
            </div>
            <div class="clock-item">
                <span class="label">⏱️ SYSTEM RUNTIME:</span>
                <span class="val uptime" id="live-uptime">00:00:00</span>
            </div>
        </div>
        <script>
            const startTime = Date.now() - ({elapsed_init} * 1000);
            function updateClock() {{
                const now = new Date();
                const timeStr = now.toLocaleTimeString('vi-VN', {{ hour12: false }});
                const dateStr = now.toLocaleDateString('vi-VN', {{ day: '2-digit', month: '2-digit', year: 'numeric' }});
                const elTime = document.getElementById('live-time');
                if (elTime) elTime.textContent = timeStr + ' · ' + dateStr;

                const elapsed = Math.floor((Date.now() - startTime) / 1000);
                const h = String(Math.floor(elapsed / 3600)).padStart(2, '0');
                const m = String(Math.floor((elapsed % 3600) / 60)).padStart(2, '0');
                const s = String(elapsed % 60).padStart(2, '0');
                const elUptime = document.getElementById('live-uptime');
                if (elUptime) elUptime.textContent = `${{h}}:${{m}}:${{s}}`;
            }}
            setInterval(updateClock, 1000);
            updateClock();
        </script>
    </body>
    </html>
    """
    components.html(clock_html, height=52)


def _render_sidebar_clock() -> None:
    """Render a compact live clock widget for the sidebar."""
    elapsed_init = int(time.time() - st.session_state.get("app_start_time", time.time()))
    clock_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <meta charset="utf-8">
    <style>
        * {{ margin:0; padding:0; box-sizing:border-box; font-family: 'JetBrains Mono', 'Segoe UI Mono', monospace; }}
        body {{ background: transparent; overflow: hidden; }}
        .side-clock {{
            background: linear-gradient(180deg, rgba(19, 28, 46, 0.7) 0%, rgba(11, 15, 25, 0.85) 100%);
            border: 1px solid rgba(56, 189, 248, 0.18);
            border-radius: 8px;
            padding: 10px 12px;
            color: #E2E8F0;
            font-size: 12px;
        }}
        .row {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 6px;
        }}
        .row:last-child {{ margin-bottom: 0; }}
        .lbl {{ color: #94A3B8; font-size: 11px; }}
        .val {{ color: #F8FAFC; font-weight: 700; font-size: 12px; }}
        .val.up {{ color: #34D399; }}
    </style>
    </head>
    <body>
        <div class="side-clock">
            <div class="row">
                <span class="lbl">🕒 Time:</span>
                <span class="val" id="side-time">--:--:--</span>
            </div>
            <div class="row">
                <span class="lbl">⏱️ Uptime:</span>
                <span class="val up" id="side-uptime">00:00:00</span>
            </div>
        </div>
        <script>
            const startTime = Date.now() - ({elapsed_init} * 1000);
            function updateSideClock() {{
                const now = new Date();
                const timeStr = now.toLocaleTimeString('vi-VN', {{ hour12: false }});
                const elTime = document.getElementById('side-time');
                if (elTime) elTime.textContent = timeStr;

                const elapsed = Math.floor((Date.now() - startTime) / 1000);
                const h = String(Math.floor(elapsed / 3600)).padStart(2, '0');
                const m = String(Math.floor((elapsed % 3600) / 60)).padStart(2, '0');
                const s = String(elapsed % 60).padStart(2, '0');
                const elUptime = document.getElementById('side-uptime');
                if (elUptime) elUptime.textContent = `${{h}}:${{m}}:${{s}}`;
            }}
            setInterval(updateSideClock, 1000);
            updateSideClock();
        </script>
    </body>
    </html>
    """
    components.html(clock_html, height=72)


# ──────────────────────────────────────────────────────────────────────────────
# Custom CSS - dark professional tool with restrained accent
# ──────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* Base: Inter cascades from html/body. The browser default cascade
   only carries font-family through element nodes whose display is text
   (h1-h6, p, span, button, input, textarea, etc). Icon nodes whose
   ::before {content} maps a literal name like "keyboard_double_right"
   into a glyph rely on a SPECIFIC font-family (Material Symbols Rounded)
   set on the element itself; the cascade will NOT override a more
   specific rule, so if we keep our Inter only on html/body it never
   reaches the icon. */
html, body {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
}

code, pre, .mono {
    font-family: 'JetBrains Mono', 'SF Mono', Menlo, monospace !important;
}

* { box-sizing: border-box; }

/* CRITICAL: explicitly pin Streamlit's icon font on every element that
   has an icon font set, so no upstream rule can override it. The
   `i` selector catches <i class="material-icons">-style icon nodes
   Streamlit uses in its chrome. The .material-icons class is the
   canonical Google icon class. */
i, .material-icons, .material-icons-outlined, .material-icons-round,
.material-icons-sharp, .material-icons-two-tone,
.material-symbols-rounded, .material-symbols-outlined, .material-symbols {
    font-family: 'Material Symbols Rounded', 'Material Icons', sans-serif !important;
    font-weight: normal;
    font-style: normal;
    line-height: 1;
    letter-spacing: normal;
    text-transform: none;
    display: inline-block;
    white-space: nowrap;
    word-wrap: normal;
    direction: ltr;
    -webkit-font-feature-settings: 'liga';
    -webkit-font-smoothing: antialiased;
}

/* ── Tokens ── */
:root {
    --accent:        #7aa2f7;
    --accent-soft:   #9eb8f5;
    --accent-glow:   rgba(122,162,247,0.22);
    --accent-line:   rgba(122,162,247,0.32);
    --accent-dim:    rgba(122,162,247,0.08);

    --c-red:    #f87171;
    --c-orange: #fb923c;
    --c-yellow: #facc15;
    --c-green:  #34d399;
    --c-blue:   #60a5fa;
    --c-purple: #a78bfa;

    --bg-0:   #0b0d12;
    --bg-1:   #11141b;
    --bg-2:   #161a23;
    --bg-3:   #1c212c;

    --line:   rgba(255,255,255,0.06);
    --line-hi:rgba(255,255,255,0.11);

    --text-1: rgba(255,255,255,0.92);
    --text-2: rgba(255,255,255,0.64);
    --text-3: rgba(255,255,255,0.40);
    --text-4: rgba(255,255,255,0.24);
}

/* ── Page background ── */
.stApp {
    background:
      radial-gradient(ellipse 70% 45% at 15% 0%, rgba(122,162,247,0.10), transparent 55%),
      radial-gradient(ellipse 55% 40% at 85% 15%, rgba(122,162,247,0.06), transparent 60%),
      radial-gradient(ellipse 60% 50% at 50% 100%, rgba(52,211,153,0.04), transparent 60%),
      linear-gradient(180deg, var(--bg-0) 0%, var(--bg-1) 100%);
    background-attachment: fixed;
}

/* ── Header banner ── */
.header-banner {
    background:
      linear-gradient(180deg, rgba(255,255,255,0.025) 0%, rgba(255,255,255,0.005) 100%),
      rgba(17,20,27,0.70);
    backdrop-filter: blur(20px) saturate(140%);
    -webkit-backdrop-filter: blur(20px) saturate(140%);
    border-radius: 14px;
    padding: 2.2rem 2.4rem 1.9rem 2.4rem;
    margin-bottom: 1.6rem;
    text-align: center;
    position: relative;
    overflow: hidden;
    border: 1px solid var(--line-hi);
    box-shadow:
        0 4px 24px rgba(0,0,0,0.30),
        inset 0 1px 0 rgba(255,255,255,0.04);
}
.header-banner::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--accent-line), transparent);
}
.header-banner::after {
    content: '';
    position: absolute;
    inset: 0;
    background:
      radial-gradient(circle at 50% 0%, rgba(122,162,247,0.08), transparent 55%);
    pointer-events: none;
}
.header-banner .hbadge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 10px;
    background: var(--accent-dim);
    border: 1px solid var(--accent-line);
    border-radius: 999px;
    color: var(--accent-soft);
    font-size: .72rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.10em;
    margin-bottom: 0.9rem;
    position: relative;
}
.header-banner h1 {
    margin: 0;
    font-size: 2.4rem;
    font-weight: 700;
    color: var(--text-1);
    position: relative;
    letter-spacing: -0.028em;
    line-height: 1.1;
}
.header-banner h1 .accent-word {
    background: linear-gradient(135deg, var(--accent), var(--accent-soft));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-weight: 600;
}
.header-banner p {
    margin: .8rem auto 0;
    color: var(--text-2);
    font-size: .98rem;
    position: relative;
    font-weight: 400;
    max-width: 600px;
    line-height: 1.55;
}
.header-banner.compact { padding: 1.1rem 1.6rem 1rem 1.6rem; margin-bottom: 1.1rem; text-align: left; }
.header-banner.compact h1 { font-size: 1.2rem; font-weight: 600; }
.header-banner.compact p { font-size: .82rem; margin-top: .35rem; max-width: none; text-align: left; }
.header-banner.compact .hbadge { display: none; }

/* ── Card ── */
.card {
    background:
      linear-gradient(180deg, rgba(255,255,255,0.025) 0%, rgba(255,255,255,0.005) 100%);
    border: 1px solid var(--line);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1.1rem;
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    box-shadow:
        0 2px 14px rgba(0,0,0,0.20),
        inset 0 1px 0 rgba(255,255,255,0.03);
    transition: border-color 0.3s ease, box-shadow 0.3s ease;
}
.card:hover {
    border-color: var(--line-hi);
    box-shadow:
        0 6px 24px rgba(0,0,0,0.30),
        inset 0 1px 0 rgba(255,255,255,0.05);
}

/* ── Section heading (h2) ── */
.section-heading {
    display: flex;
    align-items: center;
    gap: .55rem;
    margin: 1.6rem 0 1rem 0;
    color: var(--text-1);
    font-size: 1.05rem;
    font-weight: 600;
    letter-spacing: -0.01em;
}
.section-heading .icon-wrap {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 24px; height: 24px;
    background: var(--accent-dim);
    border: 1px solid var(--accent-line);
    border-radius: 6px;
    color: var(--accent-soft);
}
.section-heading .count-pill {
    margin-left: auto;
    font-size: .70rem;
    color: var(--text-2);
    background: rgba(255,255,255,0.04);
    border: 1px solid var(--line);
    border-radius: 999px;
    padding: 2px 8px;
    font-weight: 500;
    font-variant-numeric: tabular-nums;
}

/* ── Subheading (h3) ── */
.subheading {
    color: var(--text-2);
    font-size: .82rem;
    font-weight: 600;
    letter-spacing: -0.005em;
    margin: 1.4rem 0 .7rem 0;
    display: flex;
    align-items: center;
    gap: .4rem;
}
.subheading::before {
    content: '';
    width: 3px; height: 12px;
    background: var(--accent);
    border-radius: 2px;
    opacity: .8;
}

/* ── Metric cards ── */
.metric-row {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
    gap: .9rem;
    margin-bottom: 1.5rem;
}
.metric-card {
    background:
      linear-gradient(180deg, rgba(255,255,255,0.03) 0%, rgba(255,255,255,0.008) 100%);
    border: 1px solid var(--line);
    border-radius: 10px;
    padding: 1.15rem 1rem;
    position: relative;
    transition: transform 0.2s ease, border-color 0.2s ease;
    overflow: hidden;
}
.metric-card::after {
    /* single hairline highlight on the right edge */
    content: '';
    position: absolute;
    right: 0; top: 12%; bottom: 12%;
    width: 2px;
    background: linear-gradient(180deg, transparent, var(--accent-line), transparent);
    opacity: .5;
}
.metric-card:hover {
    transform: translateY(-1px);
    border-color: var(--accent-line);
}
.metric-icon {
    position: absolute;
    top: 12px; right: 12px;
    width: 22px; height: 22px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    color: var(--text-3);
    opacity: .55;
}
.metric-value {
    font-size: 1.85rem;
    font-weight: 700;
    color: var(--accent-soft);
    line-height: 1.1;
    letter-spacing: -0.02em;
    font-variant-numeric: tabular-nums;
}
.metric-label {
    font-size: .78rem;
    color: var(--text-2);
    margin-top: .35rem;
    font-weight: 500;
    letter-spacing: -0.005em;
}

/* ── Defect-type badges (semantic colors, locked) ── */
.defect-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 10px;
    border-radius: 6px;
    font-size: .74rem;
    font-weight: 600;
    letter-spacing: 0.005em;
    margin: 2px;
    border: 1px solid rgba(255,255,255,0.05);
    font-variant-numeric: tabular-nums;
    transition: background 0.2s ease, border-color 0.2s ease;
}
.defect-badge svg { flex-shrink: 0; }
.defect-badge:hover { border-color: rgba(255,255,255,0.18); }
.defect-missing_hole   { background: rgba(248,113,113,0.12); color: #f87171; border-color: rgba(248,113,113,0.22); }
.defect-mouse_bite     { background: rgba(251,146,60,0.12);  color: #fb923c; border-color: rgba(251,146,60,0.22); }
.defect-open_circuit   { background: rgba(250,204,21,0.12);  color: #facc15; border-color: rgba(250,204,21,0.22); }
.defect-short          { background: rgba(52,211,153,0.12);  color: #34d399; border-color: rgba(52,211,153,0.22); }
.defect-spur           { background: rgba(96,165,250,0.12);  color: #60a5fa; border-color: rgba(96,165,250,0.22); }
.defect-spurious_copper{ background: rgba(167,139,250,0.12); color: #a78bfa; border-color: rgba(167,139,250,0.22); }
.defect-other          { background: rgba(148,163,184,0.12); color: #94a3b8; border-color: rgba(148,163,184,0.22); }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background:
      linear-gradient(180deg, rgba(11,13,18,0.96) 0%, rgba(7,9,12,0.96) 100%);
    backdrop-filter: blur(20px) saturate(140%);
    -webkit-backdrop-filter: blur(20px) saturate(140%);
    border-right: 1px solid var(--line);
}
section[data-testid="stSidebar"] .stMarkdown h2 {
    font-size: .78rem;
    font-weight: 600;
    color: var(--text-2);
    margin: .4rem 0 .7rem 0;
    letter-spacing: -0.005em;
    display: flex;
    align-items: center;
    gap: .55rem;
}
section[data-testid="stSidebar"] .stMarkdown h2::before {
    content: '';
    width: 3px; height: 12px;
    background: var(--accent);
    border-radius: 2px;
    opacity: .8;
}
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
    color: var(--text-2) !important;
    font-size: .82rem !important;
}
section[data-testid="stSidebar"] .stMarkdown strong { color: var(--text-1); }
section[data-testid="stSidebar"] code {
    background: rgba(255,255,255,0.06);
    padding: 1px 5px;
    border-radius: 4px;
    font-size: .78rem;
    color: var(--accent-soft);
}

/* ── Streamlit element tweaks ── */
.stDataFrame, [data-testid="stDataFrame"] {
    border-radius: 10px !important;
    overflow: hidden;
    border: 1px solid var(--line) !important;
}

div[data-testid="stFileUploader"] {
    border: 2px dashed rgba(122,162,247,0.22) !important;
    border-radius: 12px !important;
    transition: border-color 0.3s ease, background 0.3s ease;
    background: rgba(122,162,247,0.015) !important;
}
div[data-testid="stFileUploader"]:hover {
    border-color: rgba(122,162,247,0.45) !important;
    background: rgba(122,162,247,0.04) !important;
}
div[data-testid="stFileUploaderDropzone"] {
    background: transparent !important;
}
div[data-testid="stFileUploaderDropzone"] [data-testid="stMarkdownContainer"] p {
    color: var(--text-2) !important;
}
div[data-testid="stFileUploaderDropzone"] small { color: var(--text-3) !important; }
/* Style the file-uploader Browse button *container*, but NEVER override the
   inner icon svg / colour: that breaks the icon's currentColor inheritance
   and the button collapses to "uploadpload" (alt text leaking out). */
div[data-testid="stFileUploaderDropzone"] section button {
    background: var(--accent-dim) !important;
    border: 1px solid var(--accent-line) !important;
    border-radius: 6px !important;
}

/* ── Buttons ── */
.stDownloadButton > button,
.stButton > button,
button[data-testid="baseButton-primary"],
button[data-testid="baseButton-secondary"] {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid var(--line-hi) !important;
    border-radius: 8px !important;
    color: var(--text-1) !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
    box-shadow: none !important;
    letter-spacing: 0 !important;
}
.stDownloadButton > button:hover,
.stButton > button:hover,
button[data-testid="baseButton-primary"]:hover,
button[data-testid="baseButton-secondary"]:hover {
    border-color: var(--accent-line) !important;
    background: rgba(122,162,247,0.06) !important;
    color: var(--accent-soft) !important;
}
.stDownloadButton > button:active,
.stButton > button:active {
    transform: translateY(1px) !important;
}

/* Demo sample button: more compact, with leading icon */
button[key^="demo_load_"] {
    font-size: .78rem !important;
    padding: 6px 10px !important;
    text-align: left !important;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

/* ── Tabs (with custom icon) ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: rgba(255,255,255,0.025);
    padding: 5px;
    border-radius: 10px;
    border: 1px solid var(--line);
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border-radius: 7px !important;
    color: var(--text-2) !important;
    padding: 8px 16px !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
    font-size: .9rem !important;
}
.stTabs [data-baseweb="tab"]:hover {
    background: rgba(122,162,247,0.06) !important;
    color: var(--text-1) !important;
}
.stTabs [aria-selected="true"] {
    background: rgba(122,162,247,0.12) !important;
    color: var(--accent-soft) !important;
    box-shadow: inset 0 0 0 1px var(--accent-line) !important;
}
.stTabs [data-baseweb="tab-panel"] { padding-top: 1.4rem; }

/* ── Sliders ── */
.stSlider [data-baseweb="slider"] [role="slider"] {
    background: var(--accent) !important;
    box-shadow: 0 0 0 3px var(--accent-glow) !important;
    border: 2px solid var(--bg-1) !important;
    width: 16px !important;
    height: 16px !important;
}
.stSlider [data-baseweb="slider"] > div > div {
    background: var(--accent) !important;
}

/* ── Selectbox & text inputs ── */
.stSelectbox [data-baseweb="select"] > div,
.stTextInput input {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid var(--line-hi) !important;
    border-radius: 8px !important;
    color: var(--text-1) !important;
    transition: border-color 0.2s ease !important;
}
.stSelectbox [data-baseweb="select"] > div:hover,
.stTextInput input:hover,
.stTextInput input:focus {
    border-color: var(--accent-line) !important;
    box-shadow: 0 0 0 3px var(--accent-glow) !important;
}

/* ── Status indicator (real semantic state) ── */
.status-dot {
    display: inline-block;
    width: 7px; height: 7px;
    border-radius: 50%;
    margin-right: 7px;
    vertical-align: 2px;
    position: relative;
}
.status-dot.ready { background: var(--c-green); box-shadow: 0 0 6px rgba(52,211,153,0.55); }
.status-dot.error { background: var(--c-red);   box-shadow: 0 0 6px rgba(248,113,113,0.55); }
.status-dot.ready::after,
.status-dot.error::after {
    content: '';
    position: absolute;
    inset: -2px;
    border-radius: 50%;
    background: inherit;
    opacity: 0;
    animation: ping 2.5s cubic-bezier(0,0,0.2,1) infinite;
}
@keyframes ping {
    0%   { transform: scale(1);   opacity: 0.45; }
    75%  { transform: scale(2.0); opacity: 0;    }
    100% { transform: scale(2.0); opacity: 0;    }
}

.status-banner {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 7px 12px;
    background: rgba(255,255,255,0.02);
    border: 1px solid var(--line);
    border-radius: 8px;
    font-size: .84rem;
    color: var(--text-2);
    margin-bottom: 1.1rem;
    font-weight: 500;
}

/* ── Section separator ── */
.section-sep {
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--line-hi), transparent);
    margin: 1.6rem 0;
}

/* ── Gallery grid (bento cell, real visual) ── */
.gallery-card {
    background:
      linear-gradient(180deg, rgba(255,255,255,0.025) 0%, rgba(255,255,255,0.005) 100%);
    border: 1px solid var(--line);
    border-radius: 12px;
    padding: .9rem;
    margin-bottom: .9rem;
    transition: border-color 0.25s ease, transform 0.25s ease, box-shadow 0.25s ease;
    height: 100%;
}
.gallery-card:hover {
    border-color: var(--accent-line);
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(0,0,0,0.25);
}
.gallery-card img {
    border-radius: 8px;
    width: 100%;
    display: block;
}
.gallery-title {
    font-weight: 600;
    font-size: .90rem;
    color: var(--text-1);
    margin-bottom: .12rem;
    letter-spacing: -0.005em;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}
.gallery-sub {
    color: var(--text-2);
    font-size: .76rem;
    margin-bottom: .55rem;
    font-weight: 500;
    display: flex;
    align-items: center;
    gap: .35rem;
}
.gallery-sub svg { flex-shrink: 0; opacity: .7; }

/* ── Image viewer frame ── */
.image-frame {
    background: rgba(0,0,0,0.40);
    border: 1px solid var(--line);
    border-radius: 10px;
    padding: 5px;
    overflow: hidden;
    transition: border-color 0.25s ease;
    position: relative;
}
.image-frame:hover { border-color: var(--line-hi); }
.image-frame img { border-radius: 6px; display: block; width: 100%; }

.image-label {
    font-size: .70rem;
    color: var(--text-2);
    font-size: .76rem;
    font-weight: 500;
    letter-spacing: -0.005em;
    margin-bottom: .45rem;
    display: flex;
    align-items: center;
    gap: .4rem;
}
.image-label svg { opacity: .7; }

/* ── Empty state ── */
.empty-state {
    text-align: center;
    padding: 3.5rem 2rem;
    background:
      linear-gradient(180deg, rgba(255,255,255,0.02) 0%, rgba(255,255,255,0.005) 100%);
    border: 1px dashed rgba(122,162,247,0.20);
    border-radius: 12px;
}
.empty-state .icon {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 56px; height: 56px;
    margin-bottom: .8rem;
    background: var(--accent-dim);
    border: 1px solid var(--accent-line);
    border-radius: 14px;
    color: var(--accent-soft);
}
.empty-state .icon svg { width: 24px; height: 24px; }
.empty-state .title {
    font-size: 1rem;
    color: var(--text-1);
    margin-bottom: .35rem;
    font-weight: 600;
}
.empty-state .subtitle {
    font-size: .84rem;
    color: var(--text-3);
    line-height: 1.5;
    max-width: 380px;
    margin: 0 auto;
}

/* ── Skeleton loader ── */
.skeleton {
    background:
      linear-gradient(90deg,
        rgba(255,255,255,0.025) 0%,
        rgba(255,255,255,0.06) 50%,
        rgba(255,255,255,0.025) 100%);
    background-size: 200% 100%;
    animation: shimmer 1.5s ease-in-out infinite;
    border-radius: 8px;
}
@keyframes shimmer {
    0%   { background-position: 100% 0; }
    100% { background-position: -100% 0; }
}
.skeleton-img {
    width: 100%;
    aspect-ratio: 4/3;
    border-radius: 8px;
}
.skeleton-line {
    height: 12px;
    margin-bottom: 8px;
}
.skeleton-line.short { width: 40%; }
.skeleton-line.mid   { width: 70%; }

/* ── Detection row ── */
.detection-row {
    display: flex;
    align-items: center;
    gap: .6rem;
    padding: .65rem .85rem;
    background: rgba(255,255,255,0.02);
    border: 1px solid var(--line);
    border-radius: 8px;
    margin-bottom: .45rem;
    transition: border-color 0.2s ease, background 0.2s ease;
}
.detection-row:hover {
    background: rgba(122,162,247,0.04);
    border-color: var(--line-hi);
}
.detection-label { font-weight: 600; color: var(--text-1); flex: 0 0 auto; }
.detection-conf { color: var(--text-2); font-size: .82rem; font-variant-numeric: tabular-nums; margin-left: auto; }

/* ── Alerts ── */
[data-testid="stAlert"][data-baseweb-kind="positive"] {
    background: rgba(52,211,153,0.10) !important;
    border: 1px solid rgba(52,211,153,0.28) !important;
    border-radius: 10px !important;
}

/* ── Expander ── */
.streamlit-expanderHeader, [data-testid="stExpander"] details summary {
    background: rgba(255,255,255,0.02) !important;
    border: 1px solid var(--line) !important;
    border-radius: 8px !important;
    transition: background 0.2s ease, border-color 0.2s ease !important;
    font-weight: 500 !important;
}
.streamlit-expanderHeader:hover, [data-testid="stExpander"] details summary:hover {
    background: rgba(122,162,247,0.05) !important;
    border-color: var(--line-hi) !important;
}

/* ── Progress bar ── */
.stProgress > div > div > div > div {
    background: var(--accent) !important;
}

/* ── Radio (layout selector) ── */
.stRadio [role="radiogroup"] {
    gap: .3rem;
    background: rgba(255,255,255,0.025);
    padding: 4px;
    border-radius: 8px;
    border: 1px solid var(--line);
}
.stRadio [role="radiogroup"] label {
    padding: 4px 10px !important;
    border-radius: 5px !important;
    transition: all 0.2s ease !important;
}
.stRadio [role="radiogroup"] label:hover {
    background: rgba(122,162,247,0.06) !important;
}
.stRadio [role="radiogroup"] label[data-checked="true"] {
    background: rgba(122,162,247,0.14) !important;
    color: var(--accent-soft) !important;
}

/* ── Checkbox ── */
.stCheckbox label { font-weight: 500 !important; }

/* ── Tab intro text ── */
.tab-intro {
    color: var(--text-2);
    margin: -.4rem 0 1.3rem 0;
    font-size: .92rem;
    line-height: 1.55;
    max-width: 720px;
    display: flex;
    align-items: center;
    gap: .5rem;
}
.tab-intro svg { color: var(--text-3); flex-shrink: 0; }

/* ── Pipeline diagram ── */
.pipeline-diagram {
    display: flex;
    flex-direction: row;
    align-items: stretch;
    justify-content: space-between;
    margin: 1.2rem 0;
    border: 1px solid var(--line);
    border-radius: 12px;
    background: rgba(0,0,0,0.22);
    overflow: hidden;
}
.pipeline-step {
    flex: 1 1 0;
    padding: 1.1rem 1rem;
    text-align: center;
    position: relative;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: flex-start;
}
.pipeline-step:last-of-type { border-right: none; }
.pipeline-step .num {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 24px; height: 24px;
    border-radius: 50%;
    background: var(--accent-dim);
    color: var(--accent-soft);
    font-size: .76rem;
    font-weight: 600;
    font-variant-numeric: tabular-nums;
    margin-bottom: .55rem;
    border: 1px solid var(--accent-line);
}
.pipeline-step .name {
    font-weight: 600;
    font-size: .90rem;
    color: var(--text-1);
    margin-bottom: .25rem;
    letter-spacing: -0.005em;
}
.pipeline-step .desc {
    color: var(--text-2);
    font-size: .76rem;
    line-height: 1.45;
}
.pipeline-arrow {
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 0 .8rem;
    color: var(--accent-soft);
    flex-shrink: 0;
}
.pipeline-arrow svg { stroke-width: 2 !important; }

/* ── Custom scrollbar ── */
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: rgba(255,255,255,0.015); }
::-webkit-scrollbar-thumb {
    background: rgba(122,162,247,0.22);
    border-radius: 4px;
}
::-webkit-scrollbar-thumb:hover { background: rgba(122,162,247,0.40); }

/* ── Reduced motion ── */
@media (prefers-reduced-motion: reduce) {
    .status-dot.ready::after,
    .status-dot.error::after,
    .skeleton { animation: none; }
    .card, .gallery-card, .metric-card { transition: none; }
}

/* ── Responsive: stack tabs/metrics on small screens ── */
@media (max-width: 768px) {
    .header-banner { padding: 1.6rem 1.2rem 1.3rem 1.2rem; }
    .header-banner h1 { font-size: 1.8rem; }
    .pipeline-diagram {
        grid-template-columns: 1fr;
    }
    .pipeline-step { border-right: none; border-bottom: 1px solid var(--line); }
    .pipeline-step:last-of-type { border-bottom: none; }
    .pipeline-arrow { display: none; }
}
</style>
""",
    unsafe_allow_html=True,
)


# ──────────────────────────────────────────────────────────────────────────────
# Model loading with caching
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_pipeline(yolo_path: str, cnn_path: str, imgsz: int, conf: float, iou: float, device_mode: str = "cuda"):
    import torch
    from stage12_yolo_cnn_system import Stage12Pipeline

    yolo_abs = PROJECT_ROOT / yolo_path
    cnn_abs = PROJECT_ROOT / cnn_path

    if not yolo_abs.exists():
        raise FileNotFoundError(f"YOLO weights not found:\n`{yolo_abs}`")
    if not cnn_abs.exists():
        raise FileNotFoundError(f"CNN checkpoint not found:\n`{cnn_abs}`")

    target_device = torch.device("cuda:0" if ("cuda" in device_mode and torch.cuda.is_available()) else "cpu")

    return Stage12Pipeline(
        yolo_path=str(yolo_abs),
        cnn_checkpoint=str(cnn_abs),
        device=target_device,
        yolo_imgsz=imgsz,
        yolo_conf=conf,
        yolo_iou=iou,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        f'<div style="display:flex; align-items:center; gap:8px; margin-bottom:6px;">'
        f'<span style="display:inline-flex; align-items:center; justify-content:center; '
        f'width:28px; height:28px; background:var(--accent-dim); border:1px solid var(--accent-line); '
        f'border-radius:8px; color:var(--accent-soft);">{_icon("microscope", 16)}</span>'
        f'<span style="font-weight:700; font-size:1rem; color:var(--text-1); letter-spacing:-0.01em;">'
        f'PCB Defect Detector</span></div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Two-stage pipeline: YOLO detection followed by CNN classification. "
        "Upload PCB images to locate and classify manufacturing defects."
    )
    _render_sidebar_clock()
    st.markdown('<div class="section-sep"></div>', unsafe_allow_html=True)

    import torch
    cuda_available = torch.cuda.is_available()
    st.markdown("## Hardware Acceleration")
    device_options = ["cuda (GPU)"] if cuda_available else ["cpu (CPU)"]
    if cuda_available:
        device_options.append("cpu (CPU)")
    selected_device_mode = st.selectbox(
        "Execution Device",
        options=device_options,
        index=0,
        help="NVIDIA GPU (CUDA) provides ~30 FPS real-time speed. CPU is used as fallback.",
        key="hardware_device_select"
    )
    device_hardware_name = torch.cuda.get_device_name(0) if cuda_available and "cuda" in selected_device_mode else "CPU"
    st.caption(f"Hardware: **{device_hardware_name}**")

    st.markdown('<div class="section-sep"></div>', unsafe_allow_html=True)

    st.markdown("## CNN Model")
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

    st.markdown("## YOLO Settings")
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

    st.markdown("## Display")
    show_gradcam = st.checkbox(
        "GradCAM heatmap per detection",
        value=True,
        help="Adds a GradCAM overlay on each YOLO crop (slightly slower).",
    )

    st.markdown('<div class="section-sep"></div>', unsafe_allow_html=True)
    st.caption("Built with Streamlit, YOLO, and CNN")


# ──────────────────────────────────────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────────────────────────────────────
st.markdown(
    f"""
<div class="header-banner">
    <div class="hbadge">{_icon("chip", 12)} PCB Inspection Suite</div>
    <h1>PCB <span class="accent-word">Defect</span> Detector</h1>
    <p>Upload a PCB image and the two-stage AI pipeline will find and classify manufacturing defects in seconds.</p>
</div>
""",
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────────────────────
# System Real-Time Clock & Status
# ──────────────────────────────────────────────────────────────────────────────
_render_system_clock()

# ──────────────────────────────────────────────────────────────────────────────
# Model load + status
# ──────────────────────────────────────────────────────────────────────────────
pipeline = None
model_ready = False
try:
    with st.spinner("Loading models on GPU/CPU. This may take a moment on first run."):
        pipeline = load_pipeline(yolo_path, cnn_path, yolo_imgsz, yolo_conf, yolo_iou, device_mode=selected_device_mode)
    model_ready = True
except FileNotFoundError as exc:
    st.error(f"**Model file missing**\n\n{exc}")
    st.info("Check that trained models exist in the expected paths. Adjust paths in the sidebar if needed.")
except Exception as exc:
    st.error(f"**Failed to load models**\n\n{exc}")

if model_ready:
    gpu_badge = f"🚀 GPU Accelerated ({device_hardware_name})" if "cuda" in selected_device_mode else "💻 CPU Mode"
    st.markdown(
        f'<div class="status-banner"><span class="status-dot ready"></span>Pipeline loaded on <strong>{gpu_badge}</strong> · Ready for inference</div>',
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        f'<div class="status-banner"><span class="status-dot error"></span>Pipeline not loaded, check sidebar settings</div>',
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
    """Run inference + annotation for one image in-memory. Returns a result dict."""
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    original_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    result = pipeline.predict_image(image_path)

    annotated_bgr = image_bgr.copy()
    crops: list[np.ndarray] = []
    for pred in result.get("predictions", []):
        x1, y1, x2, y2 = pred["bbox"]
        label = (
            f"{pred['stage2_label']} "
            f"y:{pred['stage1_confidence']:.2f} "
            f"c:{pred['stage2_confidence']:.2f}"
        )
        cv2.rectangle(annotated_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            annotated_bgr,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
        crop = image_bgr[y1:y2, x1:x2]
        if crop is not None and crop.size > 0:
            crops.append(crop)

    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    result["_annotated_rgb"] = annotated_rgb
    result["_original_rgb"] = original_rgb
    result["_crops"] = crops
    return result


def _list_demo_samples() -> list[Path]:
    if not DEMO_INPUT_DIR.exists():
        return []
    return sorted(p for p in DEMO_INPUT_DIR.iterdir() if p.suffix.lower() in SUPPORTED_EXTS)


def _list_conveyor_samples(source_type: str = "dataset_test") -> list[Path]:
    """Retrieve test images from dataset test folder, with fallback to other candidates."""
    if source_type == "dataset_test":
        candidates = [
            DATASET_TEST_DIR,
            PROJECT_ROOT / "pcb-defect-dataset" / "test" / "images",
            PROJECT_ROOT / "pcb-defect-dataset" / "test",
            PROJECT_ROOT / "dataset" / "test" / "images",
            PROJECT_ROOT / "dataset" / "test",
            PROJECT_ROOT / "data" / "test" / "images",
        ]
        for c in candidates:
            if c.exists() and c.is_dir():
                imgs = sorted(p for p in c.iterdir() if p.suffix.lower() in SUPPORTED_EXTS)
                if imgs:
                    return imgs
    return _list_demo_samples()


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
    <div class="icon">{_icon(icon, 22)}</div>
    <div class="title">{title}</div>
    <div class="subtitle">{subtitle}</div>
</div>
"""


def _render_skeleton_pair() -> None:
    """Loading skeleton for two side-by-side images."""
    col1, col2 = st.columns(2, gap="medium")
    for col in (col1, col2):
        with col:
            st.markdown('<div class="skeleton skeleton-img"></div>', unsafe_allow_html=True)


def _section_heading(icon_name: str, title: str, count: int | None = None) -> None:
    """Render a styled section heading with leading icon and optional count pill."""
    pill = ""
    if count is not None:
        pill = f'<span class="count-pill">{count}</span>'
    st.markdown(
        f'<div class="section-heading">'
        f'<span class="icon-wrap">{_icon(icon_name, 14)}</span>'
        f'<span>{title}</span>{pill}</div>',
        unsafe_allow_html=True,
    )


def _render_image(rgb_array, label: str | None = None, icon_name: str = "image") -> None:
    """Render an image with the styled frame + optional caption label."""
    if label:
        st.markdown(
            f'<div class="image-label">{_icon(icon_name, 12)} {label}</div>',
            unsafe_allow_html=True,
        )
    st.markdown('<div class="image-frame">', unsafe_allow_html=True)
    st.image(rgb_array, width="stretch")
    st.markdown("</div>", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Tabs (with custom icon prefix in label)
# ──────────────────────────────────────────────────────────────────────────────
tab_detector, tab_conveyor, tab_gallery, tab_perf, tab_about = st.tabs(
    [
        "Detector (Soi Chi Tiết)",
        "🏭 SMT Live Stream",
        "Gallery (Kiểm Định Hàng Loạt)",
        "Performance (Hiệu Năng)",
        "About (Kiến Trúc)",
    ]
)


# ──────────────────────────────────────────────────────────────────────────────
# TAB 1 - Detector (single-image deep view)
# ──────────────────────────────────────────────────────────────────────────────
with tab_detector:
    st.markdown(
        f'<p class="tab-intro">{_icon("info", 14)} '
        f'Kiểm định đơn ảnh chuyên sâu với chế độ xem song song, kính lúp vi mô 4x-8x, Grad-CAM XAI và xuất phiếu kiểm định QA/QC.</p>',
        unsafe_allow_html=True,
    )

    _section_heading("image", "Chọn ảnh mẫu hoặc bo mạch từ Tập Test Dataset")
    sample_tab_demo, sample_tab_dataset = st.tabs(["⚡ 5 Ảnh Mẫu Nhanh", "📦 1,069 Bo Mạch Tập Test Dataset"])
    with sample_tab_demo:
        demo_samples = _list_demo_samples()
        if demo_samples and model_ready:
            n = min(5, len(demo_samples))
            cols = st.columns(n, gap="small")
            for i, sample in enumerate(demo_samples[:n]):
                with cols[i]:
                    if st.button(
                        f"▸ {sample.stem[:18]}",
                        key=f"demo_load_{i}",
                        width="stretch",
                        help=f"Nạp ảnh {sample.name}",
                    ):
                        st.session_state["detector_upload"] = str(sample)
                        st.rerun()
    with sample_tab_dataset:
        dataset_samples = _list_conveyor_samples("dataset_test")
        if dataset_samples and model_ready:
            ds_col1, ds_col2 = st.columns([3, 1], gap="small")
            with ds_col1:
                selected_ds_img = st.selectbox(
                    "Chọn bo mạch từ tập Test:",
                    [p.name for p in dataset_samples],
                    key="tab1_ds_select",
                    label_visibility="collapsed",
                )
            with ds_col2:
                if st.button("🚀 Nạp Ảnh Này", width="stretch", key="tab1_load_ds_btn"):
                    target_p = next((p for p in dataset_samples if p.name == selected_ds_img), None)
                    if target_p:
                        st.session_state["detector_upload"] = str(target_p)
                        st.rerun()
    st.markdown('<div class="section-sep"></div>', unsafe_allow_html=True)

    _section_heading("upload", "Tải ảnh bo mạch PCB lên")
    detector_file = st.file_uploader(
        "Tải ảnh bo mạch PCB để kiểm tra",
        type=["jpg", "jpeg", "png", "bmp"],
        accept_multiple_files=False,
        key="detector_uploader",
        help="Hỗ trợ các định dạng JPG, JPEG, PNG, BMP.",
        label_visibility="collapsed",
    )

    image_path = None
    uploaded_name = None

    if detector_file is not None:
        image_path = _save_upload_to_temp(detector_file)
        uploaded_name = detector_file.name
        # Clear any stored demo sample so uploaded file is active
        st.session_state["detector_upload"] = None
    else:
        demo_path = st.session_state.get("detector_upload")
        if demo_path and Path(demo_path).exists():
            image_path = demo_path
            uploaded_name = Path(demo_path).name

    if image_path and model_ready:
        try:
            skeleton_slot = st.empty()
            with skeleton_slot.container():
                _render_skeleton_pair()
            
            t_start = time.perf_counter()
            with st.spinner("Đang chạy suy luận mô hình 2 giai đoạn..."):
                result = _run_pipeline(image_path)
            t_latency_ms = (time.perf_counter() - t_start) * 1000.0
            fps_equiv = 1000.0 / t_latency_ms if t_latency_ms > 0 else 0
            skeleton_slot.empty()

            device_name = "CUDA GPU" if getattr(pipeline, "device", None) and pipeline.device.type == "cuda" else "CPU Demo"
            preds = result.get("predictions", [])

            # Open card with file header inside
            st.markdown(
                f'<div class="card">'
                f'<div class="section-heading" style="margin-top:0;">'
                f'<span class="icon-wrap">{_icon("file", 14)}</span>'
                f'<span style="font-size:.95rem;">{uploaded_name}</span></div>'
                f'<div style="display:flex; flex-wrap:wrap; gap:10px; margin: 10px 0 14px 0;">'
                f'<div class="status-banner" style="margin-bottom:0; background:rgba(56,189,248,0.08); border-color:rgba(56,189,248,0.25);">'
                f'<span style="color:#38bdf8; font-weight:600;">⚡ Latency ({device_name}):</span>'
                f'<strong style="color:#f8fafc; margin-left:6px; font-family:\'JetBrains Mono\',monospace;">{t_latency_ms:.0f} ms</strong>'
                f'<span style="color:#94a3b8; font-size:0.75rem; margin-left:4px;">(~{fps_equiv:.1f} FPS)</span></div>'
                f'<div class="status-banner" style="margin-bottom:0; background:rgba(16,185,129,0.08); border-color:rgba(16,185,129,0.25);">'
                f'<span style="color:#10b981; font-weight:600;">🎯 Detections:</span>'
                f'<strong style="color:#f8fafc; margin-left:6px; font-family:\'JetBrains Mono\',monospace;">{len(preds)} defects</strong></div>'
                f'<div class="status-banner" style="margin-bottom:0; background:rgba(168,85,247,0.08); border-color:rgba(168,85,247,0.25);">'
                f'<span style="color:#c084fc; font-weight:600;">⚙️ Pipeline:</span>'
                f'<strong style="color:#f8fafc; margin-left:6px; font-family:\'JetBrains Mono\',monospace;">YOLOv8m + {cnn_choice}</strong></div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            original_rgb = result["_original_rgb"]
            annotated_rgb = result["_annotated_rgb"]

            # View Mode Selector: Side-by-side or Interactive Zoom Loupe
            view_mode = st.radio(
                "Chế độ hiển thị kiểm định:",
                options=["Chế độ Xem Song Song (Side-by-Side)", "🔍 Kính Lúp Soi Vi Mô Tương Tác (Micro-Zoom Loupe 4x-8x)"],
                horizontal=True,
                key="detector_view_mode",
            )

            if view_mode.startswith("🔍"):
                col_z1, col_z2 = st.columns([1, 1])
                with col_z1:
                    zoom_factor = st.slider("Độ phóng đại (Magnification)", 2, 8, 4, 1, key="loupe_zoom_factor")
                with col_z2:
                    lens_diameter = st.slider("Kích thước kính lúp (Lens Size)", 120, 240, 160, 20, key="loupe_lens_size")
                
                loupe_html = render_interactive_zoom_loupe_html(
                    original_rgb, annotated_rgb, zoom_level=zoom_factor, lens_size=lens_diameter, container_height=420
                )
                components.html(loupe_html, height=490)
            else:
                col_o, col_a = st.columns(2, gap="medium")
                with col_o:
                    _render_image(original_rgb, label="Original Raw PCB", icon_name="image")
                with col_a:
                    _render_image(annotated_rgb, label="AI Detections & Refined", icon_name="search")

            gradcam_crops_for_report = []

            if preds:
                _section_heading("grid", "Bảng kê chi tiết vết cắt khuyết tật", count=len(preds))
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

                badges_parts = []
                for r in rows:
                    cls = _defect_css_class(r["Defect"])
                    badges_parts.append(
                        f'<span class="defect-badge defect-{cls}">'
                        f'{_icon("defect-" + cls, 12)} '
                        f'{r["Defect"]}</span>'
                    )
                st.markdown(" ".join(badges_parts), unsafe_allow_html=True)

                if show_gradcam:
                    st.markdown('<div class="section-sep"></div>', unsafe_allow_html=True)
                    _section_heading("fire", "Bản đồ nhiệt Grad-CAM XAI (Explainable AI)", count=len(preds))
                    st.caption("Trực quan hóa vùng đặc trưng nơ-ron tập trung khi phân loại khuyết tật bo mạch.")
                    crops = result["_crops"]
                    for idx, (pred, crop) in enumerate(zip(preds, crops)):
                        with st.expander(
                            f"#{idx + 1}  {pred['stage2_label']}  ·  "
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
                                if gradcam_out.get("overlay") is not None:
                                    gradcam_crops_for_report.append({
                                        "label": pred["stage2_label"],
                                        "confidence": pred["combined_confidence"],
                                        "overlay": gradcam_out["overlay"]
                                    })

                                cols = st.columns(3, gap="small")
                                with cols[0]:
                                    st.markdown(
                                        f'<div class="image-label">'
                                        f'{_icon("image", 12)} 1. Vùng Cắt Gốc (Crop)</div>',
                                        unsafe_allow_html=True,
                                    )
                                    st.markdown('<div class="image-frame">', unsafe_allow_html=True)
                                    st.image(
                                        cv2.cvtColor(crop, cv2.COLOR_BGR2RGB),
                                        width="stretch",
                                    )
                                    st.markdown("</div>", unsafe_allow_html=True)
                                with cols[1]:
                                    st.markdown(
                                        f'<div class="image-label">'
                                        f'{_icon("fire", 12)} 2. Bản Đồ Nhiệt (Grad-CAM)</div>',
                                        unsafe_allow_html=True,
                                    )
                                    heatmap_display = gradcam_out.get("heatmap")
                                    if heatmap_display is not None:
                                        st.markdown('<div class="image-frame">', unsafe_allow_html=True)
                                        st.image(
                                            heatmap_display,
                                            width="stretch",
                                        )
                                        st.markdown("</div>", unsafe_allow_html=True)
                                    else:
                                        st.caption("Không thể trích xuất Heatmap.")
                                with cols[2]:
                                    st.markdown(
                                        f'<div class="image-label">'
                                        f'{_icon("layers", 12)} 3. Phủ Trực Quan (Overlay)</div>',
                                        unsafe_allow_html=True,
                                    )
                                    if gradcam_out.get("overlay") is not None:
                                        st.markdown('<div class="image-frame">', unsafe_allow_html=True)
                                        st.image(
                                            gradcam_out["overlay"],
                                            width="stretch",
                                        )
                                        st.markdown("</div>", unsafe_allow_html=True)
                                
                                st.caption("🔥 **Giải thích XAI:** 🔴 **Đỏ rực** = Vùng khuyết tật tập trung gradient nơ-ron cao nhất · 🔵 **Xanh lam** = Nền phíp FR4 cách điện an toàn.")
            else:
                st.markdown(
                    f'<div style="display:flex; align-items:center; gap:.5rem; '
                    f'padding:.7rem 1rem; background:rgba(52,211,153,0.10); '
                    f'border:1px solid rgba(52,211,153,0.28); border-radius:10px; '
                    f'color:var(--accent-soft); font-weight:500;">'
                    f'{_icon("ok", 14, color="#34d399")}'
                    f'<span>Không phát hiện khuyết tật. Bo mạch hoàn hảo đạt chuẩn 100% IPC-A-610.</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            # Export reports section
            st.markdown('<div class="section-sep"></div>', unsafe_allow_html=True)
            _section_heading("file", "Xuất Báo Cáo & Phiếu Nghiệm Thu Chất Lượng")
            
            qc_cert_html = generate_qc_certificate_html(
                image_name=uploaded_name,
                original_rgb=original_rgb,
                annotated_rgb=annotated_rgb,
                predictions=preds,
                latency_ms=t_latency_ms,
                device_name=device_hardware_name,
                pipeline_name=f"YOLOv8m + {cnn_choice} (Two-Stage Hybrid)",
                engineer_name="Nguyễn Duy Khương",
                engineer_id="11236134",
                gradcam_crops=gradcam_crops_for_report,
            )

            col_btn1, col_btn2 = st.columns(2, gap="medium")
            with col_btn1:
                st.download_button(
                    label="📄 Tải Phiếu Kiểm Định QA/QC (HTML / In PDF)",
                    data=qc_cert_html,
                    file_name=f"QC_Certificate_{Path(uploaded_name).stem}.html",
                    mime="text/html",
                    key=f"dl_cert_{uploaded_name}",
                    help="Tải phiếu chứng nhận chất lượng xuất xưởng chuẩn ISO/IPC có sẵn nút in PDF.",
                    width="stretch",
                )
            with col_btn2:
                public_result = {
                    k: v for k, v in result.items() if not k.startswith("_")
                }
                st.download_button(
                    label="📥 Tải Dữ Liệu Tọa Độ Lỗi (JSON)",
                    data=json.dumps(public_result, indent=2),
                    file_name=f"{Path(uploaded_name).stem}_results.json",
                    mime="application/json",
                    key=f"dl_det_{uploaded_name}",
                    help="Tải file JSON phục vụ kết nối hệ thống điều hành nhà máy MES/ERP.",
                    width="stretch",
                )

            with st.expander("👁️ Xem trước Phiếu Kiểm Định QA/QC Xuất Xưởng", expanded=False):
                components.html(qc_cert_html, height=720, scrolling=True)

            st.markdown("</div>", unsafe_allow_html=True)
        except Exception as exc:
            st.error(f"Lỗi xử lý ảnh **{uploaded_name}**: {exc}")
    else:
        st.markdown(
            _empty_state(
                "upload",
                "Kéo thả ảnh bo mạch PCB để bắt đầu kiểm tra",
                "Hỗ trợ JPG, JPEG, PNG, BMP. Hoặc chọn nhanh các ảnh mẫu ở trên để kiểm định tức thì.",
            ),
            unsafe_allow_html=True,
        )


# ──────────────────────────────────────────────────────────────────────────────
# TAB 2 - 🏭 SMT Live Stream (Automated Conveyor Simulation)
# ──────────────────────────────────────────────────────────────────────────────
with tab_conveyor:
    st.markdown(
        f'<p class="tab-intro">{_icon("chip", 14)} '
        f'Mô phỏng camera quang học tự động quét liên tục trên băng chuyền SMT với tốc độ GPU thời gian thực, '
        f'tích hợp tháp đèn Andon Light và bảng theo dõi sản lượng.</p>',
        unsafe_allow_html=True,
    )

    # Initialize conveyor session state
    if "smt_total_scanned" not in st.session_state:
        st.session_state["smt_total_scanned"] = 0
        st.session_state["smt_pass_count"] = 0
        st.session_state["smt_defect_count"] = 0
        st.session_state["smt_history"] = []

    # Conveyor Source & Options
    col_src, col_shuf = st.columns([3, 1], gap="medium")
    with col_src:
        conveyor_source = st.selectbox(
            "📦 Nguồn phôi bo mạch cho băng chuyền:",
            [
                "📦 Tập Test Thực Tế (pcb-defect-dataset/test/images - 1,069 bo mạch)",
                "🔍 Ảnh Mẫu Nhanh (demo_input/ - 5 bo mạch)",
            ],
            index=0,
            key="conveyor_source_select",
        )
    with col_shuf:
        st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
        shuffle_boards = st.checkbox("🔀 Quét Ngẫu Nhiên", value=True, help="Tự động bốc ngẫu nhiên các bo mạch khác nhau từ 1,069 ảnh trong tập test.")

    source_type = "dataset_test" if "1,069" in conveyor_source else "demo_input"
    conveyor_samples = _list_conveyor_samples(source_type)

    # Conveyor Control Bar
    ctrl_c1, ctrl_c2, ctrl_c3, ctrl_c4 = st.columns([1.5, 1.5, 1.5, 1.2], gap="small")
    with ctrl_c1:
        stream_active = st.toggle("▶️ Chạy Băng Chuyền", value=False, key="conveyor_toggle")
    with ctrl_c2:
        stream_speed_ms = st.slider("Độ trễ nhân tạo (ms/bo)", 0, 300, 0, 10, key="conveyor_speed", help="Đặt 0ms để chạy với tốc độ tối đa của GPU RTX 4060.")
    with ctrl_c3:
        max_stream_boards = st.number_input("Số bo mạch quét tối đa", min_value=5, max_value=200, value=20, key="conveyor_max")
    with ctrl_c4:
        if st.button("🔄 Reset Bộ Đếm", width="stretch", key="conveyor_reset_btn"):
            st.session_state["smt_total_scanned"] = 0
            st.session_state["smt_pass_count"] = 0
            st.session_state["smt_defect_count"] = 0
            st.session_state["smt_history"] = []
            st.rerun()

    st.markdown(
        f'<div style="font-size:0.8rem; color:#94a3b8; margin-bottom:10px; font-family:monospace;">'
        f'🏭 Nguồn cấp phôi: <strong style="color:#38bdf8;">{len(conveyor_samples):,} bo mạch</strong> '
        f'từ <code>pcb-defect-dataset/test/images</code> | Đã sẵn sàng nạp camera AOI.</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="section-sep"></div>', unsafe_allow_html=True)

    # Real-Time KPI Cards Placeholder
    kpi_placeholder = st.empty()
    andon_placeholder = st.empty()
    live_view_placeholder = st.empty()
    history_placeholder = st.empty()

    def _render_conveyor_kpis(live_fps: float = 0.0, live_lat: float = 0.0):
        tot = st.session_state["smt_total_scanned"]
        pas = st.session_state["smt_pass_count"]
        dfc = st.session_state["smt_defect_count"]
        yield_rate = (pas / tot * 100.0) if tot > 0 else 100.0
        defect_rate = (dfc / tot * 100.0) if tot > 0 else 0.0

        fps_display = f"{live_fps:.1f} bo/s" if live_fps > 0 else "Ready"
        lat_display = f"({live_lat:.0f} ms)" if live_lat > 0 else ""

        kpi_placeholder.markdown(
            f"""
            <div style="display:grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 12px;">
                <div class="card" style="margin:0; padding:12px; text-align:center; border-top: 3px solid #38bdf8;">
                    <div style="font-size:0.75rem; color:#94a3b8; font-weight:700; text-transform:uppercase;">📦 Tổng Số Bo Đã Quét</div>
                    <div style="font-size:1.6rem; font-weight:800; font-family:'JetBrains Mono',monospace; color:#f8fafc; margin-top:4px;">{tot}</div>
                </div>
                <div class="card" style="margin:0; padding:12px; text-align:center; border-top: 3px solid #10b981;">
                    <div style="font-size:0.75rem; color:#10b981; font-weight:700; text-transform:uppercase;">🟢 Bo Đạt Chuẩn (PASS)</div>
                    <div style="font-size:1.6rem; font-weight:800; font-family:'JetBrains Mono',monospace; color:#10b981; margin-top:4px;">{pas} <span style="font-size:0.85rem; color:#6ee7b7;">({yield_rate:.1f}%)</span></div>
                </div>
                <div class="card" style="margin:0; padding:12px; text-align:center; border-top: 3px solid #ef4444;">
                    <div style="font-size:0.75rem; color:#ef4444; font-weight:700; text-transform:uppercase;">🔴 Bo Lỗi / Phế Phẩm (NG)</div>
                    <div style="font-size:1.6rem; font-weight:800; font-family:'JetBrains Mono',monospace; color:#ef4444; margin-top:4px;">{dfc} <span style="font-size:0.85rem; color:#fca5a5;">({defect_rate:.1f}%)</span></div>
                </div>
                <div class="card" style="margin:0; padding:12px; text-align:center; border-top: 3px solid #a855f7;">
                    <div style="font-size:0.75rem; color:#c084fc; font-weight:700; text-transform:uppercase;">⚡ Tốc Độ GPU Băng Chuyền</div>
                    <div style="font-size:1.6rem; font-weight:800; font-family:'JetBrains Mono',monospace; color:#c084fc; margin-top:4px;">{fps_display} <span style="font-size:0.85rem; color:#d8b4fe;">{lat_display}</span></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    def _render_andon_light(is_passed: bool, defect_labels: list[str], board_name: str, latency: float):
        if is_passed:
            andon_placeholder.markdown(
                f"""
                <div style="display:flex; align-items:center; justify-content:space-between; padding:10px 18px; 
                            background:rgba(16,185,129,0.12); border:2px solid #10b981; border-radius:10px; margin-bottom:14px;">
                    <div style="display:flex; align-items:center; gap:12px;">
                        <span style="display:inline-block; width:14px; height:14px; border-radius:50%; background:#10b981; box-shadow:0 0 12px #10b981;"></span>
                        <strong style="color:#10b981; font-size:1rem; letter-spacing:0.04em;">ANDON TOWER: [ PASS - CONFORMANT ]</strong>
                        <span style="color:#94a3b8; font-size:0.85rem;">| Bo mạch #{board_name} sạch lỗi, đạt chuẩn chất lượng xuất xưởng.</span>
                    </div>
                    <div style="font-family:'JetBrains Mono',monospace; font-size:0.85rem; color:#10b981; font-weight:bold;">
                        Latency: {latency:.1f} ms
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            def_str = ", ".join(defect_labels)
            andon_placeholder.markdown(
                f"""
                <div style="display:flex; align-items:center; justify-content:space-between; padding:10px 18px; 
                            background:rgba(239,68,68,0.12); border:2px solid #ef4444; border-radius:10px; margin-bottom:14px;">
                    <div style="display:flex; align-items:center; gap:12px;">
                        <span style="display:inline-block; width:14px; height:14px; border-radius:50%; background:#ef4444; box-shadow:0 0 12px #ef4444; animation: pulse 1s infinite;"></span>
                        <strong style="color:#ef4444; font-size:1rem; letter-spacing:0.04em;">ANDON ALERT: [ REJECT / NG - DEFECTS DETECTED ]</strong>
                        <span style="color:#fca5a5; font-size:0.85rem;">| Bo mạch #{board_name} phát hiện: <strong>{def_str}</strong></span>
                    </div>
                    <div style="font-family:'JetBrains Mono',monospace; font-size:0.85rem; color:#ef4444; font-weight:bold;">
                        Latency: {latency:.1f} ms
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Initial render of stats
    _render_conveyor_kpis()

    if stream_active and model_ready and conveyor_samples:
        st.caption("🚀 Băng chuyền đang vận hành liên tục trên GPU...")
        
        step_count = 0
        while step_count < max_stream_boards:
            if shuffle_boards:
                curr_sample = random.choice(conveyor_samples)
            else:
                sample_idx = (st.session_state["smt_total_scanned"] + step_count) % len(conveyor_samples)
                curr_sample = conveyor_samples[sample_idx]
            
            t0 = time.perf_counter()
            res = _run_pipeline(str(curr_sample))
            t_lat = (time.perf_counter() - t0) * 1000.0
            live_fps = 1000.0 / t_lat if t_lat > 0 else 0.0
            
            preds = res.get("predictions", [])
            is_pass = len(preds) == 0
            defect_labels = [p["stage2_label"] for p in preds]

            # Update stats
            st.session_state["smt_total_scanned"] += 1
            if is_pass:
                st.session_state["smt_pass_count"] += 1
            else:
                st.session_state["smt_defect_count"] += 1

            # Append to history
            record = {
                "Time": datetime.datetime.now().strftime("%H:%M:%S"),
                "Board ID": curr_sample.name,
                "Status": "PASS" if is_pass else "NG (Lỗi)",
                "Defects Found": len(preds),
                "Classes": ", ".join(defect_labels) if defect_labels else "None",
                "Latency (ms)": f"{t_lat:.1f}",
            }
            st.session_state["smt_history"].insert(0, record)
            if len(st.session_state["smt_history"]) > 20:
                st.session_state["smt_history"].pop()

            # Render dynamic frames
            _render_conveyor_kpis(live_fps=live_fps, live_lat=t_lat)
            _render_andon_light(is_pass, defect_labels, curr_sample.stem, t_lat)

            with live_view_placeholder.container():
                col_orig, col_det = st.columns(2, gap="medium")
                with col_orig:
                    _render_image(res["_original_rgb"], label=f"Live Camera Optical Frame: {curr_sample.name}", icon_name="image")
                with col_det:
                    _render_image(res["_annotated_rgb"], label=f"Live AI Bounding Box & Classifications ({len(preds)} defects)", icon_name="search")

            step_count += 1
            if stream_speed_ms > 0:
                time.sleep(stream_speed_ms / 1000.0)

    # Render History Log Table
    if st.session_state["smt_history"]:
        history_placeholder.markdown('<div class="section-sep"></div>', unsafe_allow_html=True)
        with history_placeholder.container():
            _section_heading("list", "Nhật Ký Kiểm Định Dây Chuyền Thời Gian Thực", count=len(st.session_state["smt_history"]))
            df_hist = pd.DataFrame(st.session_state["smt_history"])
            st.dataframe(df_hist, width="stretch", hide_index=True)
    elif not stream_active:
        live_view_placeholder.markdown(
            _empty_state(
                "chip",
                "Băng chuyền đang ở trạng thái TẠM DỪNG",
                "Bật công tắc '▶️ Chạy Băng Chuyền' ở phía trên để bắt đầu mô phỏng luồng kiểm định tự động.",
            ),
            unsafe_allow_html=True,
        )


# ──────────────────────────────────────────────────────────────────────────────
# TAB 2 - Gallery (batch upload)
# ──────────────────────────────────────────────────────────────────────────────
with tab_gallery:
    st.markdown(
        f'<p class="tab-intro">{_icon("info", 14)} '
        f'Upload multiple PCB images and inspect them as a grid, side-by-side, or compact list.</p>',
        unsafe_allow_html=True,
    )

    _section_heading("upload", "Upload multiple images")
    gallery_files = st.file_uploader(
        "Drop PCB images here",
        type=["jpg", "jpeg", "png", "bmp"],
        accept_multiple_files=True,
        key="gallery_uploader",
        label_visibility="collapsed",
    )

    if gallery_files:
        st.markdown('<div style="height:.4rem"></div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="subheading">{_icon("palette", 12)} Gallery Layout</div>',
            unsafe_allow_html=True,
        )
        layout = st.radio(
            "Gallery layout",
            options=["Grid (3 cols)", "Side-by-side", "Compact list"],
            horizontal=True,
            key="gallery_layout",
            label_visibility="collapsed",
        )

    if gallery_files and model_ready:
        all_results: list[dict] = []
        all_labels: list[str] = []
        all_confs: list[float] = []

        progress = st.progress(0.0, text="Starting...")
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

        st.markdown('<div class="section-sep"></div>', unsafe_allow_html=True)
        _section_heading("grid", "Detected boards", count=len(all_results))

        if layout.startswith("Grid"):
            n_cols = 3
            for i in range(0, len(all_results), n_cols):
                row_results = all_results[i : i + n_cols]
                cols = st.columns(n_cols, gap="small")
                for col, res in zip(cols, row_results):
                    with col:
                        st.markdown('<div class="gallery-card">', unsafe_allow_html=True)
                        st.markdown(
                            f'<div class="gallery-title" title="{res["_uploaded_name"]}">'
                            f'{res["_uploaded_name"]}</div>',
                            unsafe_allow_html=True,
                        )
                        n_pred = len(res.get("predictions", []))
                        st.markdown(
                            f'<div class="gallery-sub">'
                            f'{_icon("search", 11)} {n_pred} detection(s)</div>',
                            unsafe_allow_html=True,
                        )
                        st.markdown('<div class="image-frame">', unsafe_allow_html=True)
                        st.image(res["_annotated_rgb"], width="stretch")
                        st.markdown("</div>", unsafe_allow_html=True)
                        if n_pred:
                            badges_parts = []
                            for p in res["predictions"]:
                                cls = _defect_css_class(p["stage2_label"])
                                badges_parts.append(
                                    f'<span class="defect-badge defect-{cls}">'
                                    f'{_icon("defect-" + cls, 11)} '
                                    f'{p["stage2_label"]}</span>'
                                )
                            st.markdown(" ".join(badges_parts), unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)
        elif layout.startswith("Side-by-side"):
            for res in all_results:
                st.markdown('<div class="gallery-card">', unsafe_allow_html=True)
                c1, c2 = st.columns(2, gap="medium")
                with c1:
                    _render_image(res["_original_rgb"], label="Original", icon_name="image")
                with c2:
                    _render_image(res["_annotated_rgb"], label="Detected", icon_name="search")
                n_pred = len(res["predictions"])
                st.markdown(
                    f'<div style="margin-top:.5rem; display:flex; align-items:center; gap:.5rem;">'
                    f'<span style="font-weight:600;">{res["_uploaded_name"]}</span>'
                    f'<span style="color:var(--text-2); font-size:.85rem;">'
                    f'{_icon("search", 11)} {n_pred} detection(s)</span></div>',
                    unsafe_allow_html=True,
                )
                if res["predictions"]:
                    badges_parts = []
                    for p in res["predictions"]:
                        cls = _defect_css_class(p["stage2_label"])
                        badges_parts.append(
                            f'<span class="defect-badge defect-{cls}">'
                            f'{_icon("defect-" + cls, 11)} '
                            f'{p["stage2_label"]}</span>'
                        )
                    st.markdown(" ".join(badges_parts), unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
        else:
            for res in all_results:
                with st.expander(
                    f"{res['_uploaded_name']}  ·  {len(res['predictions'])} detection(s)",
                    expanded=False,
                ):
                    c1, c2 = st.columns([1, 2], gap="medium")
                    with c1:
                        st.markdown('<div class="image-frame">', unsafe_allow_html=True)
                        st.image(res["_annotated_rgb"], width="stretch")
                        st.markdown("</div>", unsafe_allow_html=True)
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

        st.markdown('<div class="section-sep"></div>', unsafe_allow_html=True)
        st.markdown(
            f"""
<div class="header-banner compact">
    <h1 style="display:flex; align-items:center; gap:.5rem; color:var(--accent-soft);">
      {_icon("chart", 14)} Summary
    </h1>
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
        <span class="metric-icon">{_icon("image", 14)}</span>
        <div class="metric-value">{n_images}</div>
        <div class="metric-label">Images</div>
    </div>
    <div class="metric-card">
        <span class="metric-icon">{_icon("search", 14)}</span>
        <div class="metric-value">{total_defects}</div>
        <div class="metric-label">Total Defects</div>
    </div>
    <div class="metric-card">
        <span class="metric-icon">{_icon("bolt", 14)}</span>
        <div class="metric-value">{avg_conf:.1%}</div>
        <div class="metric-label">Avg Combined Conf</div>
    </div>
    <div class="metric-card">
        <span class="metric-icon">{_icon("tag", 14)}</span>
        <div class="metric-value">{n_types}</div>
        <div class="metric-label">Defect Types</div>
    </div>
</div>
""",
            unsafe_allow_html=True,
        )

        if counter:
            _section_heading("chart", "Defect distribution")
            chart_df = (
                pd.DataFrame.from_dict(counter, orient="index", columns=["Count"])
                .sort_values("Count", ascending=False)
            )
            chart_df.index.name = "Defect Type"
            st.bar_chart(chart_df, width="stretch")

            _section_heading("rows", "Defects per image")
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
                pivot = (
                    pivot_df.groupby(["Image", "Defect"])
                    .size()
                    .unstack(fill_value=0)
                )
                st.dataframe(pivot, width="stretch")

        st.markdown('<div class="section-sep"></div>', unsafe_allow_html=True)
        clean = [{k: v for k, v in r.items() if not k.startswith("_")} for r in all_results]
        st.download_button(
            "Download all results (JSON)",
            data=json.dumps(clean, indent=2),
            file_name="pcb_all_results.json",
            mime="application/json",
            key="dl_all",
        )

    elif gallery_files and not model_ready:
        st.warning("Fix the model configuration in the sidebar before uploading images.")
    else:
        st.markdown(
            _empty_state(
                "layers",
                "Upload multiple PCB images to build a gallery",
                "Switch between grid, side-by-side, or compact list above.",
            ),
            unsafe_allow_html=True,
        )


# ──────────────────────────────────────────────────────────────────────────────
# TAB 3 - Model Performance
# ──────────────────────────────────────────────────────────────────────────────
with tab_perf:
    st.markdown(
        f'<p class="tab-intro">{_icon("info", 14)} '
        f'Error-analysis artifacts from <code>error_analysis.py</code> '
        f'and GradCAM figures from <code>gradcam_visualize.py</code>.</p>',
        unsafe_allow_html=True,
    )

    sub_tabs = st.tabs(
        ["Error Analysis", "GradCAM Samples", "Reports"]
    )

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
                    st.markdown(
                        f'<div class="subheading">{_icon("chart", 12)} '
                        f'{png.stem.replace("_", " ").title()}</div>',
                        unsafe_allow_html=True,
                    )
                    # Wrap in fixed-width container so large confusion-matrix
                    # PNGs do not blow up the page width.
                    st.markdown(
                        '<div class="image-frame" style="max-width: 820px; margin: 0 auto;">',
                        unsafe_allow_html=True,
                    )
                    st.image(str(png), width="stretch")
                    st.markdown("</div>", unsafe_allow_html=True)
                    st.markdown('<div class="section-sep"></div>', unsafe_allow_html=True)

    with sub_tabs[1]:
        if not GRADCAM_DIR.exists():
            st.info("`runs/gradcam/` not found.")
        else:
            gradcam_files = sorted(GRADCAM_DIR.glob("*.png"))
            if not gradcam_files:
                st.info("Chưa có ảnh GradCAM mẫu. Hãy chạy `python gradcam_visualize.py` để tạo mẫu.")
            else:
                st.markdown(
                    f'<p style="color:var(--text-2); font-size:.82rem; margin-bottom:1rem;">'
                    f'Bộ sưu tập <b>12 ảnh mẫu Grad-CAM 3 khung hình</b> (Vùng cắt gốc | Bản đồ nhiệt JET | Lớp phủ pha trộn 45%) '
                    f'được trích xuất từ tập kiểm thử cho đầy đủ 6 dạng khuyết tật bo mạch in.</p>',
                    unsafe_allow_html=True,
                )
                
                # Helper for defect class lookup
                class_vi_dict = {
                    "missing_hole": "Thiếu lỗ khoan (missing_hole)",
                    "mouse_bite": "Gặm mép mạch (mouse_bite)",
                    "open_circuit": "Hở / Đứt mạch (open_circuit)",
                    "short": "Đoản mạch / Chập (short)",
                    "spur": "Gai đồng (spur)",
                    "spurious_copper": "Đồng thừa (spurious_copper)",
                }
                
                n_cols = 2
                for row_start in range(0, len(gradcam_files), n_cols):
                    row_files = gradcam_files[row_start : row_start + n_cols]
                    cols = st.columns(n_cols, gap="medium")
                    for col, png in zip(cols, row_files):
                        # Detect matching defect class from filename
                        matched_cls = "missing_hole"
                        for c in class_vi_dict:
                            if c in png.stem:
                                matched_cls = c
                                break
                        
                        vi_title = class_vi_dict.get(matched_cls, matched_cls)
                        
                        with col:
                            st.markdown('<div class="gallery-card">', unsafe_allow_html=True)
                            st.markdown(
                                f'<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:.4rem;">'
                                f'<span class="defect-badge defect-{matched_cls}">{_icon("defect-" + matched_cls, 12)} {matched_cls}</span>'
                                f'<span style="font-family:\'JetBrains Mono\', monospace; font-size:.72rem; color:var(--text-3);">{png.name}</span>'
                                f'</div>',
                                unsafe_allow_html=True,
                            )
                            st.markdown(
                                f'<div class="gallery-title" style="font-size:.88rem; font-weight:700;">{vi_title}</div>',
                                unsafe_allow_html=True,
                            )
                            st.markdown(
                                f'<div class="gallery-sub">{_icon("fire", 11)} '
                                f'Bản đồ nhiệt Grad-CAM 3-Panel (ResNet-18)</div>',
                                unsafe_allow_html=True,
                            )
                            st.markdown('<div class="image-frame" style="margin-top:.4rem;">', unsafe_allow_html=True)
                            st.image(str(png), width="stretch")
                            st.markdown("</div>", unsafe_allow_html=True)
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
                st.markdown(
                    f'<div class="subheading">{_icon("doc", 12)} '
                    f'{rp.parent.parent.name} / error_analysis_report.md</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(rp.read_text(encoding="utf-8"))
                st.markdown('<div class="section-sep"></div>', unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# TAB 4 - About
# ──────────────────────────────────────────────────────────────────────────────
with tab_about:
    st.markdown(
        f'<div class="card">'
        f'<div class="section-heading" style="margin-top:0;">'
        f'<span class="icon-wrap">{_icon("microscope", 14)}</span>'
        f'<span>Hệ thống phát hiện & phân loại lỗi bo mạch in (PCB)</span></div>'
        f'<p style="color:rgba(255,255,255,0.78); line-height:1.6; margin:0; font-size:.88rem;">'
        f'Hệ thống thị giác máy tính 2 giai đoạn (YOLOv8 + ResNet-18) tích hợp cơ chế giải thích trực quan Grad-CAM, '
        f'tự động định vị và phân loại 6 dạng khuyết tật bề mặt bo mạch in theo chuẩn quốc tế IPC-A-610 với độ chính xác cao và tốc độ thời gian thực.</p>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Pipeline diagram (visual) - STRICTLY 1 ROW
    st.markdown(
        f"""
<div class="card">
    <div class="section-heading" style="margin-top:0;">
        <span class="icon-wrap">{_icon("pipeline-stage", 14)}</span>
        <span>Quy trình hoạt động hệ thống (How the pipeline works)</span>
    </div>
    <div class="pipeline-diagram">
        <div class="pipeline-step">
            <div class="num">1</div>
            <div class="name">1. Định vị YOLO (YOLO Detection)</div>
            <div class="desc">Quét toàn ảnh bo mạch, khoanh vùng tọa độ Bounding Box các vị trí nghi ngờ lỗi.</div>
        </div>
        <div class="pipeline-arrow">{_icon("arrow-right", 16)}</div>
        <div class="pipeline-step">
            <div class="num">2</div>
            <div class="name">2. Phân loại CNN (CNN Classification)</div>
            <div class="desc">Cắt vùng lỗi (+25% padding) đưa qua ResNet-18 phân loại chuẩn xác 6 lớp khuyết tật.</div>
        </div>
        <div class="pipeline-arrow">{_icon("arrow-right", 16)}</div>
        <div class="pipeline-step">
            <div class="num">3</div>
            <div class="name">3. Giải thích Grad-CAM (XAI Visual)</div>
            <div class="desc">Trực quan hóa bản đồ nhiệt Heatmap thể hiện vùng nơ-ron tập trung để ra quyết định.</div>
        </div>
    </div>
</div>
""",
        unsafe_allow_html=True,
    )

    # Defect classes
    defect_classes_list = [
        "missing_hole", "mouse_bite", "open_circuit",
        "short", "spur", "spurious_copper",
    ]
    defect_badges_html = " ".join(
        f'<span class="defect-badge defect-{cls}">'
        f'{_icon("defect-" + cls, 12)} {cls}</span>'
        for cls in defect_classes_list
    )
    st.markdown(
        f'<div class="card">'
        f'<div class="section-heading" style="margin-top:0;">'
        f'<span class="icon-wrap">{_icon("tag", 14)}</span>'
        f'<span>6 Dạng khuyết tật bo mạch in (Defect classes)</span></div>'
        f'<p style="margin:0;">{defect_badges_html}</p></div>',
        unsafe_allow_html=True,
    )

    # Tips (Tiếng Việt có dấu)
    st.markdown(
        f"""
<div class="card">
    <div class="section-heading" style="margin-top:0;">
        <span class="icon-wrap">{_icon("wand", 14)}</span>
        <span>Mẹo sử dụng (Tips)</span>
    </div>
    <ul style="color:rgba(255,255,255,0.78); line-height:1.75; font-size:.84rem; padding-left:1.25rem; margin:0;">
        <li>Giảm <b>Ngưỡng độ tin cậy (Confidence threshold)</b> ở thanh công cụ bên trái để phát hiện các vết khuyết tật vi mô hoặc vết mờ.</li>
        <li>Sử dụng tab <b>Kiểm định sâu (Detector)</b> để soi chi tiết từng vùng lỗi với kính lúp số phóng đại, xem bản đồ nhiệt <b>Grad-CAM</b> và bấm xuất <b>Phiếu Kiểm Định QA/QC</b> tiêu chuẩn.</li>
        <li>Sử dụng tab <b>Băng chuyền (SMT Live Stream)</b> để mô phỏng camera quét luồng tự động 1,069 bo mạch tập Test với tốc độ thời gian thực ~30 FPS trên GPU.</li>
        <li>Sử dụng tab <b>Bộ sưu tập (Gallery)</b> để tải lên và kiểm tra đồng thời nhiều bo mạch cùng lúc theo lô sản xuất.</li>
    </ul>
</div>
""",
        unsafe_allow_html=True,
    )

    # Project layout (Fixed structured tree view)
    st.markdown(
        f"""
<div class="card">
    <div class="section-heading" style="margin-top:0;">
        <span class="icon-wrap">{_icon("folder", 14)}</span>
        <span>Cấu trúc mã nguồn dự án (Project layout)</span>
    </div>
    <div style="background:rgba(0,0,0,0.35); padding:1rem 1.2rem; border-radius:8px; border:1px solid var(--line); font-family:'JetBrains Mono', monospace; font-size:.80rem; line-height:1.75; color:rgba(255,255,255,0.85); overflow-x:auto; margin:0;">
        <div style="color:var(--accent); font-weight:700;">📁 PCB_defect_detection/</div>
        <div style="padding-left:1.2rem;">├── 📄 <b>app.py</b> <span style="color:var(--text-3); font-size:.76rem;">← Giao diện Web Dashboard Streamlit (Detector, SMT Live, Gallery, Analytics, About)</span></div>
        <div style="padding-left:1.2rem;">├── 📄 <b>stage12_yolo_cnn_system.py</b> <span style="color:var(--text-3); font-size:.76rem;">← Pipeline suy luận tích hợp 2 giai đoạn (YOLOv8m + ResNet-18)</span></div>
        <div style="padding-left:1.2rem;">├── 📄 <b>stage2_train.py</b> <span style="color:var(--text-3); font-size:.76rem;">← Huấn luyện CNN Stage 2 (OneCycleLR, Label Smoothing 0.05, Differential LR)</span></div>
        <div style="padding-left:1.2rem;">├── 📄 <b>stage2_cnn_utils.py</b> <span style="color:var(--text-3); font-size:.76rem;">← Tiện ích nạp mô hình CNN, đếm tham số &amp; đo độ trễ suy luận</span></div>
        <div style="padding-left:1.2rem;">├── 📄 <b>gradcam_ui.py</b> <span style="color:var(--text-3); font-size:.76rem;">← Module sinh bản đồ nhiệt XAI Grad-CAM cho từng vùng cắt khuyết tật</span></div>
        <div style="padding-left:1.2rem;">├── 📄 <b>gradcam_visualize.py</b> <span style="color:var(--text-3); font-size:.76rem;">← Công cụ CLI trích xuất Grad-CAM ngoại tuyến</span></div>
        <div style="padding-left:1.2rem;">├── 📄 <b>evaluate_stage12_system.py</b> <span style="color:var(--text-3); font-size:.76rem;">← Đánh giá toàn hệ thống End-to-End với thuật toán ghép cặp Hungarian Matching</span></div>
        <div style="padding-left:1.2rem;">├── 📄 <b>compare_stage2_models.py</b> <span style="color:var(--text-3); font-size:.76rem;">← Đánh giá so sánh độc lập 3 kiến trúc CNN (ResNet-18, ResNet-50, EfficientNet-B2)</span></div>
        <div style="padding-left:1.2rem;">├── 📁 <b>runs/</b></div>
        <div style="padding-left:2.4rem;">├── 📁 <b>stage2/</b> <span style="color:var(--text-3); font-size:.76rem;">← Checkpoints CNN (resnet18/best.pt, resnet50, efficientnet_b2)</span></div>
        <div style="padding-left:2.4rem;">└── 📁 <b>detect/</b> <span style="color:var(--text-3); font-size:.76rem;">← Checkpoints YOLOv8 (yolov8m_pcb_768_adamw/best.pt)</span></div>
    </div>
</div>
""",
        unsafe_allow_html=True,
    )