"""
Interactive Micro-Zoom Loupe Component for PCB High-Resolution Inspection.

Provides a circular magnifying lens (2x - 8x) that follows the user's cursor over
high-resolution PCB images to inspect micro-defects (mouse bite, spur, open circuit)
down to individual copper traces and pixels.
"""

from __future__ import annotations

import base64
import io
from typing import Any, Optional

import numpy as np
from PIL import Image


def _img_to_b64(image_input: Any) -> str:
    """Convert numpy array (RGB) or PIL Image to base64 data URL."""
    if isinstance(image_input, np.ndarray):
        if image_input.dtype != np.uint8:
            image_input = (np.clip(image_input, 0, 1) * 255).astype(np.uint8) if image_input.max() <= 1.0 else image_input.astype(np.uint8)
        pil_img = Image.fromarray(image_input)
        buffered = io.BytesIO()
        pil_img.save(buffered, format="JPEG", quality=95)
        encoded = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{encoded}"
    if isinstance(image_input, Image.Image):
        buffered = io.BytesIO()
        image_input.save(buffered, format="JPEG", quality=95)
        encoded = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{encoded}"
    return ""


def render_interactive_zoom_loupe_html(
    original_rgb: np.ndarray,
    annotated_rgb: np.ndarray,
    zoom_level: int = 4,
    lens_size: int = 160,
    container_height: int = 420,
) -> str:
    """Generate a dual-view interactive zoom loupe HTML component."""
    orig_b64 = _img_to_b64(original_rgb)
    annot_b64 = _img_to_b64(annotated_rgb)

    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@500;700&family=Plus+Jakarta+Sans:wght@600;700&display=swap');

            * {{
                box-sizing: border-box;
                margin: 0;
                padding: 0;
            }}

            body {{
                background: transparent;
                font-family: 'Plus Jakarta Sans', sans-serif;
                color: #e2e8f0;
                user-select: none;
            }}

            .loupe-grid {{
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 16px;
                width: 100%;
            }}

            .loupe-card {{
                background: #090d16;
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 12px;
                overflow: hidden;
                box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
                display: flex;
                flex-direction: column;
            }}

            .card-header {{
                padding: 8px 14px;
                background: rgba(15, 23, 42, 0.9);
                border-bottom: 1px solid rgba(255, 255, 255, 0.08);
                display: flex;
                align-items: center;
                justify-content: space-between;
                font-size: 12px;
                font-weight: 700;
            }}

            .badge {{
                font-family: 'JetBrains Mono', monospace;
                font-size: 10px;
                padding: 2px 8px;
                border-radius: 999px;
                font-weight: 700;
                text-transform: uppercase;
            }}

            .badge-cyan {{
                background: rgba(56, 189, 248, 0.15);
                color: #38bdf8;
                border: 1px solid rgba(56, 189, 248, 0.3);
            }}

            .badge-emerald {{
                background: rgba(16, 185, 129, 0.15);
                color: #10b981;
                border: 1px solid rgba(16, 185, 129, 0.3);
            }}

            .image-viewport {{
                position: relative;
                width: 100%;
                height: {container_height}px;
                display: flex;
                align-items: center;
                justify-content: center;
                cursor: crosshair;
                overflow: hidden;
                background: #06090f;
            }}

            .viewport-img {{
                max-width: 100%;
                max-height: 100%;
                object-fit: contain;
                display: block;
                pointer-events: auto;
            }}

            .zoom-lens {{
                position: absolute;
                width: {lens_size}px;
                height: {lens_size}px;
                border-radius: 50%;
                border: 3px solid #38bdf8;
                box-shadow: 0 0 20px rgba(56, 189, 248, 0.6), inset 0 0 10px rgba(0,0,0,0.5);
                pointer-events: none;
                display: none;
                background-repeat: no-repeat;
                z-index: 100;
                backdrop-filter: brightness(1.05);
            }}

            .lens-crosshair {{
                position: absolute;
                top: 50%;
                left: 50%;
                width: 14px;
                height: 14px;
                transform: translate(-50%, -50%);
                pointer-events: none;
            }}

            .lens-crosshair::before, .lens-crosshair::after {{
                content: '';
                position: absolute;
                background: rgba(56, 189, 248, 0.85);
            }}

            .lens-crosshair::before {{
                top: 6px;
                left: 0;
                width: 14px;
                height: 2px;
            }}

            .lens-crosshair::after {{
                top: 0;
                left: 6px;
                width: 2px;
                height: 14px;
            }}

            .lens-badge {{
                position: absolute;
                bottom: 6px;
                left: 50%;
                transform: translateX(-50%);
                background: rgba(0, 0, 0, 0.75);
                color: #38bdf8;
                font-family: 'JetBrains Mono', monospace;
                font-size: 10px;
                font-weight: bold;
                padding: 1px 6px;
                border-radius: 4px;
                border: 1px solid rgba(56, 189, 248, 0.4);
                white-space: nowrap;
            }}

            .instruction-bar {{
                padding: 6px 12px;
                background: #0b1120;
                border-top: 1px solid rgba(255, 255, 255, 0.06);
                font-size: 11px;
                color: #94a3b8;
                display: flex;
                align-items: center;
                justify-content: space-between;
            }}
        </style>
    </head>
    <body>

        <div class="loupe-grid">
            <!-- Box 1: Raw Original PCB -->
            <div class="loupe-card">
                <div class="card-header">
                    <span style="color: #f8fafc;">1. Original Raw PCB</span>
                    <span class="badge badge-cyan">Raw Optical</span>
                </div>
                <div class="image-viewport" id="viewport-orig">
                    <img src="{orig_b64}" class="viewport-img" id="img-orig" alt="Original PCB"/>
                    <div class="zoom-lens" id="lens-orig">
                        <div class="lens-crosshair"></div>
                        <div class="lens-badge">{zoom_level}X MAGNIFIER</div>
                    </div>
                </div>
                <div class="instruction-bar">
                    <span>🔍 Rê chuột vào ảnh để soi vết cắt</span>
                    <span style="font-family: 'JetBrains Mono', monospace; color: #38bdf8;">Zoom: {zoom_level}x</span>
                </div>
            </div>

            <!-- Box 2: Annotated Detections PCB -->
            <div class="loupe-card">
                <div class="card-header">
                    <span style="color: #f8fafc;">2. AI Detections & Refined</span>
                    <span class="badge badge-emerald">AI Inspection</span>
                </div>
                <div class="image-viewport" id="viewport-annot">
                    <img src="{annot_b64}" class="viewport-img" id="img-annot" alt="Annotated PCB"/>
                    <div class="zoom-lens" id="lens-annot" style="border-color: #10b981; box-shadow: 0 0 20px rgba(16, 185, 129, 0.6);">
                        <div class="lens-crosshair" style="filter: hue-rotate(90deg);"></div>
                        <div class="lens-badge" style="color: #10b981; border-color: rgba(16, 185, 129, 0.4);">{zoom_level}X MAGNIFIER</div>
                    </div>
                </div>
                <div class="instruction-bar">
                    <span>🎯 Soi chi tiết BBox & vi mạch</span>
                    <span style="font-family: 'JetBrains Mono', monospace; color: #10b981;">Lens: {lens_size}px</span>
                </div>
            </div>
        </div>

        <script>
            function setupLoupe(viewportId, imgId, lensId, zoomFactor) {{
                const viewport = document.getElementById(viewportId);
                const img = document.getElementById(imgId);
                const lens = document.getElementById(lensId);

                function updateLoupe(e) {{
                    const rect = img.getBoundingClientRect();
                    const viewRect = viewport.getBoundingClientRect();

                    // Pointer relative to image
                    let clientX = e.clientX || (e.touches && e.touches[0].clientX);
                    let clientY = e.clientY || (e.touches && e.touches[0].clientY);

                    if (!clientX || !clientY) return;

                    let x = clientX - rect.left;
                    let y = clientY - rect.top;

                    // Bounds check
                    if (x < 0 || x > rect.width || y < 0 || y > rect.height) {{
                        lens.style.display = 'none';
                        return;
                    }}

                    lens.style.display = 'block';

                    const lensRadius = lens.offsetWidth / 2;
                    // Position lens relative to viewport
                    const lensX = (clientX - viewRect.left) - lensRadius;
                    const lensY = (clientY - viewRect.top) - lensRadius;

                    lens.style.left = lensX + 'px';
                    lens.style.top = lensY + 'px';

                    // Background zoom math
                    const bgWidth = rect.width * zoomFactor;
                    const bgHeight = rect.height * zoomFactor;
                    const bgX = (x / rect.width) * bgWidth - lensRadius;
                    const bgY = (y / rect.height) * bgHeight - lensRadius;

                    lens.style.backgroundImage = `url(${{img.src}})`;
                    lens.style.backgroundSize = `${{bgWidth}}px ${{bgHeight}}px`;
                    lens.style.backgroundPosition = `-${{bgX}}px -${{bgY}}px`;
                }}

                viewport.addEventListener('mousemove', updateLoupe);
                viewport.addEventListener('touchmove', updateLoupe, {{ passive: true }});

                viewport.addEventListener('mouseleave', () => {{
                    lens.style.display = 'none';
                }});
                viewport.addEventListener('touchend', () => {{
                    lens.style.display = 'none';
                }});
            }}

            window.addEventListener('DOMContentLoaded', () => {{
                setupLoupe('viewport-orig', 'img-orig', 'lens-orig', {zoom_level});
                setupLoupe('viewport-annot', 'img-annot', 'lens-annot', {zoom_level});
            }});
        </script>
    </body>
    </html>
    """
    return html_code
