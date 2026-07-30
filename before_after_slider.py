"""
Before/After image comparison slider for Streamlit.

Self-contained: builds an HTML/JS widget that takes two RGB images (encoded
as base64 data-URIs) and overlays them with a draggable vertical divider.

Returns the rendered component via ``st.components.v1.html``.
"""

from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path

import streamlit as st
from PIL import Image


_SLIDER_HTML_TEMPLATE = """
<style>
  .ba-wrap {{
    position: relative;
    width: 100%;
    max-width: 100%;
    user-select: none;
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid rgba(167,139,250,0.18);
    box-shadow: 0 6px 24px rgba(0,0,0,0.30);
    background: #0f0c29;
  }}
  .ba-wrap img {{
    display: block;
    width: 100%;
    height: auto;
    pointer-events: none;
  }}
  .ba-layer {{
    position: absolute;
    inset: 0;
    overflow: hidden;
  }}
  .ba-layer img {{
    width: var(--ba-w);
    height: var(--ba-h);
    max-width: none;
  }}
  .ba-divider {{
    position: absolute;
    top: 0;
    bottom: 0;
    width: 3px;
    background: linear-gradient(180deg, #a78bfa, #60a5fa);
    transform: translateX(-50%);
    box-shadow: 0 0 12px rgba(167,139,250,0.55);
    cursor: ew-resize;
  }}
  .ba-handle {{
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    width: 40px;
    height: 40px;
    border-radius: 50%;
    background: linear-gradient(135deg, #a78bfa, #60a5fa);
    box-shadow: 0 4px 14px rgba(0,0,0,0.45);
    display: flex;
    align-items: center;
    justify-content: center;
    color: #fff;
    font-weight: 700;
    font-size: 16px;
  }}
  .ba-label {{
    position: absolute;
    top: 12px;
    padding: 4px 10px;
    border-radius: 6px;
    background: rgba(15,12,41,0.78);
    color: #fff;
    font-family: 'Inter', sans-serif;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.04em;
    backdrop-filter: blur(6px);
  }}
  .ba-label.left  {{ left: 12px; }}
  .ba-label.right {{ right: 12px; }}
</style>

<div class="ba-wrap" id="{elem_id}" style="--ba-w: {width}px; --ba-h: {height}px;">
  <img src="{right_src}" alt="after" />
  <div class="ba-layer" id="{elem_id}-left">
    <img src="{left_src}" alt="before" />
  </div>
  <div class="ba-label left">Original</div>
  <div class="ba-label right">{right_label}</div>
  <div class="ba-divider" id="{elem_id}-div" style="left: 50%;">
    <div class="ba-handle">⇆</div>
  </div>
</div>

<script>
(function () {{
  const root = document.getElementById("{elem_id}");
  if (!root) return;
  const layer = document.getElementById("{elem_id}-left");
  const divider = document.getElementById("{elem_id}-div");
  let dragging = false;

  function setPos(pct) {{
    pct = Math.max(0, Math.min(100, pct));
    divider.style.left = pct + "%";
    layer.style.width = pct + "%";
  }}
  setPos(50);

  function pctFromEvent(evt) {{
    const rect = root.getBoundingClientRect();
    const x = (evt.touches ? evt.touches[0].clientX : evt.clientX) - rect.left;
    return (x / rect.width) * 100;
  }}

  root.addEventListener("mousedown", function (e) {{
    dragging = true;
    setPos(pctFromEvent(e));
    e.preventDefault();
  }});
  window.addEventListener("mousemove", function (e) {{
    if (!dragging) return;
    setPos(pctFromEvent(e));
  }});
  window.addEventListener("mouseup", function () {{ dragging = false; }});

  root.addEventListener("touchstart", function (e) {{
    dragging = true;
    setPos(pctFromEvent(e));
  }}, {{ passive: true }});
  window.addEventListener("touchmove", function (e) {{
    if (!dragging) return;
    setPos(pctFromEvent(e));
  }}, {{ passive: true }});
  window.addEventListener("touchend", function () {{ dragging = false; }});
}})();
</script>
"""


def _img_to_data_uri(image_input) -> tuple[str, int, int]:
    """Encode a PIL Image / path / ndarray as (data_uri, width, height)."""
    if isinstance(image_input, (str, Path)):
        pil = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, Image.Image):
        pil = image_input.convert("RGB")
    else:
        # assume numpy array (RGB)
        pil = Image.fromarray(image_input)

    buf = BytesIO()
    pil.save(buf, format="JPEG", quality=90)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}", pil.width, pil.height


def before_after_slider(
    original,
    modified,
    right_label: str = "Detected",
    height: int = 480,
    key: str | None = None,
) -> None:
    """Render a draggable before/after image comparison widget.

    Parameters
    ----------
    original : PIL.Image | str | Path | np.ndarray
        Image shown on the **left** (typically the input image).
    modified : PIL.Image | str | Path | np.ndarray
        Image shown on the **right** (typically the annotated image).
    right_label : str
        Caption rendered on the right side (e.g. "Detected", "GradCAM").
    """
    left_uri, w, h = _img_to_data_uri(original)
    right_uri, _, _ = _img_to_data_uri(modified)

    # Scale rendered pixel size to the desired height while keeping aspect.
    rendered_h = height
    rendered_w = int(w * rendered_h / h)

    elem_id = f"ba-{key or 'main'}"
    html = _SLIDER_HTML_TEMPLATE.format(
        elem_id=elem_id,
        width=rendered_w,
        height=rendered_h,
        left_src=left_uri,
        right_src=right_uri,
        right_label=right_label,
    )
    st.components.v1.html(html, height=rendered_h + 20, scrolling=False)
