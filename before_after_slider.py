"""
Before/After image comparison slider for Streamlit.

Self-contained: builds an HTML/JS widget that takes two RGB images, normalises
them to the same dimensions, and overlays them with a draggable vertical
divider. The two images are guaranteed to share an identical aspect ratio
and pixel size, so the "before" / "after" reveal is pixel-perfect.
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
    aspect-ratio: {ratio};
  }}
  .ba-base, .ba-top {{
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
  }}
  .ba-base img, .ba-top img {{
    width: 100%;
    height: 100%;
    object-fit: contain;
    display: block;
    pointer-events: none;
  }}
  .ba-top {{
    overflow: hidden;
    width: 50%;
  }}
  .ba-divider {{
    position: absolute;
    top: 0;
    bottom: 0;
    left: 50%;
    width: 3px;
    background: linear-gradient(180deg, #a78bfa, #60a5fa);
    transform: translateX(-50%);
    box-shadow: 0 0 12px rgba(167,139,250,0.55);
    cursor: ew-resize;
    z-index: 3;
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
    z-index: 4;
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
    z-index: 5;
  }}
  .ba-label.left  {{ left: 12px; }}
  .ba-label.right {{ right: 12px; }}
</style>

<div class="ba-wrap" id="{elem_id}">
  <div class="ba-base">
    <img src="{right_src}" alt="after" />
  </div>
  <div class="ba-top" id="{elem_id}-top">
    <img src="{left_src}" alt="before" />
  </div>
  <div class="ba-label left">Original</div>
  <div class="ba-label right">{right_label}</div>
  <div class="ba-divider" id="{elem_id}-div">
    <div class="ba-handle">⇆</div>
  </div>
</div>

<script>
(function () {{
  const root = document.getElementById("{elem_id}");
  if (!root) return;
  const top = document.getElementById("{elem_id}-top");
  const divider = document.getElementById("{elem_id}-div");
  let dragging = false;

  function setPos(pct) {{
    pct = Math.max(0, Math.min(100, pct));
    top.style.width = pct + "%";
    divider.style.left = pct + "%";
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
    return pil, pil.width, pil.height


def _normalise_size(pil_a: Image.Image, pil_b: Image.Image, max_side: int) -> tuple[Image.Image, Image.Image]:
    """Resize both images to the same dimensions so the overlay lines up.

    The result keeps aspect ratio by using the larger of the two aspect
    ratios and clamping the long side to ``max_side`` pixels.
    """
    aw, ah = pil_a.size
    bw, bh = pil_b.size

    target_w = max(aw, bw)
    target_h = max(ah, bh)
    scale = max(target_w / max_side, target_h / max_side, 1.0)
    target_w = int(round(target_w / scale))
    target_h = int(round(target_h / scale))

    pil_a = pil_a.resize((target_w, target_h), Image.LANCZOS)
    pil_b = pil_b.resize((target_w, target_h), Image.LANCZOS)
    return pil_a, pil_b


def _to_data_uri(pil: Image.Image) -> str:
    buf = BytesIO()
    pil.save(buf, format="JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def before_after_slider(
    original,
    modified,
    right_label: str = "Detected",
    max_side: int = 640,
    key: str | None = None,
) -> None:
    """Render a draggable before/after image comparison widget.

    The two images are resized to the same pixel dimensions so the overlay
    reveal is pixel-aligned.

    Parameters
    ----------
    original : PIL.Image | str | Path | np.ndarray
        Image shown on the **left** (typically the input image).
    modified : PIL.Image | str | Path | np.ndarray
        Image shown on the **right** (typically the annotated image).
    right_label : str
        Caption rendered on the right side (e.g. "Detected", "GradCAM").
    max_side : int
        Upper bound for the longer side after resize (keeps payload small).
    """
    pil_left, _, _ = _img_to_data_uri(original)
    pil_right, _, _ = _img_to_data_uri(modified)

    pil_left, pil_right = _normalise_size(pil_left, pil_right, max_side)

    left_uri = _to_data_uri(pil_left)
    right_uri = _to_data_uri(pil_right)
    ratio = pil_left.width / pil_left.height

    elem_id = f"ba-{key or 'main'}"
    html = _SLIDER_HTML_TEMPLATE.format(
        elem_id=elem_id,
        ratio=f"{ratio:.4f}",
        left_src=left_uri,
        right_src=right_uri,
        right_label=right_label,
    )
    rendered_height = int(560 / ratio) + 24
    st.components.v1.html(html, height=rendered_height, scrolling=False)
