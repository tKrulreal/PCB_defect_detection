"""
Before/After image comparison slider for Streamlit.

Self-contained: builds an HTML/JS widget that takes two RGB images, normalises
them to the same dimensions, and renders them in a side-by-side flex layout
with a draggable divider in the middle. Both images are guaranteed to share
an identical pixel size, so the reveal is pixel-perfect.

UI behaviour
------------
The widget displays two image panels in a flex row. A vertical divider sits in
the middle and can be dragged left/right. The left panel shows the *original*
image, the right panel shows the *modified* (annotated / GradCAM) image. The
slider position determines how much of each panel is visible:

    slider = 30%  →  left panel takes 30%, right panel takes 70%
    slider = 50%  →  equal split
    slider = 80%  →  left panel takes 80%, right panel takes 20%

The container keeps a 16:9-ish aspect ratio and is responsive to column width.
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
    display: flex;
    flex-direction: row;
  }}
  .ba-panel {{
    position: relative;
    flex: 1 1 50%;
    min-width: 0;
    overflow: hidden;
    background: #0f0c29;
    display: flex;
    align-items: center;
    justify-content: center;
  }}
  .ba-panel img {{
    width: 100%;
    height: 100%;
    object-fit: contain;
    display: block;
    pointer-events: none;
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
  <div class="ba-panel" id="{elem_id}-left">
    <img src="{left_src}" alt="before" />
  </div>
  <div class="ba-panel" id="{elem_id}-right">
    <img src="{right_src}" alt="after" />
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
  const left = document.getElementById("{elem_id}-left");
  const right = document.getElementById("{elem_id}-right");
  const divider = document.getElementById("{elem_id}-div");
  let dragging = false;

  function setPos(pct) {{
    pct = Math.max(0, Math.min(100, pct));
    left.style.flex = "1 1 " + pct + "%";
    right.style.flex = "1 1 " + (100 - pct) + "%";
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


def _to_pil(image_input) -> Image.Image:
    """Coerce input (path | PIL | ndarray) into a PIL RGB image."""
    if isinstance(image_input, (str, Path)):
        return Image.open(image_input).convert("RGB")
    if isinstance(image_input, Image.Image):
        return image_input.convert("RGB")
    # assume numpy array (RGB)
    return Image.fromarray(image_input)


def _normalise_size(pil_a: Image.Image, pil_b: Image.Image, max_side: int) -> tuple[Image.Image, Image.Image]:
    """Resize both images to the same dimensions so the side-by-side layout lines up.

    The target size uses the larger of the two original dimensions, then the
    long side is clamped to ``max_side`` pixels so the encoded payload stays
    small.
    """
    aw, ah = pil_a.size
    bw, bh = pil_b.size

    target_w = max(aw, bw)
    target_h = max(ah, bh)
    scale = max(target_w / max_side, target_h / max_side, 1.0)
    target_w = int(round(target_w / scale))
    target_h = int(round(target_h / scale))

    return (
        pil_a.resize((target_w, target_h), Image.LANCZOS),
        pil_b.resize((target_w, target_h), Image.LANCZOS),
    )


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
    """Render a side-by-side image comparison widget with a draggable divider.

    Parameters
    ----------
    original : PIL.Image | str | Path | np.ndarray
        Image shown on the **left** panel.
    modified : PIL.Image | str | Path | np.ndarray
        Image shown on the **right** panel.
    right_label : str
        Caption rendered on the right panel (e.g. "Detected", "GradCAM").
    max_side : int
        Upper bound for the longer side after resize (keeps payload small).
    key : str | None
        Unique key to avoid DOM id collisions when multiple sliders are shown.
    """
    pil_left = _to_pil(original)
    pil_right = _to_pil(modified)

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
    # Account for the 12px border on each side + a small buffer for shadows.
    rendered_height = int(560 / ratio) + 24
    st.components.v1.html(html, height=rendered_height, scrolling=False)
