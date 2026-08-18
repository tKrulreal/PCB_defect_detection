"""
QA/QC Quality Inspection Certificate Generator for PCB Defect Detection System.

Generates self-contained, printable industrial inspection certificates in HTML format
(ready for saving as PDF via browser print dialog) with embedded base64 images,
inspection metadata, digital stamps, defect coordinates, and Grad-CAM visualizations.
"""

from __future__ import annotations

import base64
import datetime
import io
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from PIL import Image


def _encode_image_to_base64(image_input: Any) -> str:
    """Convert numpy array (RGB/BGR), PIL Image, or file path to base64 PNG data URL."""
    if image_input is None:
        return ""

    if isinstance(image_input, (str, Path)):
        image_path = Path(image_input)
        if not image_path.exists():
            return ""
        with open(image_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
        return f"data:image/jpeg;base64,{encoded}"

    if isinstance(image_input, np.ndarray):
        # If 3 channels, assume RGB unless specified
        if image_input.dtype != np.uint8:
            image_input = (np.clip(image_input, 0, 1) * 255).astype(np.uint8) if image_input.max() <= 1.0 else image_input.astype(np.uint8)
        
        pil_img = Image.fromarray(image_input)
        buffered = io.BytesIO()
        pil_img.save(buffered, format="PNG")
        encoded = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{encoded}"

    if isinstance(image_input, Image.Image):
        buffered = io.BytesIO()
        image_input.save(buffered, format="PNG")
        encoded = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{encoded}"

    return ""


def generate_qc_certificate_html(
    image_name: str,
    original_rgb: np.ndarray,
    annotated_rgb: np.ndarray,
    predictions: List[Dict[str, Any]],
    latency_ms: float,
    device_name: str = "NVIDIA RTX 4060 GPU",
    pipeline_name: str = "YOLOv8m + ResNet18 (Stage 1-2 Hybrid)",
    engineer_name: str = "Nguyễn Duy Khương",
    engineer_id: str = "11236134",
    gradcam_crops: Optional[List[Dict[str, Any]]] = None,
    batch_id: Optional[str] = None,
) -> str:
    """Generate a clean, high-precision industrial QA/QC Inspection Certificate."""
    now = datetime.datetime.now()
    timestamp_str = now.strftime("%Y-%m-%d %H:%M:%S")
    date_str = now.strftime("%d/%m/%Y")
    
    if batch_id is None:
        batch_id = f"SMT-BATCH-{now.strftime('%Y%m%d')}-01"
    
    cert_id = f"QC-{now.strftime('%y%m%d')}-{abs(hash(image_name)) % 100000:05d}"
    
    num_defects = len(predictions)
    is_passed = num_defects == 0
    verdict_text = "PASSED (CONFORMANT)" if is_passed else "REJECTED (DEFECTS DETECTED)"
    verdict_color = "#10B981" if is_passed else "#EF4444"
    verdict_badge_bg = "rgba(16, 185, 129, 0.12)" if is_passed else "rgba(239, 68, 68, 0.12)"
    verdict_border = "#10B981" if is_passed else "#EF4444"

    orig_b64 = _encode_image_to_base64(original_rgb)
    annot_b64 = _encode_image_to_base64(annotated_rgb)

    # Defect rows
    defect_rows_html = ""
    if predictions:
        for idx, pred in enumerate(predictions, start=1):
            label = pred.get("stage2_label", pred.get("defect", "unknown"))
            s1_conf = pred.get("stage1_confidence", 0.0)
            s2_conf = pred.get("stage2_confidence", 0.0)
            comb_conf = pred.get("combined_confidence", 0.0)
            bbox = pred.get("bbox", [0, 0, 0, 0])
            bbox_str = f"[{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]"
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]

            defect_rows_html += f"""
            <tr>
                <td style="text-align: center; font-weight: bold; color: #64748b;">#{idx}</td>
                <td><strong style="color: #0f172a; text-transform: uppercase; font-size: 13px;">{label.replace('_', ' ')}</strong></td>
                <td style="font-family: monospace; font-size: 12px; color: #334155;">{bbox_str} ({w}×{h} px)</td>
                <td style="text-align: right; font-family: monospace; font-weight: 600; color: #475569;">{s1_conf*100:.1f}%</td>
                <td style="text-align: right; font-family: monospace; font-weight: 600; color: #475569;">{s2_conf*100:.1f}%</td>
                <td style="text-align: right; font-family: monospace; font-weight: 700; color: #ef4444;">{comb_conf*100:.1f}%</td>
                <td style="text-align: center;"><span style="display:inline-block; padding: 2px 8px; border-radius: 4px; background: #fee2e2; color: #991b1b; font-size: 11px; font-weight: bold;">FAIL</span></td>
            </tr>
            """
    else:
        defect_rows_html = """
        <tr>
            <td colspan="7" style="text-align: center; padding: 20px; color: #10b981; font-weight: 600; font-size: 14px;">
                ✓ Không phát hiện khuyết tật nào trên bo mạch. Bo mạch đạt chuẩn 100% IPC-A-610 Class 3.
            </td>
        </tr>
        """

    # GradCAM gallery HTML
    gradcam_html = ""
    if gradcam_crops:
        gradcam_items = ""
        for idx, gc in enumerate(gradcam_crops[:4], start=1):
            overlay_b64 = _encode_image_to_base64(gc.get("overlay"))
            crop_label = gc.get("label", "Defect")
            crop_conf = gc.get("confidence", 0.0)
            if overlay_b64:
                gradcam_items += f"""
                <div style="flex: 1; min-width: 140px; max-width: 180px; border: 1px solid #e2e8f0; border-radius: 8px; padding: 8px; background: #f8fafc; text-align: center;">
                    <div style="font-size: 11px; font-weight: bold; color: #1e293b; margin-bottom: 4px; text-transform: uppercase;">#{idx} {crop_label}</div>
                    <img src="{overlay_b64}" style="width: 100%; height: 110px; object-fit: cover; border-radius: 4px; border: 1px solid #cbd5e1;" alt="GradCAM"/>
                    <div style="font-size: 10px; color: #64748b; margin-top: 4px; font-family: monospace;">Conf: <strong>{crop_conf*100:.1f}%</strong></div>
                </div>
                """
        if gradcam_items:
            gradcam_html = f"""
            <div style="margin-top: 24px;">
                <h3 style="font-size: 14px; text-transform: uppercase; letter-spacing: 0.05em; color: #334155; margin-bottom: 10px; border-bottom: 1px solid #e2e8f0; padding-bottom: 4px;">
                    🔬 Trực quan hóa Giải thích AI (Explainable AI - Grad-CAM XAI Heatmaps)
                </h3>
                <div style="display: flex; gap: 12px; flex-wrap: wrap;">
                    {gradcam_items}
                </div>
            </div>
            """

    html_content = f"""<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Quality Inspection Certificate - {cert_id}</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600;700&display=swap');

        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}

        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            color: #0f172a;
            background-color: #f1f5f9;
            padding: 24px;
            font-size: 13px;
            line-height: 1.5;
        }}

        .cert-container {{
            max-width: 900px;
            margin: 0 auto;
            background: #ffffff;
            border-radius: 12px;
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.06);
            border: 1px solid #e2e8f0;
            padding: 36px 44px;
            position: relative;
        }}

        .header {{
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            border-bottom: 2px solid #0f172a;
            padding-bottom: 20px;
            margin-bottom: 24px;
        }}

        .company-logo {{
            display: flex;
            align-items: center;
            gap: 12px;
        }}

        .logo-box {{
            width: 44px;
            height: 44px;
            background: #0f172a;
            color: #38bdf8;
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 800;
            font-size: 20px;
            font-family: 'JetBrains Mono', monospace;
        }}

        .company-info h1 {{
            font-size: 18px;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: -0.01em;
            color: #0f172a;
        }}

        .company-info p {{
            font-size: 11px;
            color: #64748b;
            margin-top: 2px;
        }}

        .cert-title-block {{
            text-align: right;
        }}

        .cert-title-block h2 {{
            font-size: 18px;
            font-weight: 800;
            color: #0f172a;
            letter-spacing: -0.02em;
            text-transform: uppercase;
        }}

        .cert-code {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 12px;
            font-weight: 700;
            color: #2563eb;
            margin-top: 4px;
        }}

        .meta-grid {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 12px;
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 14px 18px;
            margin-bottom: 24px;
        }}

        .meta-item {{
            display: flex;
            flex-direction: column;
        }}

        .meta-label {{
            font-size: 10px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #64748b;
        }}

        .meta-value {{
            font-size: 12px;
            font-weight: 600;
            color: #0f172a;
            margin-top: 2px;
        }}

        .verdict-banner {{
            border-radius: 8px;
            border: 2px solid {verdict_border};
            background: {verdict_badge_bg};
            padding: 16px 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 24px;
        }}

        .verdict-title {{
            font-size: 16px;
            font-weight: 800;
            color: {verdict_color};
            text-transform: uppercase;
            letter-spacing: 0.02em;
        }}

        .verdict-sub {{
            font-size: 12px;
            color: #334155;
            margin-top: 2px;
        }}

        .stamp-box {{
            border: 3px dashed {verdict_color};
            padding: 6px 14px;
            border-radius: 6px;
            color: {verdict_color};
            font-weight: 800;
            font-size: 14px;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            transform: rotate(-3deg);
        }}

        .images-section {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 16px;
            margin-bottom: 24px;
        }}

        .image-card {{
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            overflow: hidden;
            background: #0f172a;
        }}

        .image-card-header {{
            background: #1e293b;
            color: #f8fafc;
            padding: 6px 12px;
            font-size: 11px;
            font-weight: 700;
            display: flex;
            justify-content: space-between;
        }}

        .image-card img {{
            width: 100%;
            height: 220px;
            object-fit: contain;
            display: block;
            background: #090d16;
        }}

        .table-section {{
            margin-bottom: 24px;
        }}

        .table-section h3 {{
            font-size: 14px;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #334155;
            margin-bottom: 8px;
            border-bottom: 1px solid #e2e8f0;
            padding-bottom: 4px;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 12px;
        }}

        th {{
            background: #f1f5f9;
            color: #475569;
            font-weight: 700;
            text-align: left;
            padding: 8px 10px;
            border-bottom: 2px solid #cbd5e1;
            font-size: 11px;
            text-transform: uppercase;
        }}

        td {{
            padding: 8px 10px;
            border-bottom: 1px solid #e2e8f0;
            color: #1e293b;
        }}

        .footer-signatures {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 40px;
            margin-top: 32px;
            padding-top: 20px;
            border-top: 1px solid #e2e8f0;
        }}

        .signature-box {{
            display: flex;
            flex-direction: column;
            align-items: center;
            text-align: center;
        }}

        .signature-title {{
            font-size: 11px;
            font-weight: 700;
            text-transform: uppercase;
            color: #64748b;
        }}

        .signature-line {{
            margin-top: 45px;
            width: 80%;
            border-top: 1px solid #94a3b8;
            padding-top: 6px;
            font-weight: 700;
            font-size: 12px;
            color: #0f172a;
        }}

        .print-btn-bar {{
            max-width: 900px;
            margin: 0 auto 16px auto;
            display: flex;
            justify-content: flex-end;
            gap: 10px;
        }}

        .btn {{
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 8px 16px;
            border-radius: 6px;
            font-weight: 600;
            font-size: 13px;
            cursor: pointer;
            text-decoration: none;
            border: none;
            transition: all 0.2s;
        }}

        .btn-primary {{
            background: #0f172a;
            color: #ffffff;
        }}

        .btn-primary:hover {{
            background: #1e293b;
        }}

        @media print {{
            body {{
                background: #ffffff;
                padding: 0;
            }}
            .cert-container {{
                box-shadow: none;
                border: none;
                padding: 10px;
            }}
            .print-btn-bar {{
                display: none;
            }}
        }}
    </style>
</head>
<body>

    <div class="print-btn-bar">
        <button class="btn btn-primary" onclick="window.print()">🖨️ In Phiếu Kiểm Định / Lưu PDF (Ctrl + P)</button>
    </div>

    <div class="cert-container">
        
        <!-- Header -->
        <div class="header">
            <div class="company-logo">
                <div class="logo-box">AOI</div>
                <div class="company-info">
                    <h1>PCB Quality Inspection Station</h1>
                    <p>Hệ thống Kiểm định Quang học Tự động — Chuẩn IPC-A-610 Class 3</p>
                </div>
            </div>
            <div class="cert-title-block">
                <h2>Phiếu Kiểm Định Chất Lượng</h2>
                <div class="cert-code">{cert_id}</div>
            </div>
        </div>

        <!-- Meta Grid -->
        <div class="meta-grid">
            <div class="meta-item">
                <span class="meta-label">Mã Bo Mạch / File</span>
                <span class="meta-value" style="font-family: monospace;">{image_name}</span>
            </div>
            <div class="meta-item">
                <span class="meta-label">Lô Sản Xuất (Batch ID)</span>
                <span class="meta-value">{batch_id}</span>
            </div>
            <div class="meta-item">
                <span class="meta-label">Thời Gian Kiểm Định</span>
                <span class="meta-value">{timestamp_str}</span>
            </div>
            <div class="meta-item">
                <span class="meta-label">Kỹ Sư Kiểm Định</span>
                <span class="meta-value">{engineer_name} ({engineer_id})</span>
            </div>
            <div class="meta-item">
                <span class="meta-label">Mô Hình AI Pipeline</span>
                <span class="meta-value">{pipeline_name}</span>
            </div>
            <div class="meta-item">
                <span class="meta-label">Phần Cứng Thực Thi</span>
                <span class="meta-value">{device_name}</span>
            </div>
            <div class="meta-item">
                <span class="meta-label">Thời Gian Suy Luận (Latency)</span>
                <span class="meta-value" style="color: #2563eb;">{latency_ms:.1f} ms (~{1000.0/latency_ms if latency_ms > 0 else 0:.1f} FPS)</span>
            </div>
            <div class="meta-item">
                <span class="meta-label">Số Lỗi Phát Hiện</span>
                <span class="meta-value" style="color: {'#ef4444' if num_defects > 0 else '#10b981'}; font-weight: 700;">{num_defects} Khuyết tật</span>
            </div>
        </div>

        <!-- Inspection Verdict -->
        <div class="verdict-banner">
            <div>
                <div class="verdict-title">KẾT QUẢ NGHIỆM THU: {verdict_text}</div>
                <div class="verdict-sub">
                    {'Bo mạch KHÔNG ĐẠT tiêu chuẩn xuất xưởng do có khuyết tật vi mô vượt ngưỡng cho phép.' if num_defects > 0 else 'Bo mạch hoàn toàn sạch lỗi, đạt chuẩn chất lượng xuất xưởng của dây chuyền SMT.'}
                </div>
            </div>
            <div class="stamp-box">
                {'❌ REJECTED' if num_defects > 0 else '✅ QC PASSED'}
            </div>
        </div>

        <!-- Inspection Images -->
        <div class="images-section">
            <div class="image-card">
                <div class="image-card-header">
                    <span>1. ẢNH QUANG HỌC GỐC (RAW PCB)</span>
                    <span>100% SCALE</span>
                </div>
                <img src="{orig_b64}" alt="Original PCB"/>
            </div>
            <div class="image-card">
                <div class="image-card-header">
                    <span>2. ẢNH PHÂN TÍCH AI (BOUNDING BOX & REFINED)</span>
                    <span style="color: #38bdf8;">{num_defects} DETECTIONS</span>
                </div>
                <img src="{annot_b64}" alt="Annotated PCB"/>
            </div>
        </div>

        <!-- Defect Details Table -->
        <div class="table-section">
            <h3>Danh Sách Chi Tiết Vết Cắt Khuyết Tật (Defect Inspection Breakdown)</h3>
            <table>
                <thead>
                    <tr>
                        <th style="width: 40px; text-align: center;">STT</th>
                        <th>Loại Khuyết Tật (Defect Class)</th>
                        <th>Tọa Độ & Kích Thước (BBox x1 y1 x2 y2)</th>
                        <th style="text-align: right;">Stage 1 (YOLO)</th>
                        <th style="text-align: right;">Stage 2 (CNN)</th>
                        <th style="text-align: right;">Độ Tin Cậy Kết Hợp</th>
                        <th style="text-align: center; width: 60px;">Đánh Giá</th>
                    </tr>
                </thead>
                <tbody>
                    {defect_rows_html}
                </tbody>
            </table>
        </div>

        <!-- GradCAM Section (if available) -->
        {gradcam_html}

        <!-- Footer Signatures -->
        <div class="footer-signatures">
            <div class="signature-box">
                <div class="signature-title">HỆ THỐNG KIỂM ĐỊNH TỰ ĐỘNG (SMT AOI)</div>
                <div class="signature-line">Smart Factory Automated Engine v2.4</div>
            </div>
            <div class="signature-box">
                <div class="signature-title">KỸ SƯ TRƯỞNG KIỂM SOÁT CHẤT LƯỢNG (QA/QC)</div>
                <div class="signature-line">{engineer_name} — {engineer_id}</div>
            </div>
        </div>

    </div>

</body>
</html>
"""
    return html_content
