"""Comprehensive unit tests for utils.py.

Tests cover geometry helpers (IoU, padding), YOLO label parsing,
filename stem utilities, and Hungarian prediction matching.
"""

import pytest
import numpy as np

from utils import (
    compute_iou,
    yolo_line_to_xyxy,
    canonicalize_stem,
    extract_size_suffix,
    add_padding_and_clip,
    match_predictions,
)


# ===================================================================
# compute_iou
# ===================================================================


class TestComputeIoU:
    """Tests for compute_iou(box_a, box_b)."""

    def test_perfect_overlap(self):
        """Identical boxes should have IoU = 1.0."""
        box = [10, 20, 50, 60]
        assert compute_iou(box, box) == pytest.approx(1.0)

    def test_no_overlap_distant_boxes(self):
        """Non-overlapping boxes should have IoU = 0.0."""
        box_a = [0, 0, 10, 10]
        box_b = [100, 100, 200, 200]
        assert compute_iou(box_a, box_b) == pytest.approx(0.0)

    def test_no_overlap_adjacent_boxes(self):
        """Edge-touching but non-overlapping boxes should have IoU = 0.0."""
        box_a = [0, 0, 10, 10]
        box_b = [10, 0, 20, 10]
        assert compute_iou(box_a, box_b) == pytest.approx(0.0)

    def test_partial_overlap(self):
        """Partially overlapping boxes should return the correct IoU.

        box_a = [0,0,10,10]  → area = 100
        box_b = [5,5,15,15]  → area = 100
        intersection = [5,5,10,10] → area = 25
        union = 100 + 100 - 25 = 175
        IoU = 25 / 175 ≈ 0.142857
        """
        box_a = [0, 0, 10, 10]
        box_b = [5, 5, 15, 15]
        expected = 25.0 / 175.0
        assert compute_iou(box_a, box_b) == pytest.approx(expected, abs=1e-6)

    def test_zero_area_box(self):
        """A zero-area box (line or point) should produce IoU = 0.0."""
        box_a = [5, 5, 5, 10]   # zero width
        box_b = [0, 0, 10, 10]
        assert compute_iou(box_a, box_b) == pytest.approx(0.0)

    def test_zero_area_both_boxes(self):
        """Two zero-area boxes should produce IoU = 0.0."""
        box_a = [5, 5, 5, 5]  # point
        box_b = [5, 5, 5, 5]
        assert compute_iou(box_a, box_b) == pytest.approx(0.0)

    def test_one_box_inside_another(self):
        """Inner box fully contained → IoU = inner_area / outer_area.

        inner = [2,2,4,4] → area = 4
        outer = [0,0,10,10] → area = 100
        intersection = 4
        union = 100 + 4 - 4 = 100
        IoU = 4 / 100 = 0.04
        """
        outer = [0, 0, 10, 10]
        inner = [2, 2, 4, 4]
        expected = 4.0 / 100.0
        assert compute_iou(outer, inner) == pytest.approx(expected, abs=1e-6)

    def test_one_box_inside_another_large_overlap(self):
        """Large inner box → higher IoU.

        inner = [1,1,9,9] → area = 64
        outer = [0,0,10,10] → area = 100
        union = 100 + 64 - 64 = 100
        IoU = 64 / 100 = 0.64
        """
        outer = [0, 0, 10, 10]
        inner = [1, 1, 9, 9]
        expected = 64.0 / 100.0
        assert compute_iou(outer, inner) == pytest.approx(expected, abs=1e-6)

    def test_symmetry(self):
        """IoU should be symmetric: IoU(a, b) == IoU(b, a)."""
        box_a = [0, 0, 10, 10]
        box_b = [3, 3, 12, 12]
        assert compute_iou(box_a, box_b) == pytest.approx(
            compute_iou(box_b, box_a), abs=1e-9
        )

    def test_float_coordinates(self):
        """IoU should work correctly with floating point coordinates."""
        box_a = [0.5, 0.5, 10.5, 10.5]
        box_b = [5.5, 5.5, 15.5, 15.5]
        # intersection: [5.5,5.5,10.5,10.5] → 5*5 = 25
        # areas: 10*10 = 100 each
        # union: 100 + 100 - 25 = 175
        expected = 25.0 / 175.0
        assert compute_iou(box_a, box_b) == pytest.approx(expected, abs=1e-6)


# ===================================================================
# yolo_line_to_xyxy
# ===================================================================


class TestYoloLineToXyxy:
    """Tests for yolo_line_to_xyxy(line, img_w, img_h)."""

    def test_normal_conversion(self):
        """Standard center-format to xyxy conversion.

        class=2, cx=0.5, cy=0.5, w=0.5, h=0.5 on a 640x480 image.
        cx_px=320, cy_px=240, w_px=320, h_px=240
        x1=160, y1=120, x2=480, y2=360
        """
        line = "2 0.5 0.5 0.5 0.5"
        cls_id, bbox = yolo_line_to_xyxy(line, 640, 480)
        assert cls_id == 2
        assert bbox == pytest.approx([160.0, 120.0, 480.0, 360.0], abs=1e-6)

    def test_box_at_top_left_corner(self):
        """Box anchored at image origin (top-left)."""
        # cx=0.1, cy=0.1, w=0.2, h=0.2 on 100x100
        # cx_px=10, cy_px=10, w_px=20, h_px=20
        # x1=0, y1=0, x2=20, y2=20
        line = "0 0.1 0.1 0.2 0.2"
        cls_id, bbox = yolo_line_to_xyxy(line, 100, 100)
        assert cls_id == 0
        assert bbox == pytest.approx([0.0, 0.0, 20.0, 20.0], abs=1e-6)

    def test_box_at_bottom_right_border(self):
        """Box touching the bottom-right image border."""
        # cx=0.9, cy=0.9, w=0.2, h=0.2 on 100x100
        # cx_px=90, cy_px=90, w_px=20, h_px=20
        # x1=80, y1=80, x2=100, y2=100
        line = "3 0.9 0.9 0.2 0.2"
        cls_id, bbox = yolo_line_to_xyxy(line, 100, 100)
        assert cls_id == 3
        assert bbox == pytest.approx([80.0, 80.0, 100.0, 100.0], abs=1e-6)

    def test_full_image_box(self):
        """Box covering the entire image."""
        line = "1 0.5 0.5 1.0 1.0"
        cls_id, bbox = yolo_line_to_xyxy(line, 640, 480)
        assert cls_id == 1
        assert bbox == pytest.approx([0.0, 0.0, 640.0, 480.0], abs=1e-6)

    def test_invalid_line_too_few_parts(self):
        """Line with fewer than 5 fields should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid YOLO label line"):
            yolo_line_to_xyxy("2 0.5 0.5", 640, 480)

    def test_invalid_line_empty(self):
        """Empty line should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid YOLO label line"):
            yolo_line_to_xyxy("", 640, 480)

    def test_class_id_float_notation(self):
        """Class ID written as float (e.g. '2.0') should be parsed as int 2."""
        line = "2.0 0.5 0.5 0.4 0.4"
        cls_id, _ = yolo_line_to_xyxy(line, 100, 100)
        assert cls_id == 2

    def test_line_with_extra_whitespace(self):
        """Leading/trailing whitespace should be handled."""
        line = "  1 0.5 0.5 0.5 0.5  "
        cls_id, bbox = yolo_line_to_xyxy(line, 200, 200)
        assert cls_id == 1
        assert bbox == pytest.approx([50.0, 50.0, 150.0, 150.0], abs=1e-6)


# ===================================================================
# canonicalize_stem
# ===================================================================


class TestCanonicalizeStem:
    """Tests for canonicalize_stem(stem)."""

    def test_strip_numeric_suffix(self):
        assert canonicalize_stem("image_600") == "image"

    def test_no_suffix(self):
        assert canonicalize_stem("image") == "image"

    def test_multiple_underscores(self):
        assert canonicalize_stem("a_b_123") == "a_b"

    def test_non_numeric_suffix(self):
        """Non-numeric suffix should NOT be stripped."""
        assert canonicalize_stem("image_large") == "image_large"

    def test_suffix_zero(self):
        """Numeric suffix '0' should be stripped."""
        assert canonicalize_stem("test_0") == "test"

    def test_empty_string(self):
        assert canonicalize_stem("") == ""

    def test_only_underscore_digits(self):
        """Stem is '_123' → prefix is '', which is a valid string."""
        assert canonicalize_stem("_123") == ""


# ===================================================================
# extract_size_suffix
# ===================================================================


class TestExtractSizeSuffix:
    """Tests for extract_size_suffix(stem)."""

    def test_numeric_suffix(self):
        assert extract_size_suffix("image_600") == 600

    def test_no_suffix(self):
        assert extract_size_suffix("image") == -1

    def test_non_numeric_suffix(self):
        assert extract_size_suffix("image_large") == -1

    def test_multiple_underscores(self):
        assert extract_size_suffix("a_b_123") == 123

    def test_zero_suffix(self):
        assert extract_size_suffix("test_0") == 0


# ===================================================================
# add_padding_and_clip
# ===================================================================


class TestAddPaddingAndClip:
    """Tests for add_padding_and_clip(x1, y1, x2, y2, img_w, img_h, ...)."""

    def test_normal_padding(self):
        """Standard case: box in the center of the image with default padding."""
        x1, y1, x2, y2 = add_padding_and_clip(
            100, 100, 200, 200, img_w=640, img_h=480
        )
        # Original box: 100x100.  padding_ratio=0.25 → pad 25px each side.
        # Padded: (75, 75, 225, 225)  → size 150x150 > min_size=32.
        assert x1 == 75
        assert y1 == 75
        assert x2 == 225
        assert y2 == 225

    def test_clip_to_image_bounds(self):
        """Padded box extending beyond image edges should be clipped."""
        x1, y1, x2, y2 = add_padding_and_clip(
            0, 0, 40, 40, img_w=50, img_h=50
        )
        # Box 40x40, pad=10 each side → (-10,-10,50,50) → clip to (0,0,50,50)
        assert x1 >= 0
        assert y1 >= 0
        assert x2 <= 50
        assert y2 <= 50

    def test_minimum_size_enforcement(self):
        """A very small box should be expanded to at least min_size."""
        x1, y1, x2, y2 = add_padding_and_clip(
            100, 100, 102, 102, img_w=640, img_h=480, min_size=32
        )
        # Original box: 2x2, padded → 2*1.5=3 per side, so 2.5 each way.
        # But min_size=32 → should expand to at least 32x32.
        width = x2 - x1
        height = y2 - y1
        assert width >= 32
        assert height >= 32

    def test_custom_padding_ratio(self):
        """Custom padding ratio should scale padding proportionally."""
        x1, y1, x2, y2 = add_padding_and_clip(
            100, 100, 200, 200, img_w=640, img_h=480,
            padding_ratio=0.5
        )
        # Box 100x100, pad 50px each side → (50, 50, 250, 250)
        assert x1 == 50
        assert y1 == 50
        assert x2 == 250
        assert y2 == 250

    def test_returns_integers(self):
        """Output coordinates should always be integers."""
        result = add_padding_and_clip(10.3, 20.7, 30.1, 40.9, 640, 480)
        for coord in result:
            assert isinstance(coord, int)

    def test_zero_padding(self):
        """Zero padding ratio should only enforce min_size and clipping."""
        x1, y1, x2, y2 = add_padding_and_clip(
            100, 100, 200, 200, img_w=640, img_h=480, padding_ratio=0.0
        )
        assert x1 == 100
        assert y1 == 100
        assert x2 == 200
        assert y2 == 200


# ===================================================================
# match_predictions  (Hungarian matching)
# ===================================================================


class TestMatchPredictions:
    """Tests for match_predictions(predictions, ground_truths, iou_threshold)."""

    def test_simple_one_to_one(self):
        """One prediction closely matching one GT → 1 match, no unmatched."""
        preds = [{"bbox": [10, 10, 50, 50]}]
        gts = [{"bbox": [10, 10, 50, 50], "label": "defect_a"}]
        matches, unmatched_p, unmatched_g = match_predictions(preds, gts, 0.5)

        assert len(matches) == 1
        assert matches[0][0] == 0   # pred idx
        assert matches[0][1] == 0   # gt idx
        assert matches[0][2] == pytest.approx(1.0)
        assert unmatched_p == []
        assert unmatched_g == []

    def test_multiple_preds_fewer_gts(self):
        """3 preds, 1 GT → at most 1 match, 2 unmatched preds."""
        preds = [
            {"bbox": [10, 10, 50, 50]},
            {"bbox": [200, 200, 300, 300]},
            {"bbox": [400, 400, 500, 500]},
        ]
        gts = [{"bbox": [10, 10, 50, 50], "label": "defect_a"}]
        matches, unmatched_p, unmatched_g = match_predictions(preds, gts, 0.5)

        assert len(matches) == 1
        assert matches[0][0] == 0
        assert len(unmatched_p) == 2
        assert unmatched_g == []

    def test_below_iou_threshold(self):
        """Overlapping boxes below threshold → no match."""
        preds = [{"bbox": [0, 0, 10, 10]}]
        gts = [{"bbox": [8, 8, 20, 20], "label": "defect_a"}]
        # intersection: [8,8,10,10] = 2*2=4
        # union: 100 + 144 - 4 = 240
        # IoU = 4/240 ≈ 0.017  < 0.5 threshold
        matches, unmatched_p, unmatched_g = match_predictions(preds, gts, 0.5)

        assert len(matches) == 0
        assert len(unmatched_p) == 1
        assert len(unmatched_g) == 1

    def test_empty_predictions(self):
        """No predictions → all GTs unmatched."""
        preds = []
        gts = [
            {"bbox": [10, 10, 50, 50], "label": "defect_a"},
            {"bbox": [60, 60, 100, 100], "label": "defect_b"},
        ]
        matches, unmatched_p, unmatched_g = match_predictions(preds, gts, 0.5)

        assert matches == []
        assert unmatched_p == []
        assert unmatched_g == [0, 1]

    def test_empty_ground_truths(self):
        """No GTs → all predictions unmatched."""
        preds = [
            {"bbox": [10, 10, 50, 50]},
            {"bbox": [60, 60, 100, 100]},
        ]
        gts = []
        matches, unmatched_p, unmatched_g = match_predictions(preds, gts, 0.5)

        assert matches == []
        assert unmatched_p == [0, 1]
        assert unmatched_g == []

    def test_both_empty(self):
        """No predictions and no GTs → all empty."""
        matches, unmatched_p, unmatched_g = match_predictions([], [], 0.5)
        assert matches == []
        assert unmatched_p == []
        assert unmatched_g == []

    def test_class_aware_matching_same_class(self):
        """Class-aware: same label → should match normally."""
        preds = [{"bbox": [10, 10, 50, 50], "stage1_label": "defect_a"}]
        gts = [{"bbox": [10, 10, 50, 50], "label": "defect_a"}]
        matches, unmatched_p, unmatched_g = match_predictions(
            preds, gts, 0.5, class_aware=True
        )

        assert len(matches) == 1
        assert matches[0][2] == pytest.approx(1.0)

    def test_class_aware_matching_different_class(self):
        """Class-aware: different label → no match even with perfect IoU."""
        preds = [{"bbox": [10, 10, 50, 50], "stage1_label": "defect_a"}]
        gts = [{"bbox": [10, 10, 50, 50], "label": "defect_b"}]
        matches, unmatched_p, unmatched_g = match_predictions(
            preds, gts, 0.5, class_aware=True
        )

        assert len(matches) == 0
        assert len(unmatched_p) == 1
        assert len(unmatched_g) == 1

    def test_class_aware_with_label_key_fallback(self):
        """Class-aware: uses 'label' key when 'stage1_label' is absent."""
        preds = [{"bbox": [10, 10, 50, 50], "label": "defect_a"}]
        gts = [{"bbox": [10, 10, 50, 50], "label": "defect_a"}]
        matches, _, _ = match_predictions(preds, gts, 0.5, class_aware=True)
        assert len(matches) == 1

    def test_multiple_matches_optimal(self):
        """Hungarian should find the optimal assignment.

        pred0 overlaps gt0 well (IoU~0.69) and gt1 slightly,
        pred1 overlaps gt1 well (IoU~0.69) and gt0 slightly.
        Optimal: pred0↔gt0 and pred1↔gt1.
        """
        preds = [
            {"bbox": [0, 0, 10, 10]},
            {"bbox": [20, 20, 30, 30]},
        ]
        gts = [
            {"bbox": [2, 2, 12, 12], "label": "a"},
            {"bbox": [22, 22, 32, 32], "label": "b"},
        ]
        matches, unmatched_p, unmatched_g = match_predictions(
            preds, gts, 0.3
        )

        assert len(matches) == 2
        match_dict = {m[0]: m[1] for m in matches}
        assert match_dict[0] == 0  # pred0 ↔ gt0
        assert match_dict[1] == 1  # pred1 ↔ gt1
        assert unmatched_p == []
        assert unmatched_g == []
