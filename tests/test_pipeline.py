"""Integration-level tests for the PCB pipeline components.

These tests verify that the pipeline's annotate and collect helpers behave
correctly.  Model-dependent tests (classify_crop_array) are guarded behind
checkpoint existence checks so they are skipped gracefully when weights are
not available.
"""

import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest


# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEMO_INPUT_DIR = PROJECT_ROOT / "demo_input"
RESNET18_CKPT = PROJECT_ROOT / "runs" / "stage2" / "resnet18" / "best.pt"


# ===================================================================
# annotate_predictions
# ===================================================================


class TestAnnotatePredictions:
    """Tests for annotate_predictions() from stage12_yolo_cnn_system."""

    def _make_dummy_image(self, tmp_dir: Path, width=200, height=150):
        """Create a solid-colour test image and return its path."""
        img = np.full((height, width, 3), fill_value=128, dtype=np.uint8)
        path = tmp_dir / "test_image.jpg"
        cv2.imwrite(str(path), img)
        return path

    def test_returns_correct_shape(self, tmp_path):
        """Annotated image should have the same shape as the input."""
        from stage12_yolo_cnn_system import annotate_predictions

        img_path = self._make_dummy_image(tmp_path)
        prediction_result = {
            "image_path": str(img_path),
            "image_size": {"width": 200, "height": 150},
            "predictions": [
                {
                    "bbox": [10, 10, 100, 100],
                    "stage1_class_id": 0,
                    "stage1_label": "mouse_bite",
                    "stage1_confidence": 0.95,
                    "stage2_class_id": 0,
                    "stage2_label": "mouse_bite",
                    "stage2_confidence": 0.98,
                    "combined_confidence": 0.931,
                },
            ],
        }

        annotated = annotate_predictions(img_path, prediction_result)
        assert annotated.shape == (150, 200, 3)
        assert annotated.dtype == np.uint8

    def test_no_predictions_same_shape(self, tmp_path):
        """With no predictions, annotated image should still match input shape."""
        from stage12_yolo_cnn_system import annotate_predictions

        img_path = self._make_dummy_image(tmp_path, width=300, height=250)
        prediction_result = {
            "image_path": str(img_path),
            "image_size": {"width": 300, "height": 250},
            "predictions": [],
        }

        annotated = annotate_predictions(img_path, prediction_result)
        assert annotated.shape == (250, 300, 3)

    def test_multiple_predictions(self, tmp_path):
        """Multiple bounding boxes should not error and should keep shape."""
        from stage12_yolo_cnn_system import annotate_predictions

        img_path = self._make_dummy_image(tmp_path, width=640, height=480)
        predictions = [
            {
                "bbox": [x * 100, 10, x * 100 + 80, 90],
                "stage1_class_id": i,
                "stage1_label": f"defect_{i}",
                "stage1_confidence": 0.9,
                "stage2_class_id": i,
                "stage2_label": f"defect_{i}",
                "stage2_confidence": 0.85,
                "combined_confidence": 0.765,
            }
            for i, x in enumerate(range(5))
        ]
        prediction_result = {
            "image_path": str(img_path),
            "image_size": {"width": 640, "height": 480},
            "predictions": predictions,
        }

        annotated = annotate_predictions(img_path, prediction_result)
        assert annotated.shape == (480, 640, 3)


# ===================================================================
# collect_input_images
# ===================================================================


class TestCollectInputImages:
    """Tests for collect_input_images() from stage12_yolo_cnn_system."""

    def test_single_file(self, tmp_path):
        """Passing a single file path returns a list with that file."""
        from stage12_yolo_cnn_system import collect_input_images

        img = np.zeros((10, 10, 3), dtype=np.uint8)
        img_path = tmp_path / "single.jpg"
        cv2.imwrite(str(img_path), img)

        result = collect_input_images(str(img_path))
        assert len(result) == 1
        assert result[0] == img_path

    def test_directory_collects_images(self, tmp_path):
        """Passing a directory returns all image files sorted."""
        from stage12_yolo_cnn_system import collect_input_images

        img = np.zeros((10, 10, 3), dtype=np.uint8)
        for name in ["c.jpg", "a.png", "b.bmp", "readme.txt"]:
            path = tmp_path / name
            if name.endswith(".txt"):
                path.write_text("not an image")
            else:
                cv2.imwrite(str(path), img)

        result = collect_input_images(str(tmp_path))
        names = [p.name for p in result]
        assert "readme.txt" not in names
        assert len(result) == 3
        # Should be sorted
        assert names == sorted(names)

    def test_empty_directory(self, tmp_path):
        """Empty directory returns empty list or raises, depending on impl."""
        from stage12_yolo_cnn_system import collect_input_images

        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        # collect_input_images returns a list for directories
        result = collect_input_images(str(empty_dir))
        assert result == []

    def test_nonexistent_path_raises(self, tmp_path):
        """Non-existent path should raise FileNotFoundError."""
        from stage12_yolo_cnn_system import collect_input_images

        with pytest.raises(FileNotFoundError):
            collect_input_images(str(tmp_path / "does_not_exist"))

    def test_demo_input_directory(self):
        """If demo_input/ exists with images, it should return them."""
        if not DEMO_INPUT_DIR.exists():
            pytest.skip("demo_input/ not found")

        from stage12_yolo_cnn_system import collect_input_images

        result = collect_input_images(str(DEMO_INPUT_DIR))
        # demo_input should have at least 1 image
        assert len(result) >= 1
        for path in result:
            assert path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}


# ===================================================================
# classify_crop_array (requires model checkpoint)
# ===================================================================


@pytest.mark.skipif(
    not RESNET18_CKPT.exists(),
    reason=f"ResNet18 checkpoint not found: {RESNET18_CKPT}",
)
class TestClassifyCropArray:
    """Tests for classify_crop_array — requires a trained checkpoint."""

    def test_returns_expected_keys(self):
        """Output dict should contain pred_idx, pred_label, confidence, probabilities."""
        from stage2_cnn_utils import classify_crop_array, load_stage2_checkpoint

        bundle = load_stage2_checkpoint(RESNET18_CKPT)
        # Create a dummy BGR crop
        crop_bgr = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)

        result = classify_crop_array(crop_bgr, bundle)

        assert "pred_idx" in result
        assert "pred_label" in result
        assert "confidence" in result
        assert "probabilities" in result

    def test_confidence_range(self):
        """Confidence should be in [0, 1]."""
        from stage2_cnn_utils import classify_crop_array, load_stage2_checkpoint

        bundle = load_stage2_checkpoint(RESNET18_CKPT)
        crop_bgr = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)

        result = classify_crop_array(crop_bgr, bundle)
        assert 0.0 <= result["confidence"] <= 1.0

    def test_probabilities_sum_to_one(self):
        """Softmax probabilities should sum to approximately 1.0."""
        from stage2_cnn_utils import classify_crop_array, load_stage2_checkpoint

        bundle = load_stage2_checkpoint(RESNET18_CKPT)
        crop_bgr = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)

        result = classify_crop_array(crop_bgr, bundle)
        prob_sum = result["probabilities"].sum()
        assert prob_sum == pytest.approx(1.0, abs=1e-4)

    def test_pred_label_is_known_class(self):
        """Predicted label should be one of the known defect classes."""
        from stage2_cnn_utils import classify_crop_array, load_stage2_checkpoint

        bundle = load_stage2_checkpoint(RESNET18_CKPT)
        crop_bgr = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)

        result = classify_crop_array(crop_bgr, bundle)
        assert result["pred_label"] in bundle["class_names"]
        assert result["pred_idx"] == bundle["class_names"].index(result["pred_label"])
