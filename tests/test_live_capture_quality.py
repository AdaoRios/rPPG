"""Tests for non-blocking live face/ROI/lighting preview feedback."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from rPPG.capture.capture_video import (
    _live_metric_lines,
    _quality_status_lines,
    assess_live_roi_status,
)
from rPPG.capture.lighting_quality import LightingQualityResult
from rPPG.config import LIGHTING_BRIGHT_PIXEL_CHANNEL, ROI_POINTS


class LiveCaptureQualityTests(unittest.TestCase):
    def setUp(self):
        self.movement = SimpleNamespace(
            movement_metric=0.001,
            movement_threshold=0.010,
            is_stable=True,
        )
        self.framing = SimpleNamespace(
            face_detected=True,
            framing_ok=True,
            message="ROSTO ENQUADRADO",
            face_width_ratio=0.35,
            face_height_ratio=0.55,
            face_center_x=0.5,
            face_center_y=0.5,
        )
        self.lighting = LightingQualityResult(
            face_detected=True,
            mean_luminance=0.55,
            dark_pixel_ratio=0.01,
            bright_pixel_ratio=0.25,
            illumination_uniformity=0.88,
            lighting_ok=None,
            status="METRICS_AVAILABLE",
            message="LIGHTING METRICS NOT CALIBRATED",
        )

    def test_missing_face_marks_all_rois_unavailable(self):
        status = assess_live_roi_status(np.zeros((4, 4, 3)), None)
        self.assertEqual(status["valid_count"], 0)
        self.assertEqual(status["total"], len(ROI_POINTS))
        self.assertEqual(status["invalid_rois"], tuple(ROI_POINTS))

    def test_roi_preview_reports_count_and_invalid_names(self):
        with patch(
            "rPPG.capture.capture_video.extract_roi_means",
            return_value=({"testa": (1, 2, 3), "glabela": (1, 2, 3)}, {"bochecha_direita": "mask_too_small"}),
        ):
            status = assess_live_roi_status(np.zeros((4, 4, 3)), [(1, 1)] * 478)
        self.assertEqual(status["valid_count"], 2)
        self.assertEqual(status["total"], 4)
        self.assertEqual(status["invalid_rois"], ("bochecha_direita",))

    def test_overlay_exposes_face_roi_and_observational_lighting_state(self):
        lines = _quality_status_lines(
            self.movement,
            self.framing,
            self.lighting,
            {"valid_count": 4, "total": 4, "invalid_rois": ()},
        )
        text = "\n".join(item[0] if isinstance(item, tuple) else item for item in lines)
        self.assertIn("FACE: OK", text)
        self.assertIn("ROIs: 4/4 OK", text)
        self.assertIn("ILUMINACAO: MEDICAO ATIVA", text)
        self.assertIn("sem quality gate calibrado", text)

    def test_overlay_exposes_bright_pixel_ratio_and_threshold(self):
        lines = _live_metric_lines(self.movement, self.framing, self.lighting)
        text = "\n".join(item[0] if isinstance(item, tuple) else item for item in lines)
        self.assertIn(f">= {LIGHTING_BRIGHT_PIXEL_CHANNEL:.3f}", text)
        self.assertIn("25.0%", text)


if __name__ == "__main__":
    unittest.main()
