"""Unit tests for face-region lighting metrics."""

import unittest

import numpy as np

from rPPG.capture.lighting_quality import LightingQualityChecker
from rPPG.config import (
    LIGHTING_BRIGHT_PIXEL_CHANNEL,
    LIGHTING_DARK_PIXEL_LUMINANCE,
    LIGHTING_LUMA_WEIGHTS,
    LIGHTING_UNIFORMITY_GRID_COLUMNS,
    LIGHTING_UNIFORMITY_GRID_ROWS,
)


class LightingQualityCheckerTests(unittest.TestCase):
    face_bbox = (0, 0, 4, 4)

    def setUp(self):
        self.checker = LightingQualityChecker(
            LIGHTING_LUMA_WEIGHTS,
            LIGHTING_DARK_PIXEL_LUMINANCE,
            LIGHTING_BRIGHT_PIXEL_CHANNEL,
            LIGHTING_UNIFORMITY_GRID_ROWS,
            LIGHTING_UNIFORMITY_GRID_COLUMNS,
        )

    @staticmethod
    def _rgb(value):
        return np.full((4, 4, 3), value, dtype=np.uint8)

    def test_uniform_frame_has_expected_metrics(self):
        result = self.checker.evaluate(self._rgb(128), self.face_bbox)

        self.assertTrue(result.face_detected)
        self.assertAlmostEqual(result.mean_luminance, 128 / 255)
        self.assertEqual(result.dark_pixel_ratio, 0.0)
        self.assertEqual(result.bright_pixel_ratio, 0.0)
        self.assertEqual(result.illumination_uniformity, 1.0)

    def test_dark_frame_has_low_luminance_and_dark_pixels(self):
        result = self.checker.evaluate(self._rgb(0), self.face_bbox)

        self.assertEqual(result.mean_luminance, 0.0)
        self.assertEqual(result.dark_pixel_ratio, 1.0)
        self.assertEqual(result.bright_pixel_ratio, 0.0)

    def test_bright_frame_has_high_luminance_and_bright_pixels(self):
        result = self.checker.evaluate(self._rgb(255), self.face_bbox)

        self.assertEqual(result.mean_luminance, 1.0)
        self.assertEqual(result.dark_pixel_ratio, 0.0)
        self.assertEqual(result.bright_pixel_ratio, 1.0)

    def test_dark_pixel_ratio_uses_only_face_region(self):
        frame = self._rgb(128)
        frame[:2, :2] = 0

        result = self.checker.evaluate(frame, self.face_bbox)

        self.assertEqual(result.dark_pixel_ratio, 0.25)

    def test_bright_pixel_ratio_uses_only_face_region(self):
        frame = self._rgb(128)
        frame[:2, :2] = 255

        result = self.checker.evaluate(frame, self.face_bbox)

        self.assertEqual(result.bright_pixel_ratio, 0.25)

    def test_background_does_not_change_face_region_metrics(self):
        frame = np.full((6, 6, 3), 255, dtype=np.uint8)
        frame[1:5, 1:5] = 128

        result = self.checker.evaluate(frame, (1, 1, 5, 5))

        self.assertAlmostEqual(result.mean_luminance, 128 / 255)
        self.assertEqual(result.bright_pixel_ratio, 0.0)
        self.assertEqual(result.illumination_uniformity, 1.0)

    def test_uniform_illumination_has_maximum_uniformity(self):
        result = self.checker.evaluate(self._rgb(64), self.face_bbox)

        self.assertEqual(result.illumination_uniformity, 1.0)

    def test_uneven_illumination_has_low_uniformity(self):
        frame = self._rgb(0)
        frame[:, 2:] = 255

        result = self.checker.evaluate(frame, self.face_bbox)

        self.assertEqual(result.illumination_uniformity, 0.0)

    def test_missing_face_returns_unavailable_metrics(self):
        result = self.checker.evaluate(self._rgb(128), None)

        self.assertFalse(result.face_detected)
        self.assertIsNone(result.mean_luminance)
        self.assertIsNone(result.dark_pixel_ratio)
        self.assertIsNone(result.bright_pixel_ratio)
        self.assertIsNone(result.illumination_uniformity)
        self.assertIsNone(result.lighting_ok)
        self.assertEqual(result.status, "FACE_NOT_DETECTED")

    def test_invalid_or_degenerate_input_returns_unavailable_metrics(self):
        invalid_frame = self.checker.evaluate(np.zeros((4, 4), dtype=np.uint8), self.face_bbox)
        degenerate_bbox = self.checker.evaluate(self._rgb(128), (1, 1, 1, 3))

        self.assertEqual(invalid_frame.status, "INVALID_INPUT")
        self.assertEqual(degenerate_bbox.status, "INVALID_INPUT")
        self.assertFalse(invalid_frame.face_detected)
        self.assertFalse(degenerate_bbox.face_detected)

    def test_available_metrics_are_normalized_and_not_yet_a_decision(self):
        frame = np.array(
            [
                [[0, 0, 0], [64, 64, 64], [128, 128, 128], [255, 255, 255]],
                [[0, 0, 0], [64, 64, 64], [128, 128, 128], [255, 255, 255]],
                [[0, 0, 0], [64, 64, 64], [128, 128, 128], [255, 255, 255]],
                [[0, 0, 0], [64, 64, 64], [128, 128, 128], [255, 255, 255]],
            ],
            dtype=np.uint8,
        )
        result = self.checker.evaluate(frame, self.face_bbox)

        for value in (
            result.mean_luminance,
            result.dark_pixel_ratio,
            result.bright_pixel_ratio,
            result.illumination_uniformity,
        ):
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)
        self.assertIsNone(result.lighting_ok)
        self.assertEqual(result.status, "METRICS_AVAILABLE")
        self.assertEqual(result.message, "LIGHTING METRICS NOT CALIBRATED")


if __name__ == "__main__":
    unittest.main()
