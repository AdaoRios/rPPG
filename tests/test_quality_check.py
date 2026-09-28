"""Unit tests for the landmark-based capture movement metric."""

import unittest

import numpy as np

from rPPG.capture.quality_check import FaceFramingQualityChecker, MovementQualityChecker
from rPPG.config import (
    FACE_MAX_CENTER_OFFSET_X,
    FACE_MAX_CENTER_OFFSET_Y,
    FACE_MAX_HEIGHT_RATIO,
    FACE_MAX_WIDTH_RATIO,
    FACE_MIN_HEIGHT_RATIO,
    FACE_MIN_WIDTH_RATIO,
)


def _face(scale=100.0):
    return np.array([[0.0, 0.0], [scale, 0.0], [0.0, scale], [scale, scale]])


class MovementQualityCheckerTests(unittest.TestCase):
    def _checker(self, window_size=1):
        return MovementQualityChecker(0.01, window_size, ready_stable_frames=2)

    def test_identical_landmarks_have_zero_movement(self):
        checker = self._checker()
        checker.update(_face())
        result = checker.update(_face())
        self.assertAlmostEqual(result.movement_metric, 0.0)
        self.assertTrue(result.is_stable)

    def test_small_and_large_displacements_are_distinguished(self):
        checker = self._checker()
        checker.update(_face())
        small = checker.update(_face() + 0.5)
        large = checker.update(_face() + 5.0)
        self.assertLess(small.movement_metric, large.movement_metric)
        self.assertFalse(large.is_stable)

    def test_normalization_is_scale_independent(self):
        checker_small = self._checker()
        checker_small.update(_face(100))
        small_result = checker_small.update(_face(100) + 1)
        checker_large = self._checker()
        checker_large.update(_face(200))
        large_result = checker_large.update(_face(200) + 2)
        self.assertAlmostEqual(small_result.movement_metric, large_result.movement_metric)

    def test_window_uses_median_history(self):
        checker = self._checker(window_size=3)
        checker.update(_face())
        checker.update(_face() + 0.1)
        checker.update(_face() + 5.1)
        result = checker.update(_face() + 5.2)
        self.assertLess(result.movement_metric, 0.01)
        self.assertTrue(result.is_stable)

    def test_missing_face_resets_readiness(self):
        checker = self._checker()
        checker.update(_face())
        checker.update(_face())
        result = checker.update(None)
        self.assertIsNone(result.movement_metric)
        self.assertFalse(result.is_stable)
        self.assertEqual(result.consecutive_stable_frames, 0)


class FaceFramingQualityCheckerTests(unittest.TestCase):
    frame_shape = (1000, 1000, 3)

    def setUp(self):
        self.checker = FaceFramingQualityChecker(
            min_width_ratio=FACE_MIN_WIDTH_RATIO,
            max_width_ratio=FACE_MAX_WIDTH_RATIO,
            min_height_ratio=FACE_MIN_HEIGHT_RATIO,
            max_height_ratio=FACE_MAX_HEIGHT_RATIO,
            max_center_offset_x=FACE_MAX_CENTER_OFFSET_X,
            max_center_offset_y=FACE_MAX_CENTER_OFFSET_Y,
        )

    def test_calibrated_thresholds(self):
        self.assertEqual(FACE_MIN_WIDTH_RATIO, 0.28)
        self.assertEqual(FACE_MAX_WIDTH_RATIO, 0.45)
        self.assertEqual(FACE_MIN_HEIGHT_RATIO, 0.45)
        self.assertEqual(FACE_MAX_HEIGHT_RATIO, 0.80)
        self.assertEqual(FACE_MAX_CENTER_OFFSET_X, 0.15)
        self.assertEqual(FACE_MAX_CENTER_OFFSET_Y, 0.18)

    def test_face_below_minimum_size_is_not_ok(self):
        result = self.checker.evaluate([[360.5, 275], [639.5, 725]], self.frame_shape)
        self.assertFalse(result.size_ok)
        self.assertFalse(result.framing_ok)
        self.assertEqual(result.message, "APROXIME O ROSTO")

    def test_face_exactly_at_minimum_size_is_ok(self):
        result = self.checker.evaluate([[360, 275], [640, 725]], self.frame_shape)
        self.assertEqual(result.face_width_ratio, 0.28)
        self.assertEqual(result.face_height_ratio, 0.45)
        self.assertTrue(result.size_ok)
        self.assertTrue(result.framing_ok)

    def test_face_within_size_range_is_ok(self):
        result = self.checker.evaluate([[320, 225], [680, 775]], self.frame_shape)
        self.assertTrue(result.size_ok)
        self.assertTrue(result.position_ok)
        self.assertTrue(result.framing_ok)
        self.assertEqual(result.message, "ROSTO ENQUADRADO")

    def test_face_exactly_at_maximum_size_is_ok(self):
        result = self.checker.evaluate([[275, 100], [725, 900]], self.frame_shape)
        self.assertEqual(result.face_width_ratio, 0.45)
        self.assertEqual(result.face_height_ratio, 0.80)
        self.assertTrue(result.size_ok)
        self.assertTrue(result.framing_ok)

    def test_face_above_maximum_size_is_not_ok(self):
        result = self.checker.evaluate([[274.5, 100], [725.5, 900]], self.frame_shape)
        self.assertFalse(result.size_ok)
        self.assertFalse(result.framing_ok)
        self.assertEqual(result.message, "AFASTE O ROSTO")

    def test_center_exactly_at_left_limit_is_ok(self):
        result = self.checker.evaluate([[210, 275], [490, 725]], self.frame_shape)
        self.assertEqual(result.face_center_x, 0.35)
        self.assertTrue(result.position_ok)
        self.assertTrue(result.framing_ok)

    def test_center_exactly_at_right_limit_is_ok(self):
        result = self.checker.evaluate([[510, 275], [790, 725]], self.frame_shape)
        self.assertEqual(result.face_center_x, 0.65)
        self.assertTrue(result.position_ok)
        self.assertTrue(result.framing_ok)

    def test_center_exactly_at_top_limit_is_ok(self):
        result = self.checker.evaluate([[360, 95], [640, 545]], self.frame_shape)
        self.assertEqual(result.face_center_y, 0.32)
        self.assertTrue(result.position_ok)
        self.assertTrue(result.framing_ok)

    def test_center_exactly_at_bottom_limit_is_ok(self):
        result = self.checker.evaluate([[360, 455], [640, 905]], self.frame_shape)
        self.assertEqual(result.face_center_y, 0.68)
        self.assertTrue(result.position_ok)
        self.assertTrue(result.framing_ok)

    def test_face_outside_centering_region_is_not_ok(self):
        result = self.checker.evaluate([[209, 275], [489, 725]], self.frame_shape)
        self.assertTrue(result.size_ok)
        self.assertFalse(result.position_ok)
        self.assertFalse(result.framing_ok)
        self.assertEqual(result.message, "CENTRALIZE O ROSTO")

    def test_missing_face_returns_structured_result(self):
        missing = self.checker.evaluate(None, self.frame_shape)
        self.assertFalse(missing.face_detected)
        self.assertIsNone(missing.face_width_ratio)
        self.assertIsNone(missing.face_height_ratio)
        self.assertIsNone(missing.face_center_x)
        self.assertIsNone(missing.face_center_y)
        self.assertFalse(missing.size_ok)
        self.assertFalse(missing.position_ok)
        self.assertFalse(missing.framing_ok)
        self.assertIsNone(missing.bbox)
        self.assertEqual(missing.message, "ROSTO NAO DETECTADO")

    def test_invalid_and_degenerate_landmarks_are_safe(self):
        invalid = self.checker.evaluate([[1]], self.frame_shape)
        degenerate = self.checker.evaluate([[500, 500], [500, 500]], self.frame_shape)
        self.assertFalse(invalid.face_detected)
        self.assertTrue(degenerate.face_detected)
        self.assertFalse(degenerate.size_ok)
        self.assertFalse(degenerate.framing_ok)


if __name__ == "__main__":
    unittest.main()
