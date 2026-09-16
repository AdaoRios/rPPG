"""Behavioural tests for traceable weighted rPPG signal fusion."""

import unittest
from unittest.mock import patch

import numpy as np

from rPPG.config import (
    LIGHTING_BRIGHT_PIXEL_CHANNEL,
    METHOD_WEIGHTS,
    ROI_POINTS,
    ROI_WEIGHTS,
)
from rPPG.extractors.combine import (
    combine_roi_and_methods,
    normalize_weights,
    weighted_signal_combination,
)
from rPPG.preprocessing.filters import moving_average_smooth
from rPPG.roi.roi_extraction import extract_roi_means


class WeightedCombinationTests(unittest.TestCase):
    def test_weighted_algorithm_combination_matches_formula(self):
        result = weighted_signal_combination(
            {"first": np.array([1.0, 1.0, 1.0]), "second": np.array([3.0, 3.0, 3.0])},
            {"first": 0.25, "second": 0.75},
            "test algorithm fusion",
        )
        np.testing.assert_allclose(result, [2.5, 2.5, 2.5])

    def test_weighted_roi_combination_matches_formula(self):
        result = weighted_signal_combination(
            {"testa": np.array([1.0, 1.0]), "glabela": np.array([5.0, 5.0])},
            {"testa": 0.75, "glabela": 0.25},
            "test ROI fusion",
        )
        np.testing.assert_allclose(result, [2.0, 2.0])

    def test_weights_are_explicitly_normalized(self):
        normalized = normalize_weights({"a": 2.0, "b": 6.0}, ("a", "b"), "test")
        self.assertEqual(normalized, {"a": 0.25, "b": 0.75})
        result = weighted_signal_combination(
            {"a": np.array([1.0]), "b": np.array([3.0])},
            {"a": 2.0, "b": 6.0},
            "test",
        )
        np.testing.assert_allclose(result, [2.5])

    def test_invalid_weights_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "non-negative"):
            normalize_weights({"a": 1.0, "b": -1.0}, ("a", "b"), "test")
        with self.assertRaisesRegex(ValueError, "positive sum"):
            normalize_weights({"a": 0.0, "b": 0.0}, ("a", "b"), "test")
        with self.assertRaisesRegex(ValueError, "no configured weight"):
            normalize_weights({"a": 1.0}, ("a", "b"), "test")

    def test_incompatible_signal_lengths_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "incompatible lengths"):
            weighted_signal_combination(
                {"a": np.array([1.0, 1.0]), "b": np.array([2.0])},
                {"a": 0.5, "b": 0.5},
                "test",
            )


class FusionTraceTests(unittest.TestCase):
    fps = 30.0

    @staticmethod
    def _roi_rgb_signals():
        samples = np.arange(300) / 30.0
        return {
            roi_name: np.column_stack((
                100 + 2 * np.sin(2 * np.pi * 1.2 * samples + index * 0.1),
                110 + 3 * np.sin(2 * np.pi * 1.2 * samples + index * 0.1),
                120 + 4 * np.sin(2 * np.pi * 1.2 * samples + index * 0.1),
            ))
            for index, roi_name in enumerate(ROI_POINTS)
        }

    @staticmethod
    def _fake_method(frames, *_args, **_kwargs):
        return np.asarray(frames)[:, 1]

    @staticmethod
    def _fake_ica(frames, *_args, **_kwargs):
        return np.asarray(frames)[:, 1], {
            "source_selection": "test fixture",
            "selected_component": 0,
        }

    def test_all_configured_rois_and_algorithms_reach_weighted_fusion(self):
        with patch("rPPG.extractors.combine.chrom_algorithm", self._fake_method), \
             patch("rPPG.extractors.combine.pos_algorithm", self._fake_method), \
             patch("rPPG.extractors.combine.green_algorithm", self._fake_method), \
             patch("rPPG.extractors.combine.ica_algorithm", self._fake_ica):
            fusion = combine_roi_and_methods(
                self._roi_rgb_signals(), self.fps, return_audit=True
            )

        self.assertEqual(fusion.audit["final_signal"]["source"], "weighted_combination")
        self.assertEqual(fusion.audit["valid_rois"], tuple(ROI_POINTS))
        self.assertEqual(set(fusion.audit["effective_roi_weights"]), set(ROI_POINTS))
        self.assertAlmostEqual(sum(fusion.audit["effective_roi_weights"].values()), 1.0)
        self.assertEqual(fusion.audit["excluded_components"], [])
        for roi_name in ROI_POINTS:
            record = fusion.audit["roi_records"][roi_name]
            self.assertAlmostEqual(record["algorithm_weight_sum"], 1.0)
            self.assertEqual(set(record["methods"]), set(METHOD_WEIGHTS))

        # Every fake algorithm returns the same green trace. Its per-ROI
        # algorithm fusion is therefore exactly its z-score; this lets the
        # test prove that the final signal is the configured ROI-weighted sum,
        # rather than a selected ROI or algorithm.
        expected = np.zeros_like(fusion.signal)
        for roi_name, rgb in self._roi_rgb_signals().items():
            green = moving_average_smooth(rgb, window=3)[:, 1]
            z_green = (green - np.mean(green)) / np.std(green)
            expected += ROI_WEIGHTS[roi_name] * z_green
        np.testing.assert_allclose(fusion.signal, expected, atol=1e-12)

    def test_invalid_algorithm_is_explicitly_excluded_and_weights_renormalize(self):
        def failing_ica(*_args, **_kwargs):
            raise RuntimeError("ICA fixture failure")

        with patch("rPPG.extractors.combine.chrom_algorithm", self._fake_method), \
             patch("rPPG.extractors.combine.pos_algorithm", self._fake_method), \
             patch("rPPG.extractors.combine.green_algorithm", self._fake_method), \
             patch("rPPG.extractors.combine.ica_algorithm", failing_ica):
            fusion = combine_roi_and_methods(
                self._roi_rgb_signals(), self.fps, return_audit=True
            )

        excluded = [
            item for item in fusion.audit["excluded_components"] if item["name"] == "ica"
        ]
        self.assertEqual(len(excluded), len(ROI_POINTS))
        self.assertTrue(all("ICA fixture failure" in item["reason"] for item in excluded))
        methods = fusion.audit["roi_records"]["testa"]["methods"]
        self.assertAlmostEqual(methods["chrom"]["effective_weight"], 3 / 7)
        self.assertAlmostEqual(methods["pos"]["effective_weight"], 3 / 7)
        self.assertAlmostEqual(methods["green"]["effective_weight"], 1 / 7)


class GlabelaAndLightingTests(unittest.TestCase):
    def test_configured_fusion_weights_are_preserved(self):
        self.assertEqual(
            METHOD_WEIGHTS,
            {"chrom": 0.30, "pos": 0.30, "ica": 0.30, "green": 0.10},
        )
        self.assertEqual(
            ROI_WEIGHTS,
            {
                "testa": 0.36,
                "bochecha_esquerda": 0.27,
                "bochecha_direita": 0.27,
                "glabela": 0.10,
            },
        )

    def test_glabela_is_a_weighted_configured_roi(self):
        self.assertIn("glabela", ROI_POINTS)
        self.assertIn("glabela", ROI_WEIGHTS)
        self.assertEqual(ROI_POINTS["glabela"], [107, 9, 336, 168])
        self.assertAlmostEqual(sum(ROI_WEIGHTS.values()), 1.0)

    def test_glabela_mask_extracts_rgb_mean(self):
        landmarks = [(0, 0)] * 478
        landmarks[107] = (40, 20)
        landmarks[9] = (60, 10)
        landmarks[336] = (80, 20)
        landmarks[168] = (60, 50)
        frame = np.full((80, 120, 3), (10, 20, 30), dtype=np.uint8)
        means, invalid = extract_roi_means(
            frame, landmarks, {"glabela": ROI_POINTS["glabela"]}, return_diagnostics=True
        )
        self.assertEqual(invalid, {})
        np.testing.assert_allclose(means["glabela"], (10, 20, 30))

    def test_experimental_bright_pixel_threshold_is_normalized(self):
        self.assertEqual(LIGHTING_BRIGHT_PIXEL_CHANNEL, 0.784)
        self.assertGreater(LIGHTING_BRIGHT_PIXEL_CHANNEL, 0.0)
        self.assertLess(LIGHTING_BRIGHT_PIXEL_CHANNEL, 1.0)


if __name__ == "__main__":
    unittest.main()
