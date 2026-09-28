"""Webcam-free coverage for the isolated calibration infrastructure."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

try:
    from rPPG.research.calibration.metrics import absolute_error, aggregate_benchmarks, relative_error_percent
    from rPPG.research.calibration.runner import record_from_result, run_calibration
except ImportError:
    from rPPG.calibration.metrics import absolute_error, aggregate_benchmarks, relative_error_percent
    from rPPG.calibration.runner import record_from_result, run_calibration
from rPPG.config import METHOD_WEIGHTS, ROI_WEIGHTS


def fake_result():
    return SimpleNamespace(
        heart_rate=72.0, fps=30.0, valid_frames=60,
        signal_metrics={"snr": 3.0, "spectral_concentration": .7, "fft_peak": 2, "amplitude": 1, "std": .2, "energy": 4},
        spectral_data={"frequency_hz": np.array([0., .5]), "magnitude": np.array([1., 2.])},
        audit={"frame_collection": {"frames_read": 62, "lighting": {"bright_pixel_channel_threshold": .784,
            "bright_pixel_ratio": .1, "mean_luminance": .5, "dark_pixel_ratio": .2, "illumination_uniformity": .8}},
            "hr_fft": {"frequency_resolution_hz": .5, "bpm_resolution": 30., "selected_peak": {"frequency_hz": 1.2, "hr_bpm": 72.}, "second_local_peak": {}},
            "fusion": {"roi_benchmark": {"testa": {"CHROM": {"hr_bpm": 70.}, "POS": {"hr_bpm": 74.}}}}},
    )


class CalibrationTests(unittest.TestCase):
    def test_errors_and_aggregation(self):
        self.assertEqual(absolute_error(70, 72), 2)
        self.assertAlmostEqual(relative_error_percent(70, 72), 100 * 2 / 72)
        self.assertIsNone(absolute_error(70, None))
        record = record_from_result("capture_001", 20, fake_result(), 72)
        summary = aggregate_benchmarks([record])
        chrom = next(item for item in summary if item["group"] == "TESTA_CHROM")
        self.assertEqual(chrom["mae_bpm"], 2)
        self.assertEqual(chrom["n_valid"], 1)

    def test_multiple_captures_write_json_csv_and_spectra(self):
        with tempfile.TemporaryDirectory() as directory:
            records = run_calibration(2, 20, directory, capture_fn=lambda *_: "fake.mp4",
                                      analyze_fn=lambda _: fake_result(), input_fn=lambda _: "72")
            root = Path(directory)
            self.assertEqual(len(records), 2)
            self.assertTrue((root / "results.csv").is_file())
            self.assertTrue((root / "summary.csv").is_file())
            self.assertTrue((root / "spectra" / "capture_001.csv").is_file())
            payload = json.loads((root / "data" / "capture_001.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["reference_hr_bpm"], 72)
            self.assertIn("TESTA", payload["roi_algorithm_results"])

    def test_missing_or_invalid_reference_is_preserved_as_null(self):
        self.assertIsNone(record_from_result("capture_001", 20, fake_result(), "invalid")["reference_hr_bpm"])

    def test_production_weights_are_not_changed(self):
        # Guard test: pins the FROZEN production weights defined in config.py.
        # Method weights are frozen (chrom 0.30 / pos 0.40 / ica 0.20 / green
        # 0.10) and ROI weights are frozen as configuration D2 after the final
        # A-vs-D2 verification on the 25 exported captures
        # (see config.py and final_d2_check.json).
        self.assertEqual(METHOD_WEIGHTS, {"chrom": .3, "pos": .4, "ica": .2, "green": .1})
        self.assertEqual(ROI_WEIGHTS, {"testa": .40, "bochecha_esquerda": .20,
                                       "bochecha_direita": .30, "glabela": .10})