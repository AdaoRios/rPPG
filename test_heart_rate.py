"""Synthetic tests for FFT heart-rate estimation and diagnostics."""

import unittest

import numpy as np

from rPPG.biomarkers.heart_rate import analyze_hr_fft, compute_hr_fft
from rPPG.config import HR_HIGH_HZ, HR_LOW_HZ


def _sine(frequency_hz, duration_s, fps=30.0, amplitude=1.0):
    samples = int(round(duration_s * fps))
    time = np.arange(samples) / fps
    return amplitude * np.sin(2 * np.pi * frequency_hz * time)


class HeartRateFFTTests(unittest.TestCase):
    def test_1_5_hz_signal_estimates_90_bpm(self):
        signal = _sine(1.5, duration_s=20.0)
        diagnostics = analyze_hr_fft(signal, 30.0)
        self.assertAlmostEqual(compute_hr_fft(signal, 30.0), 90.0, places=6)
        self.assertAlmostEqual(diagnostics["selected_peak"]["frequency_hz"], 1.5, places=6)
        self.assertAlmostEqual(diagnostics["bpm_resolution"], 3.0, places=6)

    def test_1_4167_hz_signal_estimates_85_bpm(self):
        # 34 cycles in 24 s is exactly 1.416666... Hz / 85 bpm.
        signal = _sine(85.0 / 60.0, duration_s=24.0)
        diagnostics = analyze_hr_fft(signal, 30.0)
        self.assertAlmostEqual(diagnostics["selected_peak"]["hr_bpm"], 85.0, places=6)
        self.assertFalse(diagnostics["zero_padding"])
        self.assertFalse(diagnostics["peak_interpolation"])

    def test_resolution_uses_effective_samples_and_fps(self):
        signal = _sine(16 / 264 * 30.0, duration_s=264 / 30.0)
        diagnostics = analyze_hr_fft(signal, 30.0)
        self.assertEqual(diagnostics["samples"], 264)
        self.assertAlmostEqual(diagnostics["frequency_resolution_hz"], 30 / 264)
        self.assertAlmostEqual(diagnostics["bpm_resolution"], 60 * 30 / 264)
        self.assertAlmostEqual(diagnostics["selected_peak"]["hr_bpm"], 16 * 60 * 30 / 264)

    def test_peak_selection_respects_cardiac_band(self):
        cardiac = _sine(1.5, duration_s=20.0, amplitude=1.0)
        out_of_band = _sine(5.0, duration_s=20.0, amplitude=10.0)
        diagnostics = analyze_hr_fft(cardiac + out_of_band, 30.0)
        selected_frequency = diagnostics["selected_peak"]["frequency_hz"]
        self.assertGreaterEqual(selected_frequency, HR_LOW_HZ)
        self.assertLessEqual(selected_frequency, HR_HIGH_HZ)
        self.assertAlmostEqual(selected_frequency, 1.5, places=6)

    def test_current_rule_can_select_a_dominant_harmonic(self):
        fundamental = _sine(1.5, duration_s=20.0, amplitude=1.0)
        harmonic = _sine(3.0, duration_s=20.0, amplitude=2.0)
        diagnostics = analyze_hr_fft(fundamental + harmonic, 30.0)
        self.assertAlmostEqual(diagnostics["selected_peak"]["hr_bpm"], 180.0, places=6)
        self.assertEqual(
            diagnostics["selection_method"],
            "maximum FFT magnitude within configured cardiac band",
        )

    def test_flat_and_too_short_signals_fail_instead_of_dividing_by_zero(self):
        with self.assertRaisesRegex(RuntimeError, "sem variabilidade"):
            analyze_hr_fft(np.zeros(120), 30.0)
        with self.assertRaisesRegex(RuntimeError, "ao menos duas amostras"):
            analyze_hr_fft(np.array([1.0]), 30.0)
        with self.assertRaisesRegex(RuntimeError, "faixa espectral vazia"):
            analyze_hr_fft(np.array([0.0, 1.0]), 30.0)


if __name__ == "__main__":
    unittest.main()
