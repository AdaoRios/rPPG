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

    def test_fundamental_dominates_and_harmonic_is_diagnostic(self):
        signal = _sine(1.2, 20.0) + _sine(2.4, 20.0, amplitude=0.3)
        diagnostics = analyze_hr_fft(signal, 30.0)
        self.assertAlmostEqual(diagnostics["selected_peak"]["hr_bpm"], 72.0)
        self.assertEqual(diagnostics["decision"]["selected_candidate_id"],
                         diagnostics["decision"]["original_maximum_candidate_id"])
        self.assertEqual(len(diagnostics["candidate_detection"]["candidates"]), 2)
        self.assertTrue(diagnostics["relationships"][0]["harmonic_relationship"])

    def test_supported_dominant_harmonic_reconsiders_fundamental(self):
        signal = _sine(1.5, 20.0) + _sine(3.0, 20.0, amplitude=2.0)
        diagnostics = analyze_hr_fft(signal, 30.0)
        self.assertAlmostEqual(diagnostics["selected_peak"]["hr_bpm"], 90.0)
        self.assertTrue(diagnostics["decision"]["harmonic_reconsideration"])
        self.assertNotEqual(diagnostics["decision"]["selected_candidate_id"],
                            diagnostics["decision"]["original_maximum_candidate_id"])

    def test_weak_fundamental_does_not_force_harmonic_reinterpretation(self):
        signal = _sine(1.5, 20.0, amplitude=0.1) + _sine(3.0, 20.0, amplitude=2.0)
        diagnostics = analyze_hr_fft(signal, 30.0)
        self.assertAlmostEqual(diagnostics["selected_peak"]["hr_bpm"], 180.0)
        self.assertFalse(diagnostics["decision"]["harmonic_reconsideration"])

    def test_unrelated_larger_peak_remains_the_selection(self):
        signal = _sine(2.2, 20.0, amplitude=2.0) + _sine(1.7, 20.0, amplitude=0.8)
        diagnostics = analyze_hr_fft(signal, 30.0)
        self.assertAlmostEqual(diagnostics["selected_peak"]["hr_bpm"], 132.0)
        self.assertTrue(all(not item["harmonic_relationship"]
                            for item in diagnostics["relationships"]))

    def test_isolated_larger_peak_is_not_reinterpreted_without_harmonic_support(self):
        signal = _sine(1.5, 20.0) + _sine(2.3, 20.0, amplitude=1.5)
        diagnostics = analyze_hr_fft(signal, 30.0)
        self.assertAlmostEqual(diagnostics["selected_peak"]["hr_bpm"], 138.0)
        self.assertFalse(diagnostics["decision"]["harmonic_reconsideration"])

    def test_unresolved_close_peaks_are_one_candidate(self):
        signal = _sine(1.5, 20.0) + _sine(1.6, 20.0, amplitude=0.8)
        diagnostics = analyze_hr_fft(signal, 30.0)
        self.assertEqual(len(diagnostics["candidate_detection"]["candidates"]), 1)
        self.assertEqual(len(diagnostics["candidate_detection"]["grouped_peak_bins"]), 1)

    def test_multiple_harmonics_are_reported_without_low_frequency_bias(self):
        signal = (_sine(1.0, 20.0) + _sine(2.0, 20.0, amplitude=0.7)
                  + _sine(3.0, 20.0, amplitude=0.4))
        diagnostics = analyze_hr_fft(signal, 30.0)
        self.assertAlmostEqual(diagnostics["selected_peak"]["hr_bpm"], 60.0)
        self.assertEqual(len(diagnostics["candidate_detection"]["candidates"]), 3)
        self.assertTrue(any(item["lower_candidate_id"] != item["upper_candidate_id"]
                            for item in diagnostics["relationships"]))

    def test_single_candidate_and_low_middle_high_rates(self):
        for frequency_hz in (2.5, 0.8, 1.5, 3.2):
            diagnostics = analyze_hr_fft(_sine(frequency_hz, 20.0), 30.0)
            self.assertAlmostEqual(
                diagnostics["selected_peak"]["frequency_hz"], frequency_hz
            )
            self.assertEqual(len(diagnostics["candidate_detection"]["candidates"]), 1)

    def test_off_grid_leakage_is_grouped_as_one_candidate(self):
        diagnostics = analyze_hr_fft(_sine(1.537, 20.0), 30.0)
        self.assertEqual(len(diagnostics["candidate_detection"]["candidates"]), 1)
        self.assertGreater(len(diagnostics["candidate_detection"]["candidates"][0]["supporting_bins"]), 1)

    def test_short_signal_exposes_ambiguity_in_available_resolution(self):
        signal = _sine(1.5, duration_s=20.0, amplitude=0.3) + _sine(3.0, duration_s=20.0)
        diagnostics = analyze_hr_fft(signal, 30.0)
        self.assertTrue(diagnostics["decision"]["ambiguous"])

    def test_spectral_noise_does_not_create_unbounded_candidates(self):
        rng = np.random.default_rng(7)
        signal = _sine(1.5, 20.0) + 0.15 * rng.standard_normal(600)
        diagnostics = analyze_hr_fft(signal, 30.0)
        self.assertAlmostEqual(diagnostics["selected_peak"]["frequency_hz"], 1.5, delta=0.1)
        self.assertLessEqual(len(diagnostics["candidate_detection"]["candidates"]), 8)

    def test_diagnostics_keep_original_and_selected_candidates(self):
        diagnostics = analyze_hr_fft(_sine(1.5, 20.0) + _sine(3.0, 20.0, amplitude=2.0), 30.0)
        self.assertIn("candidate_detection", diagnostics)
        self.assertIn("relationships", diagnostics)
        self.assertIn("decision", diagnostics)
        self.assertIn("original_maximum_candidate_id", diagnostics["decision"])
        self.assertIn("selected_candidate_id", diagnostics["decision"])
        self.assertEqual(
            diagnostics["selection_method"],
            "independent spectral candidates with harmonic evidence",
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
