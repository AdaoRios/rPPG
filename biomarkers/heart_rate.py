"""Heart-rate estimation and non-invasive FFT diagnostics."""

from __future__ import annotations

import numpy as np
from scipy import signal as scipy_signal
from scipy.fft import rfft, rfftfreq

from rPPG.config import HR_HIGH_HZ, HR_LOW_HZ


def analyze_hr_fft(filtered_signal, fps):
    """Describe the existing HR FFT decision without changing it.

    The production choice remains the greatest FFT magnitude inside the
    configured cardiac band.  The returned diagnostics make the spectral
    resolution, selected bin, and a distinct secondary local peak visible; no
    zero-padding, peak interpolation, or reference-HR guidance is used.
    """
    signal = np.atleast_1d(np.asarray(filtered_signal, dtype=np.float64).squeeze())
    if signal.ndim != 1 or len(signal) < 2:
        raise RuntimeError("Não foi possível estimar HR: sinal deve ter ao menos duas amostras.")
    if not np.isfinite(signal).all():
        raise RuntimeError("Não foi possível estimar HR: sinal contém valores não finitos.")
    if np.std(signal) <= np.finfo(float).eps:
        raise RuntimeError("Não foi possível estimar HR: sinal sem variabilidade.")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps deve ser positivo e finito.")

    n_samples = len(signal)
    window = np.hanning(n_samples)
    windowed = signal * window
    frequencies = rfftfreq(n_samples, d=1.0 / fps)
    fft_magnitude = np.abs(rfft(windowed))
    valid = (frequencies >= HR_LOW_HZ) & (frequencies <= HR_HIGH_HZ)
    valid_indices = np.flatnonzero(valid)
    if len(valid_indices) == 0:
        raise RuntimeError("Não foi possível estimar HR: faixa espectral vazia.")

    selected_local_index = int(np.argmax(fft_magnitude[valid]))
    selected_index = int(valid_indices[selected_local_index])
    selected_frequency = float(frequencies[selected_index])
    selected_magnitude = float(fft_magnitude[selected_index])

    # A second *local* maximum avoids reporting an adjacent main-lobe FFT bin
    # as independent evidence. It remains diagnostic only.
    local_peak_indices, _ = scipy_signal.find_peaks(fft_magnitude[valid])
    distinct_local_peaks = [
        int(valid_indices[index]) for index in local_peak_indices
        if int(valid_indices[index]) != selected_index
    ]
    if distinct_local_peaks:
        second_index = max(distinct_local_peaks, key=lambda index: fft_magnitude[index])
        second_frequency = float(frequencies[second_index])
        second_magnitude = float(fft_magnitude[second_index])
        primary_to_second_ratio = (
            float(selected_magnitude / second_magnitude)
            if second_magnitude > 0 else float("inf")
        )
        peak_separation_hz = abs(selected_frequency - second_frequency)
    else:
        second_frequency = None
        second_magnitude = None
        primary_to_second_ratio = None
        peak_separation_hz = None

    cardiac_band_power = float(np.sum(fft_magnitude[valid] ** 2))
    total_spectral_power = float(np.sum(fft_magnitude ** 2))
    return {
        "samples": n_samples,
        "fps": float(fps),
        "duration_s": float(n_samples / fps),
        "frequency_resolution_hz": float(fps / n_samples),
        "bpm_resolution": float(60.0 * fps / n_samples),
        "cardiac_band_hz": (HR_LOW_HZ, HR_HIGH_HZ),
        "window": "Hann",
        "detrending": "none in HR FFT; input is the final band-pass-filtered signal",
        "zero_padding": False,
        "peak_interpolation": False,
        "selection_method": "maximum FFT magnitude within configured cardiac band",
        "selected_peak": {
            "fft_bin": selected_index,
            "frequency_hz": selected_frequency,
            "hr_bpm": float(selected_frequency * 60.0),
            "magnitude": selected_magnitude,
            "power": float(selected_magnitude ** 2),
        },
        "second_local_peak": {
            "frequency_hz": second_frequency,
            "magnitude": second_magnitude,
            "primary_to_second_ratio": primary_to_second_ratio,
            "separation_hz": peak_separation_hz,
            "separation_bpm": None if peak_separation_hz is None else float(peak_separation_hz * 60.0),
        },
        "cardiac_band_power": cardiac_band_power,
        "total_spectral_power": total_spectral_power,
        "cardiac_band_power_ratio": (
            cardiac_band_power / total_spectral_power if total_spectral_power > 0 else 0.0
        ),
        "harmonic_assessment": (
            "not resolved by peak selection; inspect primary/secondary peaks and validate against repeated references"
        ),
    }


def compute_hr_fft(filtered_signal, fps):
    """Estimate HR from the documented maximum FFT peak in the cardiac band."""
    return analyze_hr_fft(filtered_signal, fps)["selected_peak"]["hr_bpm"]
