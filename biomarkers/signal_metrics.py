"""Signal quality metrics."""

import numpy as np
from scipy.fft import rfft, rfftfreq

from rPPG.config import HR_HIGH_HZ, HR_LOW_HZ


def compute_signal_metrics(filtered_signal, fps):
    """Calculate the unchanged temporal and spectral signal metrics."""
    std = np.std(filtered_signal)
    amplitude = np.max(filtered_signal) - np.min(filtered_signal)
    energy = np.sum(filtered_signal ** 2)
    signal = filtered_signal - np.mean(filtered_signal)
    std_signal = np.std(signal)
    if std_signal > 0:
        signal = signal / std_signal

    frequencies = rfftfreq(len(signal), d=1.0 / fps)
    fft_magnitude = (2.0 / len(signal)) * np.abs(rfft(signal))
    heart_band = (frequencies >= HR_LOW_HZ) & (frequencies <= HR_HIGH_HZ)
    fft_band = fft_magnitude[heart_band]
    fft_peak = np.max(fft_band) if len(fft_band) else 0.0

    if len(fft_band):
        peak_idx = np.argmax(fft_band)
        signal_window = np.zeros_like(fft_band, dtype=bool)
        start = max(0, peak_idx - 2)
        end = min(len(fft_band), peak_idx + 3)
        signal_window[start:end] = True
        signal_power = np.sum(fft_band[signal_window] ** 2)
        noise_band = fft_band[~signal_window]
        noise_power = np.sum(noise_band ** 2)
        total_power = signal_power + noise_power
        spectral_concentration = signal_power / total_power if total_power > 0 else 0.0
        snr = (10 * np.log10(signal_power / noise_power) if noise_power > 0
               else (np.inf if signal_power > 0 else 0.0))
        mean_noise = np.mean(noise_band) if len(noise_band) else 0.0
        peak_ratio = (fft_peak / mean_noise if mean_noise > 0
                      else (np.inf if fft_peak > 0 else 0.0))
    else:
        snr = 0.0
        peak_ratio = 0.0
        spectral_concentration = 0.0

    return {"std": std, "amplitude": amplitude, "energy": energy,
            "fft_peak": fft_peak, "snr": snr, "peak_ratio": peak_ratio,
            "spectral_concentration": spectral_concentration}
