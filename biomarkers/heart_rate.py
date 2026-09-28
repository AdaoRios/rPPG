"""Heart-rate estimation and non-invasive FFT diagnostics."""

from __future__ import annotations

import numpy as np
from scipy import signal as scipy_signal
from scipy.fft import rfft, rfftfreq

from rPPG.config import HR_HIGH_HZ, HR_LOW_HZ


# Experimental spectral-selection heuristics. These values are operational
# thresholds for validation, not physiological limits.
_MIN_CANDIDATE_RELATIVE_MAGNITUDE = 0.05
_HARMONIC_MAX_ERROR_RESOLUTION_UNITS = 2.0
_HARMONIC_SUPPORTED_MAX_ERROR_RESOLUTION_UNITS = 1.0
_HARMONIC_RECONSIDERATION_MAGNITUDE_RATIO = 0.4
_HARMONIC_RECONSIDERATION_PROMINENCE_RATIO = 0.25
_HARMONIC_AMBIGUITY_MAGNITUDE_RATIO = 0.2
_HARMONIC_AMBIGUITY_PROMINENCE_RATIO = 0.15
_MIN_INDEPENDENT_SEPARATION_BINS = 2.0
_LOBE_WIDTH_GROUPING_RATIO = 0.5


def _candidate_dict(
    candidate_id,
    frequencies,
    magnitudes,
    prominences,
    widths,
    left,
    right,
    peak_index,
    band_power,
    grouped_peaks,
):
    local_slice = slice(left, right + 1)
    local_energy = float(np.sum(magnitudes[local_slice] ** 2))
    magnitude = float(magnitudes[peak_index])
    frequency = float(frequencies[peak_index])
    return {
        "id": candidate_id,
        "dominant_bin": int(peak_index),
        "frequency_hz": frequency,
        "frequency_bpm": float(frequency * 60.0),
        "magnitude": magnitude,
        "power": float(magnitude ** 2),
        "prominence": float(prominences[peak_index]),
        "width_bins": float(widths[peak_index]),
        "width_hz": float(widths[peak_index] * (frequencies[1] - frequencies[0])),
        "local_energy": local_energy,
        "local_concentration": local_energy / band_power if band_power > 0 else 0.0,
        "supporting_bins": [int(index) for index in range(left, right + 1)],
        "grouped_peak_bins": [int(index) for index in grouped_peaks],
    }


def _detect_spectral_candidates(frequencies, magnitudes, valid_indices):
    """Detect independent spectral concentrations rather than FFT bins."""
    band_frequencies = frequencies[valid_indices]
    band_magnitudes = magnitudes[valid_indices]
    band_power = float(np.sum(band_magnitudes ** 2))
    maximum = float(np.max(band_magnitudes))
    resolution_hz = float(frequencies[1] - frequencies[0])

    peak_indices, properties = scipy_signal.find_peaks(
        band_magnitudes,
        prominence=max(
            maximum * _MIN_CANDIDATE_RELATIVE_MAGNITUDE,
            np.finfo(float).eps,
        ),
    )
    # A band edge can contain the only visible cardiac peak.
    edge_indices = []
    if len(band_magnitudes) > 1:
        if band_magnitudes[0] >= band_magnitudes[1]:
            edge_indices.append(0)
        if band_magnitudes[-1] >= band_magnitudes[-2]:
            edge_indices.append(len(band_magnitudes) - 1)
    peak_indices = np.unique(np.concatenate((peak_indices, edge_indices))).astype(int)
    peak_indices = peak_indices[
        band_magnitudes[peak_indices]
        >= maximum * _MIN_CANDIDATE_RELATIVE_MAGNITUDE
    ]
    if not len(peak_indices):
        peak_indices = np.array([int(np.argmax(band_magnitudes))])

    prominence_values = np.zeros(len(band_magnitudes), dtype=np.float64)
    width_values = np.ones(len(band_magnitudes), dtype=np.float64)
    interior_peaks = peak_indices[
        (peak_indices > 0) & (peak_indices < len(band_magnitudes) - 1)
    ]
    if len(interior_peaks):
        prominence_values[interior_peaks] = scipy_signal.peak_prominences(
            band_magnitudes, interior_peaks
        )[0]
        width_values[interior_peaks] = scipy_signal.peak_widths(
            band_magnitudes, interior_peaks, rel_height=0.5
        )[0]

    # Peaks separated by at most two bins, or by less than half their width,
    # are one unresolved lobulation. The strongest bin represents the group.
    groups = []
    for peak_index in peak_indices:
        if not groups:
            groups.append([int(peak_index)])
            continue
        previous = groups[-1][-1]
        minimum_separation = max(
            _MIN_INDEPENDENT_SEPARATION_BINS,
            _LOBE_WIDTH_GROUPING_RATIO
            * max(width_values[previous], width_values[peak_index]),
        )
        if peak_index - previous <= minimum_separation:
            groups[-1].append(int(peak_index))
        else:
            groups.append([int(peak_index)])

    candidates = []
    grouped_bins = []
    for candidate_number, group in enumerate(groups, start=1):
        dominant = max(group, key=lambda index: band_magnitudes[index])
        left = max(0, min(group) - 1)
        right = min(len(band_magnitudes) - 1, max(group) + 1)
        candidate = _candidate_dict(
            f"candidate_{candidate_number}",
            band_frequencies,
            band_magnitudes,
            prominence_values,
            width_values,
            left,
            right,
            dominant,
            band_power,
            group,
        )
        candidate["dominant_bin"] = int(valid_indices[dominant])
        candidate["supporting_bins"] = [
            int(valid_indices[index]) for index in range(left, right + 1)
        ]
        candidate["grouped_peak_bins"] = [int(valid_indices[index]) for index in group]
        candidates.append(candidate)
        grouped_bins.append(candidate["grouped_peak_bins"])

    candidates.sort(key=lambda candidate: candidate["magnitude"], reverse=True)
    for index, candidate in enumerate(candidates, start=1):
        candidate["id"] = f"candidate_{index}"
    return candidates, grouped_bins, resolution_hz


def _find_harmonic_relationships(candidates, resolution_hz):
    relationships = []
    for lower_index, lower in enumerate(candidates):
        for upper in candidates[lower_index + 1:]:
            first, second = sorted((lower, upper), key=lambda item: item["frequency_hz"])
            if first["frequency_hz"] <= 0:
                continue
            error_hz = abs(second["frequency_hz"] - 2.0 * first["frequency_hz"])
            error_units = error_hz / resolution_hz if resolution_hz > 0 else float("inf")
            is_harmonic = error_units <= _HARMONIC_MAX_ERROR_RESOLUTION_UNITS
            relationship = {
                "lower_candidate_id": first["id"],
                "upper_candidate_id": second["id"],
                "frequency_ratio": second["frequency_hz"] / first["frequency_hz"],
                "absolute_error_hz": error_hz,
                "error_in_resolution_units": error_units,
                "harmonic_relationship": is_harmonic,
                "relationship": "harmonic_2_to_1" if is_harmonic else "unrelated",
                "strength": "supported" if error_units <= _HARMONIC_SUPPORTED_MAX_ERROR_RESOLUTION_UNITS else (
                    "weak" if is_harmonic else "none"
                ),
            }
            relationships.append(relationship)
    return relationships


def _select_candidate(candidates, relationships):
    original = candidates[0]
    selected = original
    reconsidered = False
    ambiguous = False
    reason = "largest independent candidate retained"
    harmonic_support = []
    harmonic_detected = False
    harmonic_supported = False

    for relationship in relationships:
        if not relationship["harmonic_relationship"]:
            continue
        original_is_upper = relationship["upper_candidate_id"] == original["id"]
        original_is_lower = relationship["lower_candidate_id"] == original["id"]
        if not (original_is_upper or original_is_lower):
            continue
        harmonic_detected = True
        harmonic_supported = harmonic_supported or relationship["strength"] == "supported"
        if original_is_lower:
            harmonic_support.append(relationship)
            continue
        lower = next(
            candidate for candidate in candidates
            if candidate["id"] == relationship["lower_candidate_id"]
        )
        magnitude_ratio = lower["magnitude"] / original["magnitude"] if original["magnitude"] else 0.0
        prominence_ratio = lower["prominence"] / original["prominence"] if original["prominence"] else 0.0
        relationship["fundamental_magnitude_ratio"] = magnitude_ratio
        relationship["fundamental_prominence_ratio"] = prominence_ratio
        if (
            magnitude_ratio >= _HARMONIC_RECONSIDERATION_MAGNITUDE_RATIO
            and prominence_ratio >= _HARMONIC_RECONSIDERATION_PROMINENCE_RATIO
        ):
            # A lower-frequency candidate with enough support proves that a
            # harmonic interpretation is plausible, but not which frequency
            # is physiological. Retain the dominant peak and expose uncertainty.
            ambiguous = True
            reason = (
                "dominant candidate has a supported approximately 2:1 harmonic; "
                "fundamental interpretation is ambiguous"
            )
            harmonic_support.append(relationship)
        elif (
            magnitude_ratio >= _HARMONIC_AMBIGUITY_MAGNITUDE_RATIO
            or prominence_ratio >= _HARMONIC_AMBIGUITY_PROMINENCE_RATIO
        ):
            ambiguous = True
            reason = "approximately 2:1 candidates have insufficiently decisive support"
            harmonic_support.append(relationship)

    return selected, original, {
        "original_maximum_candidate_id": original["id"],
        "selected_candidate_id": selected["id"],
        "harmonic_reconsideration": reconsidered,
        "ambiguous": ambiguous,
        "harmonic_detected": harmonic_detected,
        "harmonic_supported": harmonic_supported,
        "reason": reason,
        "harmonic_support": harmonic_support,
    }


def analyze_hr_fft(filtered_signal, fps):
    """Estimate HR using independent spectral candidates and harmonic evidence."""
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

    candidates, grouped_bins, resolution_hz = _detect_spectral_candidates(
        frequencies, fft_magnitude, valid_indices
    )
    relationships = _find_harmonic_relationships(candidates, resolution_hz)
    selected, original, decision = _select_candidate(candidates, relationships)
    selected_index = selected["dominant_bin"]
    selected_frequency = selected["frequency_hz"]
    selected_magnitude = selected["magnitude"]
    secondary = [candidate for candidate in candidates if candidate["id"] != selected["id"]]
    second = secondary[0] if secondary else None
    second_frequency = None if second is None else second["frequency_hz"]
    second_magnitude = None if second is None else second["magnitude"]
    primary_to_second_ratio = (
        selected_magnitude / second_magnitude
        if second_magnitude and second_magnitude > 0 else None
    )
    peak_separation_hz = (
        abs(selected_frequency - second_frequency)
        if second_frequency is not None else None
    )

    cardiac_band_power = float(np.sum(fft_magnitude[valid] ** 2))
    total_spectral_power = float(np.sum(fft_magnitude ** 2))
    return {
        "samples": n_samples,
        "fps": float(fps),
        "duration_s": float(n_samples / fps),
        "frequency_resolution_hz": float(fps / n_samples),
        "frequency_resolution_type": "nominal_fft_bin_spacing",
        "bpm_resolution": float(60.0 * fps / n_samples),
        "cardiac_band_hz": (HR_LOW_HZ, HR_HIGH_HZ),
        "window": "Hann",
        "detrending": "none in HR FFT; input is the final band-pass-filtered signal",
        "zero_padding": False,
        "peak_interpolation": False,
        "selection_method": "independent spectral candidates with harmonic evidence",
        "selected_peak": {
            "fft_bin": selected_index,
            "frequency_hz": selected_frequency,
            "hr_bpm": float(selected_frequency * 60.0),
            "magnitude": selected_magnitude,
            "power": float(selected_magnitude ** 2),
        },
        "second_local_peak": {
            "semantic": "second independent spectral candidate by magnitude; not necessarily a raw local maximum",
            "candidate_id": None if second is None else second["id"],
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
            "ambiguous supported harmonic relationship"
            if decision["ambiguous"] and decision["harmonic_supported"]
            else "ambiguous harmonic relationship"
            if decision["ambiguous"]
            else "harmonic relationship detected"
            if decision["harmonic_detected"]
            else "no decisive harmonic relationship"
        ),
        "candidate_detection": {
            "resolution_hz": resolution_hz,
            "resolution_type": "nominal_fft_bin_spacing",
            "minimum_independent_separation_bins": _MIN_INDEPENDENT_SEPARATION_BINS,
            "candidates": candidates,
            "grouped_peak_bins": grouped_bins,
        },
        "relationships": relationships,
        "decision": decision,
    }


def compute_hr_fft(filtered_signal, fps):
    """Estimate HR from the documented maximum FFT peak in the cardiac band."""
    return analyze_hr_fft(filtered_signal, fps)["selected_peak"]["hr_bpm"]


def classify_confidence(decision, signal_metrics):
    """Classify result confidence from explicit spectral and quality evidence."""
    snr = signal_metrics.get("snr")
    concentration = signal_metrics.get("spectral_concentration")
    if (
        decision["ambiguous"]
        or snr is None
        or concentration is None
        or not np.isfinite(snr)
        or not np.isfinite(concentration)
        or snr <= 0.0
    ):
        return "low"
    if concentration >= 0.5:
        return "high"
    return "medium"
