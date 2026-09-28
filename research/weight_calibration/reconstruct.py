"""Rebuild the final rPPG signal from exported intermediates under new weights.

The reconstruction mirrors production ``extractors.combine`` exactly:

1. per ROI: ``sum(normalize(method_weights) * method_signal)`` over the
   methods present (production normalizes over surviving components);
2. cross-ROI common-prefix alignment;
3. ``sum(normalize(roi_weights) * roi_signal)``;
4. the sole final band-pass filter (0.7-4.0 Hz) and the frozen
   ``analyze_hr_fft`` selection with the unchanged harmonic policy.

No new HR-selection rule is introduced here.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from rPPG.biomarkers.heart_rate import analyze_hr_fft, classify_confidence
from rPPG.biomarkers.signal_metrics import compute_signal_metrics
from rPPG.config import HR_HIGH_HZ, HR_LOW_HZ
from rPPG.extractors.combine import _align_to_common_prefix, weighted_signal_combination
from rPPG.preprocessing.filters import bandpass_filter


def load_capture(capture_dir) -> dict:
    """Load one exported capture (signals.npz + metadata.json)."""
    capture_dir = Path(capture_dir)
    with open(capture_dir / "metadata.json", "r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    with np.load(capture_dir / "signals.npz") as archive:
        signals = {key: np.asarray(archive[key], dtype=np.float64) for key in archive.files}
    components = {}
    for roi_name, methods in metadata["components"].items():
        components[roi_name] = {
            method_name: signals[f"{roi_name}.{method_name}"]
            for method_name in methods
        }
    return {
        "capture_id": metadata["capture_id"],
        "fps": float(metadata["fps"]),
        "metadata": metadata,
        "components": components,
        "reference_hr_bpm": metadata.get("reference_hr_bpm"),
        "recorded_final_hr_bpm": metadata.get("recorded_final_hr_bpm"),
    }


def reconstruct(capture: dict, method_weights, roi_weights) -> dict:
    """Return the frozen-pipeline HR result for one weight configuration."""
    combined_per_roi = {}
    for roi_name, method_signals in capture["components"].items():
        combined_per_roi[roi_name] = weighted_signal_combination(
            method_signals, method_weights,
            f"algorithm fusion for ROI '{roi_name}'",
        )
    aligned_rois, _ = _align_to_common_prefix(combined_per_roi)
    final_signal = weighted_signal_combination(aligned_rois, roi_weights, "ROI fusion")
    filtered = bandpass_filter(
        final_signal, capture["fps"], low_hz=HR_LOW_HZ, high_hz=HR_HIGH_HZ
    )
    hr_fft = analyze_hr_fft(filtered, capture["fps"])
    metrics = compute_signal_metrics(filtered, capture["fps"])
    decision = hr_fft["decision"]
    confidence = classify_confidence(decision, metrics)
    return {
        "capture_id": capture["capture_id"],
        "hr_bpm": float(hr_fft["selected_peak"]["hr_bpm"]),
        "reference_hr_bpm": capture["reference_hr_bpm"],
        "recorded_final_hr_bpm": capture["recorded_final_hr_bpm"],
        "confidence": confidence,
        "ambiguous": bool(decision["ambiguous"]),
        "harmonic_detected": bool(decision["harmonic_detected"]),
        "harmonic_supported": bool(decision["harmonic_supported"]),
        "decision_reason": decision["reason"],
        "snr": metrics.get("snr"),
        "spectral_concentration": metrics.get("spectral_concentration"),
        "final_signal_samples": len(final_signal),
    }