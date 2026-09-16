"""CLI orchestration for repeated, isolated calibration captures."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from rPPG.analysis.analyze_video import analyze_video
from rPPG.capture.capture_video import capture_video
from rPPG.config import LIGHTING_BRIGHT_PIXEL_CHANNEL, ROI_POINTS
from rPPG.extractors.combine import METHOD_ORDER
from rPPG.calibration.metrics import (
    absolute_error, aggregate_benchmarks, relative_error_percent, valid_reference,
)
from rPPG.calibration.report import write_report
from rPPG.calibration.storage import calibration_paths, flatten_record, save_capture, save_spectrum, write_csv


def _ask_reference(input_fn=input):
    value = input_fn("FC de referência do Apple Watch em bpm (ENTER para ausente): ").strip()
    return valid_reference(value) if value else None


def record_from_result(capture_id, requested_duration, result, reference_hr, timestamp=None):
    """Convert existing pipeline output into a self-contained observation."""
    reference = valid_reference(reference_hr)
    audit = result.audit or {}
    frame = audit.get("frame_collection", {})
    lighting = frame.get("lighting", {})
    # Per-frame lighting is intentionally unavailable in the production audit;
    # retaining its summary/provenance is preferable to inventing a threshold.
    roi_benchmarks = audit.get("fusion", {}).get("roi_benchmark", {})
    combinations = {}
    for roi in ROI_POINTS:
        methods = roi_benchmarks.get(roi, {})
        combinations[roi.upper()] = {}
        for algorithm in METHOD_ORDER:
            values = methods.get(algorithm.upper(), {})
            hr = values.get("hr_bpm") if isinstance(values, dict) else None
            combinations[roi.upper()][algorithm.upper()] = {
                "hr_bpm": hr,
                "absolute_error_bpm": absolute_error(hr, reference),
                "relative_error_percent": relative_error_percent(hr, reference),
            }
    fft = audit.get("hr_fft", {})
    selected = fft.get("selected_peak", {})
    second = fft.get("second_local_peak", {})
    quality = result.signal_metrics or {}
    return {
        "capture_id": capture_id,
        "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
        "duration_seconds": float(requested_duration),
        "fps": float(result.fps),
        "frame_count": frame.get("frames_read"),
        "effective_samples": result.valid_frames,
        "reference_hr_bpm": reference,
        "reference_role": "external Apple Watch reference; not clinical ground truth",
        "final_hr_bpm": float(result.heart_rate),
        "absolute_error_bpm": absolute_error(result.heart_rate, reference),
        "relative_error_percent": relative_error_percent(result.heart_rate, reference),
        "quality": {key: quality.get(key) for key in ("snr", "spectral_concentration", "fft_peak", "amplitude", "std", "energy")},
        "lighting": {
            "bright_pixel_ratio": lighting.get("bright_pixel_ratio"),
            "mean_luminance": lighting.get("mean_luminance"),
            "dark_pixel_ratio": lighting.get("dark_pixel_ratio"),
            "illumination_uniformity": lighting.get("illumination_uniformity"),
            "bright_pixel_threshold": lighting.get("bright_pixel_channel_threshold", LIGHTING_BRIGHT_PIXEL_CHANNEL),
            "note": "observational metrics only; 0.784 is not a quality gate",
        },
        "fft": {
            "frequency_resolution_hz": fft.get("frequency_resolution_hz"), "bpm_resolution": fft.get("bpm_resolution"),
            "selected_frequency_hz": selected.get("frequency_hz"), "selected_bpm": selected.get("hr_bpm"),
            "second_peak_frequency_hz": second.get("frequency_hz"), "second_peak_bpm": second.get("frequency_hz") * 60 if second.get("frequency_hz") is not None else None,
            "peak_ratio": second.get("primary_to_second_ratio"),
        },
        "roi_algorithm_results": combinations,
        "audit": audit,
    }


def run_calibration(captures=10, duration=20.0, output_dir="calibration", camera=0,
                    capture_fn=capture_video, analyze_fn=analyze_video, input_fn=input):
    """Capture/analyze repeatedly; injected functions keep tests webcam-free."""
    if captures < 1 or duration <= 0:
        raise ValueError("captures deve ser >= 1 e duration deve ser positiva")
    root, _, _ = calibration_paths(output_dir)
    records = []
    for index in range(1, captures + 1):
        capture_id = f"capture_{index:03d}"
        print(f"\n=== Calibration {index}/{captures}: {capture_id} ===")
        video_path = capture_fn(camera, duration)
        result = analyze_fn(video_path)
        reference = _ask_reference(input_fn)
        record = record_from_result(capture_id, duration, result, reference)
        save_capture(root, record)
        spectrum = result.spectral_data or {}
        if spectrum:
            save_spectrum(root, capture_id, spectrum["frequency_hz"], spectrum["magnitude"])
        records.append(record)
    write_csv(root / "results.csv", [flatten_record(record) for record in records])
    summary = aggregate_benchmarks(records)
    write_csv(root / "summary.csv", summary)
    write_report(root, records, summary)
    return records


def main(argv=None):
    parser = argparse.ArgumentParser(description="Experimental rPPG calibration; never changes production weights.")
    parser.add_argument("--captures", type=int, default=10)
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--output-dir", default="calibration")
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args(argv)
    return run_calibration(args.captures, args.duration, Path(args.output_dir), args.camera)
