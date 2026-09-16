"""Portable JSON, spectrum, and CSV storage for calibration runs."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path


def calibration_paths(output_dir):
    root = Path(output_dir)
    data = root / "data"
    spectra = root / "spectra"
    data.mkdir(parents=True, exist_ok=True)
    spectra.mkdir(parents=True, exist_ok=True)
    return root, data, spectra


def _json_value(value):
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def save_capture(output_dir, record):
    _, data, _ = calibration_paths(output_dir)
    path = data / f"{record['capture_id']}.json"
    path.write_text(json.dumps(_json_value(record), ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def save_spectrum(output_dir, capture_id, frequencies, magnitudes):
    _, _, spectra = calibration_paths(output_dir)
    path = spectra / f"{capture_id}.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["frequency_hz", "magnitude"])
        writer.writeheader()
        writer.writerows({"frequency_hz": float(f), "magnitude": float(m)} for f, m in zip(frequencies, magnitudes))
    return path


def flatten_record(record):
    row = {key: record.get(key) for key in (
        "capture_id", "timestamp", "duration_seconds", "fps", "frame_count", "effective_samples",
        "reference_hr_bpm", "final_hr_bpm", "absolute_error_bpm", "relative_error_percent",
    )}
    for section in ("quality", "lighting", "fft"):
        for key, value in record.get(section, {}).items():
            row[f"{section}_{key}"] = value
    for roi, methods in record.get("roi_algorithm_results", {}).items():
        for algorithm, result in methods.items():
            prefix = f"{roi}_{algorithm}"
            row[prefix] = result.get("hr_bpm")
            row[f"{prefix}_absolute_error_bpm"] = result.get("absolute_error_bpm")
            row[f"{prefix}_relative_error_percent"] = result.get("relative_error_percent")
    return _json_value(row)


def write_csv(path, rows):
    fields = sorted({field for row in rows for field in row})
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
