"""Reference comparisons and aggregation for calibration observations."""

from __future__ import annotations

import math
from collections import defaultdict


def valid_reference(value):
    """Return a positive finite external reference, or ``None``."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) and value > 0 else None


def absolute_error(estimated_hr, reference_hr):
    if valid_reference(reference_hr) is None or valid_reference(estimated_hr) is None:
        return None
    return abs(float(estimated_hr) - float(reference_hr))


def relative_error_percent(estimated_hr, reference_hr):
    error = absolute_error(estimated_hr, reference_hr)
    reference = valid_reference(reference_hr)
    return None if error is None or reference is None else 100.0 * error / reference


def error_summary(errors):
    """Summarize signed errors while ignoring unavailable observations."""
    values = [float(value) for value in errors if value is not None and math.isfinite(float(value))]
    if not values:
        return {"mae_bpm": None, "rmse_bpm": None, "mean_error_bpm": None,
                "std_error_bpm": None, "n_valid": 0}
    mean = sum(values) / len(values)
    return {
        "mae_bpm": sum(abs(value) for value in values) / len(values),
        "rmse_bpm": math.sqrt(sum(value ** 2 for value in values) / len(values)),
        "mean_error_bpm": mean,
        "std_error_bpm": math.sqrt(sum((value - mean) ** 2 for value in values) / len(values)),
        "n_valid": len(values),
    }


def aggregate_benchmarks(records):
    """Aggregate final, ROI, algorithm, and ROI×algorithm signed errors."""
    groups = defaultdict(list)
    for record in records:
        reference = record.get("reference_hr_bpm")
        if valid_reference(reference) is None:
            continue
        final = record.get("final_hr_bpm")
        if valid_reference(final) is not None:
            groups[("final", "FINAL")].append(float(final) - float(reference))
        for roi, methods in record.get("roi_algorithm_results", {}).items():
            for algorithm, result in methods.items():
                hr = result.get("hr_bpm")
                if valid_reference(hr) is None:
                    continue
                error = float(hr) - float(reference)
                groups[("algorithm", algorithm)].append(error)
                groups[("roi", roi)].append(error)
                groups[("roi_algorithm", f"{roi}_{algorithm}")].append(error)
    return [
        {"group_type": kind, "group": name, **error_summary(errors)}
        for (kind, name), errors in sorted(groups.items())
    ]
