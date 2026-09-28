"""Traceable algorithm and ROI signal combination for the rPPG pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from rPPG.biomarkers.heart_rate import compute_hr_fft
from rPPG.biomarkers.signal_metrics import compute_signal_metrics
from rPPG.config import METHOD_WEIGHTS, ROI_WEIGHTS
from rPPG.extractors.chrom import chrom_algorithm
from rPPG.extractors.green import green_algorithm
from rPPG.extractors.ica import ica_algorithm
from rPPG.extractors.pos import pos_algorithm
from rPPG.preprocessing.filters import bandpass_filter, moving_average_smooth
# Presentation lives outside the core: plotting (matplotlib) and console
# benchmark printing are imported lazily so the core never hard-depends on
# rPPG.reports.plots and can run headless (e.g. future programmatic callers).


METHOD_ORDER = ("chrom", "pos", "green", "ica")


@dataclass(frozen=True)
class SignalFusionResult:
    """The unfiltered fused signal and compact provenance for one analysis."""

    signal: np.ndarray
    audit: dict


def normalize_weights(weights: Mapping[str, float], component_names, label: str) -> dict[str, float]:
    """Validate non-negative finite weights and return weights summing to one.

    Only listed components are considered. This lets an explicitly recorded
    invalid component be excluded while preserving the relative configured
    weights of valid components. It never chooses components by quality.
    """
    names = tuple(component_names)
    if not names:
        raise ValueError(f"{label} requires at least one component")
    missing = [name for name in names if name not in weights]
    if missing:
        raise ValueError(f"{label} has no configured weight for: {', '.join(missing)}")
    values = np.asarray([weights[name] for name in names], dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"{label} weights must be finite")
    if np.any(values < 0):
        raise ValueError(f"{label} weights must be non-negative")
    weight_sum = float(np.sum(values))
    if weight_sum <= 0:
        raise ValueError(f"{label} weights must have a positive sum")
    return {name: float(value / weight_sum) for name, value in zip(names, values)}


def weighted_signal_combination(
    signals: Mapping[str, np.ndarray], weights: Mapping[str, float], label: str
) -> np.ndarray:
    """Return ``sum(weight * signal)`` after strict compatibility validation."""
    names = tuple(signals)
    normalized_weights = normalize_weights(weights, names, label)
    arrays = {}
    for name in names:
        array = np.atleast_1d(np.asarray(signals[name], dtype=np.float64).squeeze())
        if array.ndim != 1 or len(array) == 0:
            raise ValueError(f"{label} signal '{name}' must be a non-empty 1-D array")
        if not np.isfinite(array).all():
            raise ValueError(f"{label} signal '{name}' contains non-finite values")
        arrays[name] = array
    lengths = {len(array) for array in arrays.values()}
    if len(lengths) != 1:
        details = ", ".join(f"{name}={len(array)}" for name, array in arrays.items())
        raise ValueError(f"{label} signals have incompatible lengths: {details}")
    return np.sum(
        np.vstack([normalized_weights[name] * arrays[name] for name in names]), axis=0
    )


def _align_to_common_prefix(
    signals: Mapping[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
    """Explicitly align signals to their shared initial timestamps.

    CHROM's overlap-add implementation may produce a shorter tail than POS,
    GREEN, and ICA. Fusion therefore uses the common initial duration rather
    than letting a numerical routine truncate implicitly. The exact loss is
    kept in the audit record.
    """
    arrays = {
        name: np.atleast_1d(np.asarray(signal, dtype=np.float64).squeeze())
        for name, signal in signals.items()
    }
    if not arrays:
        raise ValueError("cannot align an empty signal collection")
    if any(array.ndim != 1 or len(array) == 0 for array in arrays.values()):
        raise ValueError("all signals must be non-empty 1-D arrays before alignment")
    common_length = min(len(array) for array in arrays.values())
    aligned = {name: array[:common_length] for name, array in arrays.items()}
    metadata = {
        name: {
            "original_samples": len(array),
            "aligned_samples": common_length,
            "tail_samples_discarded_for_alignment": len(array) - common_length,
        }
        for name, array in arrays.items()
    }
    return aligned, metadata


def _zscore(signal: np.ndarray, component_name: str) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float64)
    standard_deviation = float(np.std(signal))
    if not np.isfinite(standard_deviation) or standard_deviation <= np.finfo(float).eps:
        raise ValueError(f"{component_name} has zero or invalid variance")
    return (signal - np.mean(signal)) / standard_deviation


def _benchmark_metrics(signal: np.ndarray, fps: float) -> dict:
    """Measure a benchmark signal without letting it influence fusion."""
    filtered = bandpass_filter(signal, fps)
    metrics = compute_signal_metrics(filtered, fps)
    metrics["hr_bpm"] = compute_hr_fft(filtered, fps)
    return metrics


def _status(name, status, reason=None, **details) -> dict:
    return {"name": name, "status": status, "reason": reason, **details}


def _print_signal_pipeline_audit(audit: dict) -> None:
    """Emit provenance summaries only; never print signal arrays."""
    print("\n================ SIGNAL PIPELINE AUDIT ================")
    print(
        f"ROIs configured: {len(audit['configured_rois'])}; "
        f"valid: {len(audit['valid_rois'])}"
    )
    print(f"Algorithms configured: {', '.join(name.upper() for name in METHOD_ORDER)}")
    print("ROI weights (configured -> effective):")
    for name in audit["configured_rois"]:
        configured = ROI_WEIGHTS.get(name)
        effective = audit["effective_roi_weights"].get(name)
        effective_text = "EXCLUDED" if effective is None else f"{effective:.6f}"
        print(f"  {name.upper()}: {configured!s} -> {effective_text}")
    print("Algorithm weights (configured):")
    print("  " + ", ".join(
        f"{name.upper()}={METHOD_WEIGHTS[name]:.6f}" for name in METHOD_ORDER
    ))

    valid_components = []
    print("Algorithm weights (effective by valid ROI):")
    for roi_name in audit["valid_rois"]:
        methods = audit["roi_records"][roi_name]["methods"]
        effective = ", ".join(
            f"{method_name.upper()}={record['effective_weight']:.6f}"
            for method_name, record in methods.items()
            if record["status"] == "VALID"
        )
        print(f"  {roi_name.upper()}: {effective}")
        valid_components.extend(
            f"{roi_name.upper()}/{method_name.upper()}"
            for method_name, record in methods.items()
            if record["status"] == "VALID"
        )
    print("Valid components: " + ", ".join(valid_components))

    excluded = audit["excluded_components"]
    if excluded:
        print("Excluded components:")
        for component in excluded:
            location = component.get("roi", "GLOBAL").upper()
            print(f"  {location}/{component['name'].upper()}: {component['reason']}")
    else:
        print("Excluded components: none")

    print("Algorithm combination: OK (per ROI, explicit normalized weights)")
    print("ROI combination: OK (explicit normalized weights)")
    final = audit["final_signal"]
    print(
        "Final signal: "
        f"source={final['source']}; components={', '.join(final['components'])}; "
        f"samples={final['samples']}; weight_sum={final['weight_sum']:.3f}"
    )
    print("Final HR/HRV/metrics source: band-pass filtered weighted final signal")
    print("=========================================================\n")


def combine_roi_and_methods(
    roi_signals, fps, debug=False, return_audit=False, reference_hr=None,
    verbose=True,
):
    """Fuse algorithms per ROI, then valid ROIs by explicitly validated weight.

    The returned signal is intentionally *unfiltered*. ``analyze_video``
    applies the single final cardiac band-pass filter used consistently by HR,
    HRV, and final signal metrics. Benchmarks are observational and never
    select the final method or ROI. ``verbose`` only controls console
    presentation; it never changes the signal or the audit contents.
    """
    if fps <= 0 or not np.isfinite(fps):
        raise ValueError("fps must be a positive finite value")
    if not roi_signals:
        raise ValueError("at least one ROI signal is required")

    configured_rois = tuple(roi_signals)
    missing_weights = [name for name in configured_rois if name not in ROI_WEIGHTS]
    if missing_weights:
        raise ValueError(f"ROI weights missing for: {', '.join(missing_weights)}")

    roi_benchmark = {}
    method_signals_per_roi = {}
    combined_per_roi = {}
    roi_records = {}
    excluded_components = []

    for roi_name in configured_rois:
        rgb_raw = np.asarray(roi_signals[roi_name], dtype=np.float64)
        if rgb_raw.ndim != 2 or rgb_raw.shape[1] != 3 or len(rgb_raw) == 0:
            record = _status("roi", "EXCLUDED", "invalid_rgb_signal")
            excluded_components.append({"roi": roi_name, **record})
            roi_records[roi_name] = _status(roi_name, "EXCLUDED", "invalid_rgb_signal")
            continue
        if not np.isfinite(rgb_raw).all():
            record = _status("roi", "EXCLUDED", "non_finite_rgb_signal")
            excluded_components.append({"roi": roi_name, **record})
            roi_records[roi_name] = _status(roi_name, "EXCLUDED", "non_finite_rgb_signal")
            continue

        rgb_smooth = moving_average_smooth(rgb_raw, window=3)
        extractors = {
            "chrom": lambda: chrom_algorithm(rgb_smooth, fps),
            "pos": lambda: pos_algorithm(rgb_smooth, fps),
            "green": lambda: green_algorithm(rgb_smooth),
            "ica": lambda: ica_algorithm(rgb_smooth, fps, return_metadata=True),
        }
        raw_signals = {}
        method_records = {}
        for method_name in METHOD_ORDER:
            try:
                result = extractors[method_name]()
                signal, metadata = result if method_name == "ica" else (result, {})
                array = np.atleast_1d(np.asarray(signal, dtype=np.float64).squeeze())
                if array.ndim != 1 or len(array) == 0 or not np.isfinite(array).all():
                    raise ValueError("extractor returned an empty, non-1-D, or non-finite signal")
                raw_signals[method_name] = array
                method_records[method_name] = _status(
                    method_name, "VALID", None,
                    raw_weight=METHOD_WEIGHTS[method_name], **metadata,
                )
            except Exception as error:  # recorded; fusion is never silently substituted
                reason = f"{type(error).__name__}: {error}"
                method_records[method_name] = _status(method_name, "EXCLUDED", reason)
                excluded_components.append({"roi": roi_name, **method_records[method_name]})

        if not raw_signals:
            record = _status("roi", "EXCLUDED", "no_valid_algorithm_signal")
            excluded_components.append({"roi": roi_name, **record})
            roi_records[roi_name] = _status(roi_name, "EXCLUDED", "no_valid_algorithm_signal")
            continue

        usable_raw_signals = {}
        for method_name, signal in raw_signals.items():
            try:
                # Validate each method before common-length alignment so one
                # constant/invalid output cannot silently invalidate a whole
                # ROI or shorten valid method signals.
                _zscore(signal, f"{roi_name}/{method_name}")
                usable_raw_signals[method_name] = signal
            except ValueError as error:
                method_records[method_name] = _status(
                    method_name, "EXCLUDED", str(error), raw_weight=METHOD_WEIGHTS[method_name]
                )
                excluded_components.append({"roi": roi_name, **method_records[method_name]})

        if not usable_raw_signals:
            record = _status("roi", "EXCLUDED", "no_algorithm_signal_with_valid_variance")
            excluded_components.append({"roi": roi_name, **record})
            roi_records[roi_name] = _status(roi_name, "EXCLUDED", record["reason"])
            continue

        aligned_signals, alignment = _align_to_common_prefix(usable_raw_signals)
        normalized_signals = {
            method_name: _zscore(signal, f"{roi_name}/{method_name}")
            for method_name, signal in aligned_signals.items()
        }

        # This preserves CHROM as the historical polarity reference. If CHROM
        # is invalid, the first surviving configured method is used
        # deterministically, never the method with the best quality metric.
        reference_method = "chrom" if "chrom" in normalized_signals else next(
            name for name in METHOD_ORDER if name in normalized_signals
        )
        for method_name, signal in normalized_signals.items():
            correlation = float(np.corrcoef(normalized_signals[reference_method], signal)[0, 1])
            flipped = method_name != reference_method and correlation < 0
            if flipped:
                normalized_signals[method_name] = -signal
            method_records[method_name].update(alignment[method_name])
            method_records[method_name]["polarity_reference"] = reference_method
            method_records[method_name]["polarity_flipped"] = flipped

        effective_method_weights = normalize_weights(
            METHOD_WEIGHTS, normalized_signals, f"algorithm weights for ROI '{roi_name}'"
        )
        for method_name, effective_weight in effective_method_weights.items():
            method_records[method_name]["effective_weight"] = effective_weight

        try:
            combined = weighted_signal_combination(
                normalized_signals, METHOD_WEIGHTS, f"algorithm fusion for ROI '{roi_name}'"
            )
        except ValueError as error:
            record = _status("roi", "EXCLUDED", str(error))
            excluded_components.append({"roi": roi_name, **record})
            roi_records[roi_name] = _status(roi_name, "EXCLUDED", str(error))
            continue

        benchmark = {}
        for method_name, signal in normalized_signals.items():
            try:
                benchmark[method_name.upper()] = _benchmark_metrics(signal, fps)
            except ValueError as error:
                benchmark[method_name.upper()] = {"benchmark_error": str(error)}
        roi_benchmark[roi_name] = benchmark
        method_signals_per_roi[roi_name] = normalized_signals
        combined_per_roi[roi_name] = combined
        roi_records[roi_name] = _status(
            roi_name,
            "VALID",
            None,
            raw_weight=ROI_WEIGHTS[roi_name],
            methods=method_records,
            algorithm_weight_sum=float(sum(effective_method_weights.values())),
        )

        if debug and all(name in normalized_signals for name in METHOD_ORDER):
            from rPPG.reports.plots import plot_algorithm_comparison

            plot_algorithm_comparison(
                normalized_signals["chrom"], normalized_signals["pos"],
                normalized_signals["green"], normalized_signals["ica"], combined, fps,
            )

    if not combined_per_roi:
        reasons = "; ".join(
            f"{name}: {record['reason']}" for name, record in roi_records.items()
        )
        raise RuntimeError(f"No valid ROI remained for weighted rPPG fusion. {reasons}")

    combined_per_roi, roi_alignment = _align_to_common_prefix(combined_per_roi)
    effective_roi_weights = normalize_weights(ROI_WEIGHTS, combined_per_roi, "ROI weights")
    final_combined = weighted_signal_combination(combined_per_roi, ROI_WEIGHTS, "ROI fusion")

    # Algorithm benchmark is built from each method's ROI-weighted signal. It
    # is presentation data only and cannot replace final_combined.
    algorithm_benchmark = {}
    for method_name in METHOD_ORDER:
        signals_for_method = {
            roi_name: method_signals_per_roi[roi_name][method_name]
            for roi_name in combined_per_roi
            if method_name in method_signals_per_roi[roi_name]
        }
        if not signals_for_method:
            continue
        aligned_for_method, _ = _align_to_common_prefix(signals_for_method)
        method_final = weighted_signal_combination(
            aligned_for_method, ROI_WEIGHTS,
            f"ROI fusion for algorithm '{method_name}'",
        )
        try:
            algorithm_benchmark[method_name.upper()] = _benchmark_metrics(method_final, fps)
        except ValueError as error:
            algorithm_benchmark[method_name.upper()] = {"benchmark_error": str(error)}

    if verbose:
        from rPPG.reports.report import (
            print_algorithm_benchmark,
            print_roi_benchmark,
        )

        print_roi_benchmark(roi_benchmark, reference_hr=reference_hr)
        print_algorithm_benchmark(
            algorithm_benchmark.get("CHROM", {}),
            algorithm_benchmark.get("POS", {}),
            algorithm_benchmark.get("GREEN", {}),
            algorithm_benchmark.get("ICA"),
            reference_hr=reference_hr,
        )

    audit = {
        "configured_rois": configured_rois,
        "valid_rois": tuple(combined_per_roi),
        "roi_records": roi_records,
        "roi_alignment": roi_alignment,
        "effective_roi_weights": effective_roi_weights,
        "excluded_components": excluded_components,
        "roi_benchmark_source": "per-ROI, per-algorithm normalized and band-pass-filtered signals",
        # Observational results retained for the calibration module.  They are
        # deliberately not consumed by fusion or any weight-selection logic.
        "roi_benchmark": roi_benchmark,
        "algorithm_benchmark_source": "per-algorithm ROI-weighted signals, band-pass-filtered",
        "final_signal": {
            "source": "weighted_combination",
            "components": tuple(combined_per_roi),
            "samples": len(final_combined),
            "weight_sum": float(sum(effective_roi_weights.values())),
            "filtering": "not filtered here; final filter is applied once by analyze_video",
        },
    }
    if verbose:
        _print_signal_pipeline_audit(audit)
    result = SignalFusionResult(signal=final_combined, audit=audit)
    return result if return_audit else result.signal
