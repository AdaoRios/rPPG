"""Export the 16 ROI x algorithm fusion inputs for offline weight calibration.

This module is additive and read-only with respect to production code. It
replicates, with the same production primitives, the frame collection of
``analysis.analyze_video`` and the fusion of ``extractors.combine``, but it
keeps the per-method signals that production discards after building the
audit trail.

Exported arrays are exactly the signals that enter
``weighted_signal_combination`` in production: moving-average smoothed RGB ->
extractor -> length validation -> common-prefix alignment -> z-score ->
polarity correction against the CHROM reference. Nothing else is applied.

For each capture the script writes:

- ``signals.npz``: 16 float64 arrays keyed ``<roi>.<method>``
- ``metadata.json``: alignment, polarity, validity, weights, frozen pipeline
  parameters, and the reconstruction validation against the recorded final HR

Outputs go to a NEW directory (default
``C:\\rPPG\\data\\weight_calibration_intermediates``). Historical baseline and
final-validation results are never modified.

Usage (run from ``C:\\rPPG`` so the ``rPPG`` package resolves):

    python -m rPPG.weight_calibration.export_intermediates --only capture_001
    python -m rPPG.weight_calibration.export_intermediates
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from rPPG.biomarkers.heart_rate import analyze_hr_fft
from rPPG.config import (
    HR_HIGH_HZ,
    HR_LOW_HZ,
    METHOD_WEIGHTS,
    MODEL_PATH,
    ROI_POINTS,
    ROI_WEIGHTS,
)
from rPPG.extractors.combine import (
    METHOD_ORDER,
    _align_to_common_prefix,
    _zscore,
    normalize_weights,
    weighted_signal_combination,
)
from rPPG.extractors.chrom import chrom_algorithm
from rPPG.extractors.green import green_algorithm
from rPPG.extractors.ica import ica_algorithm
from rPPG.extractors.pos import pos_algorithm
from rPPG.preprocessing.filters import bandpass_filter, moving_average_smooth
from rPPG.roi.face_detection import FaceDetector
from rPPG.roi.roi_extraction import extract_roi_means

HR_TOLERANCE_BPM = 1e-6


def collect_roi_signals(video_path: Path):
    """Collect synchronized per-ROI RGB means exactly like analyze_video."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Nao foi possivel abrir o video: {video_path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    detector = FaceDetector(MODEL_PATH)
    roi_signals = {roi_name: [] for roi_name in ROI_POINTS}
    frames_read = 0
    rejected_no_face = 0
    rejected_roi_mask = 0
    frame_index = 0
    try:
        while True:
            success, frame = capture.read()
            if not success:
                break
            frames_read += 1
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            timestamp_ms = int(frame_index * 1000.0 / fps)
            landmarks = detector.detect(rgb_frame, timestamp_ms)
            if landmarks is None:
                rejected_no_face += 1
                frame_index += 1
                continue
            means, invalid_rois = extract_roi_means(
                rgb_frame, landmarks, ROI_POINTS, return_diagnostics=True
            )
            if invalid_rois:
                rejected_roi_mask += 1
                frame_index += 1
                continue
            for roi_name in ROI_POINTS:
                roi_signals[roi_name].append(means[roi_name])
            frame_index += 1
    finally:
        capture.release()
        detector.close()
    valid_frames = sum(len(values) for values in roi_signals.values()) // len(ROI_POINTS)
    if valid_frames < 2:
        raise RuntimeError("Poucos frames validos no video para analise rPPG.")
    signals = {
        name: np.asarray(values, dtype=np.float64) for name, values in roi_signals.items()
    }
    stats = {
        "frames_read": frames_read,
        "valid_frames": valid_frames,
        "rejected_no_face": rejected_no_face,
        "rejected_roi_mask": rejected_roi_mask,
        "fps": float(fps),
    }
    return signals, stats


def fuse_and_keep_components(roi_signals, fps):
    """Replicate combine_roi_and_methods while retaining the 16 fusion inputs.

    Every step, validation, and weight normalization mirrors production
    ``extractors.combine`` so the final signal is identical for the current
    weights.
    """
    if fps <= 0 or not np.isfinite(fps):
        raise ValueError("fps must be a positive finite value")
    if not roi_signals:
        raise ValueError("at least one ROI signal is required")

    configured_rois = tuple(roi_signals)
    missing_weights = [roi_name for roi_name in configured_rois if roi_name not in ROI_WEIGHTS]
    if missing_weights:
        raise ValueError(f"ROI weights missing for: {', '.join(missing_weights)}")

    method_signals_per_roi = {}
    combined_per_roi = {}
    roi_records = {}
    excluded_components = []

    for roi_name in configured_rois:
        rgb_raw = np.asarray(roi_signals[roi_name], dtype=np.float64)
        if rgb_raw.ndim != 2 or rgb_raw.shape[1] != 3 or len(rgb_raw) == 0:
            record = {"name": "roi", "status": "EXCLUDED", "reason": "invalid_rgb_signal"}
            excluded_components.append({"roi": roi_name, **record})
            roi_records[roi_name] = {"name": roi_name, "status": "EXCLUDED",
                                     "reason": "invalid_rgb_signal"}
            continue
        if not np.isfinite(rgb_raw).all():
            record = {"name": "roi", "status": "EXCLUDED", "reason": "non_finite_rgb_signal"}
            excluded_components.append({"roi": roi_name, **record})
            roi_records[roi_name] = {"name": roi_name, "status": "EXCLUDED",
                                     "reason": "non_finite_rgb_signal"}
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
                    raise ValueError(
                        "extractor returned an empty, non-1-D, or non-finite signal"
                    )
                raw_signals[method_name] = array
                method_records[method_name] = {
                    "name": method_name, "status": "VALID", "reason": None,
                    "raw_weight": METHOD_WEIGHTS[method_name], **metadata,
                }
            except Exception as error:  # recorded; never silently substituted
                reason = f"{type(error).__name__}: {error}"
                method_records[method_name] = {
                    "name": method_name, "status": "EXCLUDED", "reason": reason,
                    "raw_weight": METHOD_WEIGHTS[method_name],
                }
                excluded_components.append({"roi": roi_name, **method_records[method_name]})

        if not raw_signals:
            record = {"name": "roi", "status": "EXCLUDED", "reason": "no_valid_algorithm_signal"}
            excluded_components.append({"roi": roi_name, **record})
            roi_records[roi_name] = {"name": roi_name, "status": "EXCLUDED",
                                     "reason": "no_valid_algorithm_signal"}
            continue

        usable_raw_signals = {}
        for method_name, signal in raw_signals.items():
            try:
                _zscore(signal, f"{roi_name}/{method_name}")
                usable_raw_signals[method_name] = signal
            except ValueError as error:
                method_records[method_name] = {
                    "name": method_name, "status": "EXCLUDED", "reason": str(error),
                    "raw_weight": METHOD_WEIGHTS[method_name],
                }
                excluded_components.append({"roi": roi_name, **method_records[method_name]})

        if not usable_raw_signals:
            record = {"name": "roi", "status": "EXCLUDED",
                      "reason": "no_algorithm_signal_with_valid_variance"}
            excluded_components.append({"roi": roi_name, **record})
            roi_records[roi_name] = {"name": roi_name, "status": "EXCLUDED",
                                     "reason": record["reason"]}
            continue

        aligned_signals, alignment = _align_to_common_prefix(usable_raw_signals)
        normalized_signals = {
            method_name: _zscore(signal, f"{roi_name}/{method_name}")
            for method_name, signal in aligned_signals.items()
        }

        reference_method = "chrom" if "chrom" in normalized_signals else next(
            name for name in METHOD_ORDER if name in normalized_signals
        )
        for method_name, signal in normalized_signals.items():
            correlation = float(
                np.corrcoef(normalized_signals[reference_method], signal)[0, 1]
            )
            flipped = method_name != reference_method and correlation < 0
            if flipped:
                normalized_signals[method_name] = -signal
            method_records[method_name].update(alignment[method_name])
            method_records[method_name]["polarity_reference"] = reference_method
            method_records[method_name]["polarity_flipped"] = flipped

        effective_method_weights = normalize_weights(
            METHOD_WEIGHTS, normalized_signals,
            f"algorithm weights for ROI '{roi_name}'",
        )
        for method_name, effective_weight in effective_method_weights.items():
            method_records[method_name]["effective_weight"] = effective_weight

        try:
            combined = weighted_signal_combination(
                normalized_signals, METHOD_WEIGHTS,
                f"algorithm fusion for ROI '{roi_name}'",
            )
        except ValueError as error:
            record = {"name": "roi", "status": "EXCLUDED", "reason": str(error)}
            excluded_components.append({"roi": roi_name, **record})
            roi_records[roi_name] = {"name": roi_name, "status": "EXCLUDED",
                                     "reason": str(error)}
            continue

        method_signals_per_roi[roi_name] = normalized_signals
        combined_per_roi[roi_name] = combined
        roi_records[roi_name] = {
            "name": roi_name,
            "status": "VALID",
            "reason": None,
            "raw_weight": ROI_WEIGHTS[roi_name],
            "methods": method_records,
            "algorithm_weight_sum": float(sum(effective_method_weights.values())),
        }

    if not combined_per_roi:
        reasons = "; ".join(
            f"{name}: {record['reason']}" for name, record in roi_records.items()
        )
        raise RuntimeError(f"No valid ROI remained for weighted rPPG fusion. {reasons}")

    aligned_rois, roi_alignment = _align_to_common_prefix(combined_per_roi)
    effective_roi_weights = normalize_weights(ROI_WEIGHTS, combined_per_roi, "ROI weights")
    final_combined = weighted_signal_combination(aligned_rois, ROI_WEIGHTS, "ROI fusion")

    return {
        "method_signals_per_roi": method_signals_per_roi,
        "roi_records": roi_records,
        "excluded_components": excluded_components,
        "combined_samples_per_roi": {
            name: len(array) for name, array in combined_per_roi.items()
        },
        "roi_alignment": roi_alignment,
        "effective_roi_weights": effective_roi_weights,
        "final_signal": final_combined,
    }


def _analyze_final(final_signal, fps):
    """Apply the sole final filter and the frozen HR selection, as production."""
    filtered = bandpass_filter(final_signal, fps, low_hz=HR_LOW_HZ, high_hz=HR_HIGH_HZ)
    hr_fft = analyze_hr_fft(filtered, fps)
    return hr_fft


def _compare_with_recorded(recorded: dict, fusion: dict, hr_fft: dict, stats: dict) -> list:
    """Return a list of discrepancies versus the recorded final validation."""
    problems = []
    recorded_final = float(recorded["final_hr_bpm"])
    rebuilt_final = float(hr_fft["selected_peak"]["hr_bpm"])
    if abs(recorded_final - rebuilt_final) > HR_TOLERANCE_BPM:
        problems.append(
            f"final_hr_mismatch: recorded={recorded_final!r} rebuilt={rebuilt_final!r}"
        )

    recorded_fusion = recorded.get("audit", {}).get("fusion", {})
    rebuilt_samples = len(fusion["final_signal"])
    recorded_samples = (
        recorded.get("audit", {}).get("final_biomarker_signal", {}).get("samples")
    )
    if recorded_samples is not None and int(recorded_samples) != rebuilt_samples:
        problems.append(
            f"final_samples_mismatch: recorded={recorded_samples} rebuilt={rebuilt_samples}"
        )

    recorded_roi_weights = recorded_fusion.get("effective_roi_weights")
    if recorded_roi_weights is not None:
        for roi_name, value in recorded_roi_weights.items():
            rebuilt = fusion["effective_roi_weights"].get(roi_name)
            if rebuilt is None or abs(float(value) - float(rebuilt)) > 1e-12:
                problems.append(
                    f"roi_weight_mismatch[{roi_name}]: recorded={value} rebuilt={rebuilt}"
                )

    for roi_name, recorded_record in recorded_fusion.get("roi_records", {}).items():
        rebuilt_record = fusion["roi_records"].get(roi_name)
        if rebuilt_record is None:
            problems.append(f"roi_missing[{roi_name}]")
            continue
        if recorded_record.get("status") != rebuilt_record.get("status"):
            problems.append(
                f"roi_status_mismatch[{roi_name}]: "
                f"recorded={recorded_record.get('status')} "
                f"rebuilt={rebuilt_record.get('status')}"
            )
        for method_name, recorded_method in recorded_record.get("methods", {}).items():
            rebuilt_method = rebuilt_record.get("methods", {}).get(method_name)
            if rebuilt_method is None:
                problems.append(f"method_missing[{roi_name}.{method_name}]")
                continue
            for field in (
                "status",
                "original_samples",
                "aligned_samples",
                "tail_samples_discarded_for_alignment",
                "polarity_flipped",
            ):
                if field in recorded_method and recorded_method.get(field) != rebuilt_method.get(
                    field
                ):
                    problems.append(
                        f"{field}_mismatch[{roi_name}.{method_name}]: "
                        f"recorded={recorded_method.get(field)} "
                        f"rebuilt={rebuilt_method.get(field)}"
                    )
            if "effective_weight" in recorded_method and abs(
                float(recorded_method["effective_weight"])
                - float(rebuilt_method.get("effective_weight"))
            ) > 1e-12:
                problems.append(f"effective_weight_mismatch[{roi_name}.{method_name}]")

    recorded_alignment = recorded_fusion.get("roi_alignment", {})
    for roi_name, recorded_item in recorded_alignment.items():
        rebuilt_item = fusion["roi_alignment"].get(roi_name)
        if rebuilt_item is None:
            problems.append(f"roi_alignment_missing[{roi_name}]")
            continue
        for field in ("original_samples", "aligned_samples",
                      "tail_samples_discarded_for_alignment"):
            if field in recorded_item and int(recorded_item[field]) != int(
                rebuilt_item[field]
            ):
                problems.append(
                    f"roi_alignment_{field}_mismatch[{roi_name}]: "
                    f"recorded={recorded_item[field]} rebuilt={rebuilt_item[field]}"
                )

    recorded_frames = (
        recorded.get("audit", {}).get("frame_collection", {}).get("frames_read")
    )
    if recorded_frames is not None and int(recorded_frames) != int(stats["frames_read"]):
        problems.append(
            f"frames_read_mismatch: recorded={recorded_frames} rebuilt={stats['frames_read']}"
        )
    return problems


def export_capture(capture_id, video_path, out_dir, recorded=None):
    """Export one capture and validate the reconstruction against production."""
    roi_signals, stats = collect_roi_signals(video_path)
    fusion = fuse_and_keep_components(roi_signals, stats["fps"])
    hr_fft = _analyze_final(fusion["final_signal"], stats["fps"])
    rebuilt_hr = float(hr_fft["selected_peak"]["hr_bpm"])

    problems = []
    if recorded is not None:
        problems = _compare_with_recorded(recorded, fusion, hr_fft, stats)

    capture_dir = out_dir / capture_id
    capture_dir.mkdir(parents=True, exist_ok=True)

    arrays = {}
    for roi_name, methods in fusion["method_signals_per_roi"].items():
        for method_name, signal in methods.items():
            arrays[f"{roi_name}.{method_name}"] = np.asarray(signal, dtype=np.float64)
    np.savez(capture_dir / "signals.npz", **arrays)

    metadata = {
        "capture_id": capture_id,
        "video_path": str(video_path),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "export_point": (
            "per-method signals after moving-average smoothing, extractor, "
            "common-prefix alignment, z-score, and polarity correction; "
            "exactly the inputs of production weighted_signal_combination"
        ),
        "frame_collection": {**stats},
        "fps": stats["fps"],
        "components": {},
        "rois": {},
        "combined_samples_per_roi": fusion["combined_samples_per_roi"],
        "roi_alignment": fusion["roi_alignment"],
        "effective_roi_weights": fusion["effective_roi_weights"],
        "excluded_components": fusion["excluded_components"],
        "final_signal_samples": len(fusion["final_signal"]),
        "current_production_weights": {
            "method_weights": dict(METHOD_WEIGHTS),
            "roi_weights": dict(ROI_WEIGHTS),
        },
        "frozen_pipeline_state": {
            "hr_band_hz": [HR_LOW_HZ, HR_HIGH_HZ],
            "window": "Hann",
            "zero_padding": False,
            "peak_interpolation": False,
            "harmonic_policy": "unchanged production policy in biomarkers.heart_rate",
            "lighting_threshold_policy": "observational only; no rejection threshold",
            "reference_role": "external evaluation only; never selects HR or weights",
        },
        "rebuilt_final_hr_bpm": rebuilt_hr,
        "recorded_final_hr_bpm": (
            None if recorded is None else float(recorded["final_hr_bpm"])
        ),
        "reference_hr_bpm": (
            None if recorded is None else recorded.get("reference_hr_bpm")
        ),
        "reconstruction_matches_recorded": (
            not problems if recorded is not None else None
        ),
        "reconstruction_problems": problems,
        "decision": {
            "ambiguous": bool(hr_fft["decision"]["ambiguous"]),
            "harmonic_detected": bool(hr_fft["decision"]["harmonic_detected"]),
            "harmonic_supported": bool(hr_fft["decision"]["harmonic_supported"]),
            "reason": hr_fft["decision"]["reason"],
        },
    }
    for roi_name, record in fusion["roi_records"].items():
        metadata["rois"][roi_name] = {
            "status": record.get("status"),
            "reason": record.get("reason"),
            "raw_weight": record.get("raw_weight"),
        }
        if record.get("status") == "VALID":
            metadata["components"][roi_name] = {
                method_name: {
                    field: method_record.get(field)
                    for field in (
                        "status",
                        "reason",
                        "raw_weight",
                        "effective_weight",
                        "original_samples",
                        "aligned_samples",
                        "tail_samples_discarded_for_alignment",
                        "polarity_reference",
                        "polarity_flipped",
                        "source_selection",
                        "selected_component",
                    )
                    if method_record.get(field) is not None
                }
                for method_name, method_record in record["methods"].items()
            }

    with open(capture_dir / "metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)

    return metadata, len(arrays)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Export the 16 ROI x algorithm fusion inputs per capture."
    )
    parser.add_argument(
        "--validation-dir",
        default=r"C:\rPPG\data\results_final_validation\data",
        help="Directory with the recorded final-validation capture JSON files.",
    )
    parser.add_argument(
        "--out-dir",
        default=r"C:\rPPG\data\weight_calibration_intermediates",
        help="NEW output directory (never a historical results directory).",
    )
    parser.add_argument("--only", default=None, help="Export a single capture_id.")
    parser.add_argument("--limit", type=int, default=None, help="Export at most N captures.")
    args = parser.parse_args(argv)

    validation_dir = Path(args.validation_dir)
    out_dir = Path(args.out_dir)
    forbidden = {Path(r"C:\rPPG\data\results").resolve(),
                 Path(r"C:\rPPG\data\results_final_validation").resolve()}
    if out_dir.resolve() in forbidden:
        print("ERRO: --out-dir nao pode ser um diretorio historico de resultados.")
        return 2

    recorded_files = sorted(validation_dir.glob("capture_*.json"))
    if not recorded_files:
        print(f"ERRO: nenhum capture_*.json em {validation_dir}")
        return 2
    if args.only:
        recorded_files = [path for path in recorded_files if path.stem == args.only]
        if not recorded_files:
            print(f"ERRO: captura nao encontrada: {args.only}")
            return 2
    if args.limit:
        recorded_files = recorded_files[: args.limit]

    summary = {"generated_at_utc": datetime.now(timezone.utc).isoformat(),
               "captures": [], "failures": []}
    for path in recorded_files:
        with open(path, "r", encoding="utf-8") as handle:
            recorded = json.load(handle)
        capture_id = recorded["capture_id"]
        video_path = Path(recorded["video_path_used"])
        print(f"\n--- {capture_id}: {video_path.name} ---")
        try:
            metadata, component_count = export_capture(
                capture_id, video_path, out_dir, recorded=recorded
            )
        except Exception as error:
            print(f"ERRO em {capture_id}: {type(error).__name__}: {error}")
            summary["failures"].append(
                {"capture_id": capture_id, "error": f"{type(error).__name__}: {error}"}
            )
            with open(out_dir / "export_summary.json", "w", encoding="utf-8") as handle:
                json.dump(summary, handle, indent=2, ensure_ascii=False)
            return 1
        matches = metadata["reconstruction_matches_recorded"]
        print(
            f"componentes={component_count}; "
            f"HR_reconstruido={metadata['rebuilt_final_hr_bpm']:.6f}; "
            f"HR_registrado={metadata['recorded_final_hr_bpm']}; "
            f"reproduz={matches}"
        )
        if matches is False:
            for problem in metadata["reconstruction_problems"]:
                print(f"  PROBLEMA: {problem}")
            with open(out_dir / "export_summary.json", "w", encoding="utf-8") as handle:
                json.dump(summary, handle, indent=2, ensure_ascii=False)
            print(
                "\nPARADO: a reconstrucao nao reproduz o HR registrado. "
                "Nenhum video adicional sera processado."
            )
            return 3
        summary["captures"].append(
            {
                "capture_id": capture_id,
                "components": component_count,
                "rebuilt_final_hr_bpm": metadata["rebuilt_final_hr_bpm"],
                "recorded_final_hr_bpm": metadata["recorded_final_hr_bpm"],
                "matches_recorded": matches,
                "frames_read": metadata["frame_collection"]["frames_read"],
            }
        )

    with open(out_dir / "export_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(
        f"\nExportacao concluida: {len(summary['captures'])} capturas em {out_dir} "
        f"(falhas: {len(summary['failures'])})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())