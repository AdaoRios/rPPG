"""Video-file rPPG analysis pipeline with frame-to-biomarker provenance."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from rPPG.biomarkers.heart_rate import analyze_hr_fft, classify_confidence
from rPPG.biomarkers.hrv import compute_hrv
from rPPG.biomarkers.signal_metrics import compute_signal_metrics
from rPPG.capture.lighting_quality import LightingQualityChecker
from rPPG.config import (
    DEBUG_COMPARE_ALGORITHMS,
    HR_HIGH_HZ,
    HR_LOW_HZ,
    LIGHTING_BRIGHT_PIXEL_CHANNEL,
    LIGHTING_DARK_PIXEL_LUMINANCE,
    LIGHTING_LUMA_WEIGHTS,
    LIGHTING_UNIFORMITY_GRID_COLUMNS,
    LIGHTING_UNIFORMITY_GRID_ROWS,
    MODEL_PATH,
    ROI_POINTS,
)
from rPPG.extractors.combine import combine_roi_and_methods
from rPPG.preprocessing.filters import bandpass_filter
from rPPG.roi.face_detection import FaceDetector
from rPPG.roi.roi_extraction import extract_roi_means
from rPPG.utils.models import AnalysisResult


def _landmark_bbox(landmarks):
    """Return a face bounding box for lighting measurement from pixel landmarks."""
    points = np.asarray(landmarks, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] < 2 or len(points) == 0:
        return None
    return (
        float(np.min(points[:, 0])),
        float(np.min(points[:, 1])),
        float(np.max(points[:, 0])),
        float(np.max(points[:, 1])),
    )


def _print_frame_collection_audit(frame_audit: dict) -> None:
    """Print concise frame-quality provenance without leaking per-frame data."""
    print("\n================ FRAME COLLECTION AUDIT ================")
    print(f"Frames read: {frame_audit['frames_read']}")
    print(f"Valid synchronized ROI frames: {frame_audit['valid_frames']}")
    print(f"Rejected: no face = {frame_audit['rejected_no_face']}")
    print(f"Rejected: ROI mask invalid = {frame_audit['rejected_roi_mask']}")
    print(
        "Lighting: "
        f"evaluated={frame_audit['lighting']['evaluated_frames']}; "
        f"unavailable={frame_audit['lighting']['unavailable_frames']}; "
        f"rejected={frame_audit['lighting']['rejected_frames']}"
    )
    print(
        "Lighting bright-channel threshold: "
        f"{frame_audit['lighting']['bright_pixel_channel_threshold']:.3f} "
        "on normalized RGB [0, 1] (observational; no rejection policy configured)"
    )
    invalid = frame_audit["invalid_roi_counts"]
    if any(invalid.values()):
        print("ROI extraction failures: " + ", ".join(
            f"{name.upper()}={count}" for name, count in invalid.items() if count
        ))
    else:
        print("ROI extraction failures: none")
    print("==========================================================\n")


def _reference_validation(estimated_hr, reference_hr):
    """Return an observational external-reference comparison when supplied."""
    if reference_hr is None:
        return None
    reference_hr = float(reference_hr)
    if not np.isfinite(reference_hr) or reference_hr <= 0:
        raise ValueError("reference_hr deve ser positivo e finito.")
    absolute_error = abs(estimated_hr - reference_hr)
    return {
        "reference_hr_bpm": reference_hr,
        "estimated_hr_bpm": float(estimated_hr),
        "absolute_error_bpm": float(absolute_error),
        "relative_error_percent": float(100.0 * absolute_error / reference_hr),
        "role": "validation only; never used to select a production signal or weight",
    }


def analyze_video(video_path, reference_hr=None, verbose=True):
    """Analyze an MP4/video file and return a traceable ``AnalysisResult``.

    This is the production core entry point: ``result = analyze_video(video)``.
    It has no ``input()``, menu, GUI, or calibration dependency, and it never
    needs printing to work. ``verbose`` only toggles the console audits for
    CLI use; programmatic callers (e.g. a future API) pass ``verbose=False``
    and read the same values from the returned ``AnalysisResult``/``audit``.

    A retained sample always contains every configured ROI from the same video
    frame. Lighting is measured with the experimental 0.784 channel threshold
    but is not a frame-rejection rule: no calibrated rejection ratio exists in
    the current architecture, and inventing one would be an opaque heuristic.
    """
    if reference_hr is not None:
        reference_hr = float(reference_hr)
        if not np.isfinite(reference_hr) or reference_hr <= 0:
            raise ValueError("reference_hr deve ser positivo e finito.")
    video_path = Path(video_path)
    if not video_path.is_file():
        raise FileNotFoundError(f"Vídeo não encontrado: {video_path}")
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Não foi possível abrir o vídeo: {video_path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    detector = FaceDetector(MODEL_PATH)
    lighting_checker = LightingQualityChecker(
        LIGHTING_LUMA_WEIGHTS,
        LIGHTING_DARK_PIXEL_LUMINANCE,
        LIGHTING_BRIGHT_PIXEL_CHANNEL,
        LIGHTING_UNIFORMITY_GRID_ROWS,
        LIGHTING_UNIFORMITY_GRID_COLUMNS,
    )
    roi_signals = {roi_name: [] for roi_name in ROI_POINTS}
    frame_audit = {
        "frames_read": 0,
        "valid_frames": 0,
        "rejected_no_face": 0,
        "rejected_roi_mask": 0,
        "invalid_roi_counts": {roi_name: 0 for roi_name in ROI_POINTS},
        "lighting": {
            "metric": "face-bbox pixel ratio where max(R, G, B) >= threshold",
            "scale": "normalized RGB [0, 1]",
            "bright_pixel_channel_threshold": LIGHTING_BRIGHT_PIXEL_CHANNEL,
            "evaluated_frames": 0,
            "unavailable_frames": 0,
            "rejected_frames": 0,
            "exclusion_policy": "observational only; no calibrated lighting rejection rule",
            "observations": [],
        },
    }
    frame_index = 0
    try:
        while True:
            success, frame = capture.read()
            if not success:
                break
            frame_audit["frames_read"] += 1
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            timestamp_ms = int(frame_index * 1000.0 / fps)
            landmarks = detector.detect(rgb_frame, timestamp_ms)
            if landmarks is None:
                frame_audit["rejected_no_face"] += 1
                frame_audit["lighting"]["unavailable_frames"] += 1
                frame_index += 1
                continue

            lighting = lighting_checker.evaluate(rgb_frame, _landmark_bbox(landmarks))
            if lighting.face_detected:
                frame_audit["lighting"]["evaluated_frames"] += 1
                frame_audit["lighting"]["observations"].append({
                    "bright_pixel_ratio": lighting.bright_pixel_ratio,
                    "mean_luminance": lighting.mean_luminance,
                    "dark_pixel_ratio": lighting.dark_pixel_ratio,
                    "illumination_uniformity": lighting.illumination_uniformity,
                })
            else:
                frame_audit["lighting"]["unavailable_frames"] += 1

            means, invalid_rois = extract_roi_means(
                rgb_frame, landmarks, ROI_POINTS, return_diagnostics=True
            )
            if invalid_rois:
                frame_audit["rejected_roi_mask"] += 1
                for roi_name in invalid_rois:
                    frame_audit["invalid_roi_counts"][roi_name] += 1
                frame_index += 1
                continue
            for roi_name in ROI_POINTS:
                roi_signals[roi_name].append(means[roi_name])
            frame_audit["valid_frames"] += 1
            frame_index += 1
    finally:
        capture.release()
        detector.close()

    valid_frames = frame_audit["valid_frames"]
    if valid_frames < 2:
        raise RuntimeError("Poucos frames válidos no vídeo para análise rPPG.")
    signals = {
        name: np.asarray(values, dtype=np.float64) for name, values in roi_signals.items()
    }
    fusion = combine_roi_and_methods(
        signals,
        fps,
        debug=DEBUG_COMPARE_ALGORITHMS,
        return_audit=True,
        reference_hr=reference_hr,
        verbose=verbose,
    )

    # This is the sole final filter. The same exact signal reaches HR, HRV,
    # and final quality metrics, preventing a report/biomarker divergence.
    filtered_signal = bandpass_filter(
        fusion.signal, fps, low_hz=HR_LOW_HZ, high_hz=HR_HIGH_HZ
    )
    hr_fft = analyze_hr_fft(filtered_signal, fps)
    heart_rate = hr_fft["selected_peak"]["hr_bpm"]
    reference_validation = _reference_validation(heart_rate, reference_hr)
    signal_metrics = compute_signal_metrics(filtered_signal, fps)
    decision = hr_fft["decision"]
    confidence = classify_confidence(decision, signal_metrics)
    hr_fft["ambiguous"] = bool(decision["ambiguous"])
    hr_fft["harmonic_detected"] = bool(decision["harmonic_detected"])
    hr_fft["harmonic_supported"] = bool(decision["harmonic_supported"])
    hr_fft["decision_reason"] = decision["reason"]
    hr_fft["confidence"] = confidence
    observations = frame_audit["lighting"].pop("observations")
    for metric in ("bright_pixel_ratio", "mean_luminance", "dark_pixel_ratio", "illumination_uniformity"):
        values = [item[metric] for item in observations if item[metric] is not None]
        frame_audit["lighting"][metric] = float(np.mean(values)) if values else None
    frame_audit["valid_frame_rate"] = valid_frames / frame_audit["frames_read"]
    if verbose:
        _print_frame_collection_audit(frame_audit)
    audit = {
        "video_path": str(video_path),
        "fps": float(fps),
        "frame_collection": frame_audit,
        "fusion": fusion.audit,
        "hr_fft": hr_fft,
        "reference_validation": reference_validation,
        "final_biomarker_signal": {
            "source": "weighted_combination_then_single_final_bandpass",
            "samples": len(filtered_signal),
            "bandpass_hz": (HR_LOW_HZ, HR_HIGH_HZ),
            "consumers": ("heart_rate", "hrv", "signal_metrics"),
        },
    }
    return AnalysisResult(
        heart_rate=heart_rate,
        hrv=compute_hrv(filtered_signal, fps),
        respiratory_rate=None,
        fps=fps,
        duration=valid_frames / fps,
        valid_frames=valid_frames,
        audit=audit,
        confidence=confidence,
        ambiguous=bool(decision["ambiguous"]),
        signal_metrics=signal_metrics,
        spectral_data={
            "frequency_hz": np.fft.rfftfreq(len(filtered_signal), d=1.0 / fps),
            "magnitude": np.abs(np.fft.rfft(filtered_signal * np.hanning(len(filtered_signal)))),
        },
    )
