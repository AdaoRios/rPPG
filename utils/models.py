"""Result objects shared across pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def _scalar(value: Any) -> Any:
    """Return a JSON-friendly scalar (numpy scalars become Python numbers)."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _scalar(item())
        except (TypeError, ValueError):
            return None
    return value


@dataclass
class AnalysisResult:
    """Standard structured result returned by video analysis.

    The dataclass carries the full result, including provenance (``audit``) and
    the optional raw FFT arrays used by experimental consumers. Programmatic
    callers (CLI, reports, future API) should use :meth:`to_dict`, which is
    JSON-serializable and contains no raw arrays.
    """

    heart_rate: float
    hrv: dict[str, Any]
    respiratory_rate: float | None
    signal_metrics: dict[str, Any]
    fps: float
    duration: float
    valid_frames: int
    audit: dict[str, Any] | None = None
    # Optional raw FFT output for experimental consumers; production reports
    # continue to use the compact audit diagnostics.
    spectral_data: dict[str, Any] | None = None
    confidence: str = "low"
    ambiguous: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-serializable structured summary (API contract).

        Values are lifted from the same ``audit`` the console report consumes,
        so CLI, reports and a future API always serialize identical data.
        Raw FFT arrays (``spectral_data``) are intentionally excluded.
        """
        audit = self.audit or {}
        hr_fft = audit.get("hr_fft") or {}
        selected = hr_fft.get("selected_peak") or {}
        second = hr_fft.get("second_local_peak") or {}
        final_signal = audit.get("final_biomarker_signal") or {}
        metrics = dict(self.signal_metrics or {})

        return {
            "heart_rate_bpm": _scalar(self.heart_rate),
            "selected_frequency_hz": _scalar(selected.get("frequency_hz")),
            "confidence": self.confidence,
            "ambiguous": bool(self.ambiguous),
            "harmonic_detected": bool(hr_fft.get("harmonic_detected", False)),
            "harmonic_supported": bool(hr_fft.get("harmonic_supported", False)),
            "harmonic_reason": hr_fft.get("decision_reason"),
            "snr": _scalar(metrics.get("snr")),
            "spectral_concentration": _scalar(metrics.get("spectral_concentration")),
            "peak_ratio": _scalar(second.get("primary_to_second_ratio")),
            "effective_fps": _scalar(hr_fft.get("fps")),
            "sample_count": _scalar(hr_fft.get("samples")),
            "duration_seconds": _scalar(self.duration),
            "quality_metrics": {
                key: _scalar(value) for key, value in metrics.items()
            },
            "hrv": {key: _scalar(value) for key, value in (self.hrv or {}).items()},
            "provenance": {
                "video_path": audit.get("video_path"),
                "valid_frames": self.valid_frames,
                "capture_fps": _scalar(self.fps),
                "bandpass_hz": list(final_signal.get("bandpass_hz", ())),
                "signal_source": final_signal.get("source"),
                "window": hr_fft.get("window"),
                "selection_method": hr_fft.get("selection_method"),
                "frame_collection": audit.get("frame_collection"),
                "fusion": audit.get("fusion"),
            },
        }


__all__ = ["AnalysisResult"]