"""Result objects shared across pipelines."""

from dataclasses import dataclass
from typing import Any

from rPPG.preprocessing.detrend import detrend
from rPPG.preprocessing.processing import _process_video


@dataclass
class AnalysisResult:
    """Standard result returned by video analysis."""

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


__all__ = ["AnalysisResult", "_process_video", "detrend"]
