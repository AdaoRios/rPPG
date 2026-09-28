"""Shared rPPG configuration."""

from pathlib import Path


MODEL_PATH = str(Path(__file__).resolve().parent / "assets" / "face_landmarker.task")
DEBUG_COMPARE_ALGORITHMS = False



# Capture readiness MVP: initial experimental values, to be calibrated with
# webcam observations. Th
# e metric is median landmark displacement / face diagonal.
MOVEMENT_THRESHOLD = 0.010
MOVEMENT_WINDOW_SIZE = 5
READY_STABLE_FRAMES = 15

# Face Framing MVP: normalized to the full frame. These experimental values
# were calibrated from manual tests with one webcam/user; they are not
# clinically or universally validated.
FACE_MIN_WIDTH_RATIO = 0.28
FACE_MAX_WIDTH_RATIO = 0.45
FACE_MIN_HEIGHT_RATIO = 0.45
FACE_MAX_HEIGHT_RATIO = 0.80
FACE_MAX_CENTER_OFFSET_X = 0.15
FACE_MAX_CENTER_OFFSET_Y = 0.18

# Lighting Quality: values are normalized to [0, 1].  Bright-pixel detection is
# observational in this version: it records the proportion of face pixels with
# at least one saturated/high channel and does not silently reject frames.
# 0.784 is the experimental/recommended threshold adopted for this release;
# its external reference has not yet been added to this repository.
LIGHTING_LUMA_WEIGHTS = (0.2126, 0.7152, 0.0722)
LIGHTING_DARK_PIXEL_LUMINANCE = 0.10
LIGHTING_BRIGHT_PIXEL_CHANNEL = 0.784
LIGHTING_UNIFORMITY_GRID_ROWS = 2
LIGHTING_UNIFORMITY_GRID_COLUMNS = 2

# =============================================================================
# FROZEN SCIENTIFIC PARAMETERS (final freeze)
# -----------------------------------------------------------------------------
# The scientific part of this project is closed. Do NOT change the cardiac
# band, the method weights, the ROI weights, the final filter, the FFT
# (Hann, no zero-padding, no interpolation), the spectral selection, the
# harmonic policy, confidence, or ambiguity. Any future change requires a new
# explicit validation campaign, not an in-place edit.
# =============================================================================

# Final HR, HRV, and signal-metrics cardiac band (0.7-4.0 Hz, or 42-240 bpm).
HR_LOW_HZ = 0.7    # 42 bpm  (FROZEN)
HR_HIGH_HZ = 4.0   # 240 bpm (FROZEN)

# Method fusion weights (FROZEN), verified in the final LOOCV study.
METHOD_WEIGHTS = {
    "chrom": 0.3,   # FROZEN
    "pos": 0.4,     # FROZEN
    "ica": 0.2,     # FROZEN
    "green": 0.1,   # FROZEN
}

# ROI fusion weights (FROZEN as configuration D2). Frozen after the final
# A-vs-D2 verification on the 25 exported captures: MAE 16.4341 vs 18.1498 bpm
# (A), RMSE 20.1205 vs 20.8946, median 13.75 vs 14.89, ambiguous 3 vs 7,
# bias +6.93 vs +8.19, max error 42.15 vs 39.75 (within the pre-declared
# 3.0 bpm non-relevance margin), confidence distribution unchanged.
# Evidence: C:\rPPG\data\weight_calibration_intermediates\final_d2_check.json
# All configured ROI weights sum exactly to one.
ROI_WEIGHTS = {
    "testa": 0.40,               # FROZEN (D2)
    "bochecha_esquerda": 0.20,   # FROZEN (D2)
    "bochecha_direita": 0.30,    # FROZEN (D2)
    "glabela": 0.10,             # FROZEN (D2)
}

ROI_POINTS = {
    "testa": [109, 67, 103, 10, 332, 297, 338, 151, 9, 8],
    "bochecha_esquerda": [117, 118, 101, 123, 187, 207, 192, 214, 138, 135, 198, 50],
    "bochecha_direita": [346, 347, 330, 352, 411, 427, 416, 434, 367, 364, 418, 280],
    # MediaPipe Face Landmarker mesh: points around the inter-brow region.
    # 9 is the mid-forehead point; 107/336 are the medial eyebrow points;
    # 168 is the nasal bridge/glabella point.
    "glabela": [107, 9, 336, 168],
}
