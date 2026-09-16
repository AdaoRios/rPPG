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

# Final HR, HRV, and signal-metrics cardiac band. These values match the
# pre-existing implementation (0.7-4.0 Hz, or 42-240 bpm), now centralized so
# filtering and spectral analysis cannot silently use different ranges.
HR_LOW_HZ = 0.7    # 42 bpm
HR_HIGH_HZ = 4.0   # 240 bpm

METHOD_WEIGHTS = {
    "chrom": 0.3,
    "pos": 0.3,
    "ica": 0.3,
    "green": 0.1
}

ROI_WEIGHTS = {
    # Legacy relative weights (4:3:3) are preserved.  The experimental
    # glabella contribution receives 10%, so legacy contributions are scaled
    # by 0.90 and all configured ROI weights still sum exactly to one.
    "testa": 0.36,
    "bochecha_esquerda": 0.27,
    "bochecha_direita": 0.27,
    "glabela": 0.10,
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
