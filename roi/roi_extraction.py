import cv2
import numpy as np

def get_roi_mask(landmarks_px, roi_indices, frame_shape, erode_kernel_size=5):
    """Build an eroded ROI mask to eliminate edge motion noise."""
    points = np.array(
        [landmarks_px[idx] for idx in roi_indices if idx < len(landmarks_px)],
        dtype=np.int32
    )
    if len(points) < 3:
        return None

    mask = np.zeros(frame_shape[:2], dtype=np.uint8)
    
    cv2.fillPoly(mask, [points], 255)
    
    # Erode edges to prevent boundary mixing during slight head motion
    if erode_kernel_size > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (erode_kernel_size, erode_kernel_size)
        )
        mask = cv2.erode(mask, kernel, iterations=1)
        
    return mask


def extract_roi_means(rgb_frame, landmarks_px, roi_points, return_diagnostics=False):
    """Extract per-ROI RGB means and optionally expose invalid ROI reasons.

    All retained frame means must originate from the same video instant.  The
    default preserves the historical all-ROI behaviour: if any configured ROI
    cannot produce a sufficiently large mask, it returns ``None``.  Callers
    that need audit data may opt into ``return_diagnostics`` and receive the
    valid means plus a ``{roi_name: reason}`` mapping.
    """
    frame_means = {}
    invalid_rois = {}

    for roi_name, indices in roi_points.items():
        available_indices = [idx for idx in indices if idx < len(landmarks_px)]
        mask = get_roi_mask(landmarks_px, indices, rgb_frame.shape, erode_kernel_size=5)
        mask_pixels = 0 if mask is None else cv2.countNonZero(mask)

        if len(available_indices) < 3:
            invalid_rois[roi_name] = "insufficient_landmarks"
        elif mask_pixels < 50:
            invalid_rois[roi_name] = f"mask_too_small ({mask_pixels} pixels)"
        else:
            # RGB frame input makes this an RGB mean, despite OpenCV's BGR
            # convention in other contexts.
            frame_means[roi_name] = cv2.mean(rgb_frame, mask=mask)[:3]

    if return_diagnostics:
        return frame_means, invalid_rois
    return None if invalid_rois else frame_means
