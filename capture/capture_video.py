"""Webcam preview, continuous capture, and real-time quality feedback."""

from datetime import datetime
from enum import Enum, auto
from pathlib import Path
import time

import cv2

from rPPG.capture.quality_check import (
    FaceFramingQualityChecker,
    MovementQualityChecker,
)
from rPPG.capture.lighting_quality import LightingQualityChecker
from rPPG.config import (
    FACE_MAX_CENTER_OFFSET_X,
    FACE_MAX_CENTER_OFFSET_Y,
    FACE_MAX_HEIGHT_RATIO,
    FACE_MAX_WIDTH_RATIO,
    FACE_MIN_HEIGHT_RATIO,
    FACE_MIN_WIDTH_RATIO,
    LIGHTING_BRIGHT_PIXEL_CHANNEL,
    LIGHTING_DARK_PIXEL_LUMINANCE,
    LIGHTING_LUMA_WEIGHTS,
    LIGHTING_UNIFORMITY_GRID_COLUMNS,
    LIGHTING_UNIFORMITY_GRID_ROWS,
    MODEL_PATH,
    MOVEMENT_THRESHOLD,
    MOVEMENT_WINDOW_SIZE,
    READY_STABLE_FRAMES,
    ROI_POINTS,
)
from rPPG.roi.face_detection import FaceDetector
from rPPG.roi.roi_extraction import extract_roi_means


class CaptureState(Enum):
    """User-visible stages of webcam capture."""

    PREVIEW = auto()
    CAPTURING = auto()


COLORS = {
    "ok": (0, 255, 0),
    "warning": (0, 165, 255),
    "error": (0, 0, 255),
    "info": (255, 255, 255),
}


def _draw_lines(frame, lines, color=COLORS["info"]):
    """Draw an OpenCV overlay; entries may provide their own BGR color."""
    for index, line in enumerate(lines):
        if isinstance(line, tuple):
            text, line_color = line
        else:
            text, line_color = line, color
        cv2.putText(frame, text, (20, 35 + index * 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, line_color, 2)


def _draw_face_bbox(frame, bbox, color):
    """Draw the landmark-derived bounding box for framing calibration."""
    if bbox is not None:
        x_min, y_min, x_max, y_max = bbox
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)


def _format_ratio(value):
    return "--" if value is None else f"{value * 100:.1f}%"


def _format_metric(value):
    return "--" if value is None else f"{value:.3f}"


def assess_live_roi_status(rgb_frame, landmarks):
    """Validate the configured ROI masks for preview feedback only.

    This is the same mask validation used by offline analysis. It does not
    accept/reject a capture frame and deliberately returns concise counts rather
    than ROI pixels or RGB arrays.
    """
    total = len(ROI_POINTS)
    if landmarks is None:
        return {"valid_count": 0, "total": total, "invalid_rois": tuple(ROI_POINTS)}
    means, invalid_rois = extract_roi_means(
        rgb_frame, landmarks, ROI_POINTS, return_diagnostics=True
    )
    return {
        "valid_count": len(means),
        "total": total,
        "invalid_rois": tuple(invalid_rois),
    }


def _quality_status_lines(movement, framing, lighting, roi_status):
    """Build short, colored instructions shared by preview and recording."""
    if framing.face_detected:
        face_line = ("FACE: OK - landmarks validos", COLORS["ok"])
    else:
        face_line = ("FACE: ERRO - nao detectada", COLORS["error"])

    if roi_status["valid_count"] == roi_status["total"]:
        roi_line = (f"ROIs: {roi_status['valid_count']}/{roi_status['total']} OK", COLORS["ok"])
    elif roi_status["valid_count"] == 0:
        roi_line = (f"ROIs: 0/{roi_status['total']} indisponiveis", COLORS["error"])
    else:
        invalid_names = ", ".join(name.upper() for name in roi_status["invalid_rois"])
        roi_line = (
            f"ROIs: {roi_status['valid_count']}/{roi_status['total']} - {invalid_names}",
            COLORS["warning"],
        )

    if not framing.face_detected:
        framing_line = ("Enquadramento: ERRO - rosto nao detectado", COLORS["error"])
    elif framing.framing_ok:
        framing_line = ("Enquadramento: OK - rosto enquadrado", COLORS["ok"])
    else:
        framing_line = (
            f"Enquadramento: AVISO - {framing.message.lower()}",
            COLORS["warning"],
        )

    if movement.movement_metric is None:
        movement_line = ("Movimento: AVISO - aguardando estabilidade", COLORS["warning"])
    elif movement.is_stable:
        movement_line = ("Movimento: OK - mantenha esta posicao", COLORS["ok"])
    else:
        movement_line = ("Movimento: AVISO - fique imovel", COLORS["warning"])

    if lighting.status == "METRICS_AVAILABLE":
        lighting_line = (
            "ILUMINACAO: MEDICAO ATIVA - sem quality gate calibrado",
            COLORS["warning"],
        )
    else:
        lighting_line = ("Iluminacao: ERRO - regiao facial indisponivel", COLORS["error"])

    return [
        ("STATUS DA CAPTURA", COLORS["info"]),
        face_line,
        roi_line,
        framing_line,
        movement_line,
        lighting_line,
        ("QUALIDADE: face/ROIs monitoradas; luz observacional", COLORS["info"]),
        ("Legenda: verde = OK | amarelo = aviso | vermelho = problema", COLORS["info"]),
    ]


def _live_metric_lines(movement, framing, lighting):
    """Format the real-time numerical diagnostics shown in the video overlay."""
    metric_text = "--" if movement.movement_metric is None else f"{movement.movement_metric:.4f}"
    return [
        (f"Movimento (medida): {metric_text}", COLORS["info"]),
        f"Threshold: {movement.movement_threshold:.4f}",
        f"Face: largura {_format_ratio(framing.face_width_ratio)} | altura {_format_ratio(framing.face_height_ratio)}",
        f"Centro: X {_format_ratio(framing.face_center_x)} | Y {_format_ratio(framing.face_center_y)}",
        f"Luz: media {_format_metric(lighting.mean_luminance)} | escura {_format_ratio(lighting.dark_pixel_ratio)}",
        f"BRIGHT PIXEL RATIO (>= {LIGHTING_BRIGHT_PIXEL_CHANNEL:.3f}): {_format_ratio(lighting.bright_pixel_ratio)}",
        f"Luz: uniformidade {_format_metric(lighting.illumination_uniformity)}",
    ]


def capture_video(camera_index=0, duration_s=30.0, output_dir=None):
    """Preview first, then record every camera frame after ENTER is pressed.

    Quality components only provide feedback. They never filter, pause, or
    restart the continuous MP4 once ``CAPTURING`` begins.
    """
    output_dir = (Path(output_dir) if output_dir else
                  Path(__file__).resolve().parents[1] / "data" / "captures")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"video_{datetime.now():%Y-%m-%d_%H-%M-%S}.mp4"
    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        raise RuntimeError("Nao foi possivel acessar a camera.")

    source_fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    detector = FaceDetector(MODEL_PATH)
    movement_checker = MovementQualityChecker(
        MOVEMENT_THRESHOLD, MOVEMENT_WINDOW_SIZE, READY_STABLE_FRAMES
    )
    framing_checker = FaceFramingQualityChecker(
        FACE_MIN_WIDTH_RATIO,
        FACE_MAX_WIDTH_RATIO,
        FACE_MIN_HEIGHT_RATIO,
        FACE_MAX_HEIGHT_RATIO,
        FACE_MAX_CENTER_OFFSET_X,
        FACE_MAX_CENTER_OFFSET_Y,
    )
    lighting_checker = LightingQualityChecker(
        LIGHTING_LUMA_WEIGHTS,
        LIGHTING_DARK_PIXEL_LUMINANCE,
        LIGHTING_BRIGHT_PIXEL_CHANNEL,
        LIGHTING_UNIFORMITY_GRID_ROWS,
        LIGHTING_UNIFORMITY_GRID_COLUMNS,
    )
    writer = None
    captured_frames = 0
    state = CaptureState.PREVIEW
    capture_start_time = None
    session_start_time = time.monotonic()
    try:
        print("Preview aberto. Pressione ENTER para iniciar a captura.")
        while True:
            success, frame = capture.read()
            if not success:
                break
            elapsed_ms = int((time.monotonic() - session_start_time) * 1000)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            landmarks = detector.detect(rgb_frame, elapsed_ms)
            movement = movement_checker.update(landmarks)
            framing = framing_checker.evaluate(landmarks, frame.shape)
            lighting = lighting_checker.evaluate(rgb_frame, framing.bbox)
            roi_status = assess_live_roi_status(rgb_frame, landmarks)

            metric_lines = _live_metric_lines(movement, framing, lighting)
            lines = _quality_status_lines(movement, framing, lighting, roi_status)
            if state is CaptureState.PREVIEW:
                lines = [
                    ("PREVIEW - ajuste sua posicao", COLORS["info"]),
                    ("ENTER = iniciar | Q = sair", COLORS["info"]),
                    *lines,
                ]
            else:
                lines = [
                    ("CAPTURANDO", COLORS["info"]),
                    *lines,
                ]
            lines.extend(metric_lines)

            if not framing.face_detected:
                color = COLORS["error"]
            elif not framing.framing_ok or not movement.is_stable:
                color = COLORS["warning"]
            else:
                color = COLORS["ok"]

            capture_finished = False
            if state is CaptureState.CAPTURING:
                # This is deliberately unconditional: quality never drops frames.
                writer.write(frame)
                captured_frames += 1
                remaining = max(0.0, duration_s - (time.monotonic() - capture_start_time))
                lines.append(f"Tempo restante: {remaining:.1f}s")
                capture_finished = remaining <= 0.0

            _draw_face_bbox(frame, framing.bbox, color)
            _draw_lines(frame, lines, color)
            cv2.imshow("rPPG - Captura", frame)
            if capture_finished:
                break
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if state is CaptureState.PREVIEW and key in (10, 13):
                height, width = frame.shape[:2]
                writer = cv2.VideoWriter(
                    str(output_path), cv2.VideoWriter_fourcc(*"mp4v"),
                    source_fps, (width, height),
                )
                if not writer.isOpened():
                    raise RuntimeError("Nao foi possivel criar o arquivo de video.")
                capture_start_time = time.monotonic()
                state = CaptureState.CAPTURING
                print("Captura iniciada. Qualidade sera monitorada durante a gravacao.")
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        detector.close()
        cv2.destroyAllWindows()

    if captured_frames < 2:
        if output_path.exists():
            output_path.unlink()
        raise RuntimeError("Captura cancelada antes de gravar frames suficientes.")
    print(f"Captura concluida: {captured_frames} frames gravados em {output_path}")
    return str(output_path)
