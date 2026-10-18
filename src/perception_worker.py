"""
(v2.0): Thread-Isolated Edge CV Inference Worker & Visual Servoing Estimator.

Runs YOLOv8n-seg ONNX Runtime inference + safe-landing-pad contour segmentation
on downward aerial frames in a background daemon thread, communicating with the
10 Hz MAVLink flight loop via non-blocking latest-item queues (maxsize=1).
"""
from dataclasses import dataclass
from pathlib import Path
import queue
import threading
import time
from typing import Optional, Tuple

import cv2
import numpy as np
import structlog

from synthetic_camera import SyntheticDownwardCamera

logger = structlog.get_logger()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ONNX_PATH = PROJECT_ROOT / "models" / "yolov8n-seg.onnx"


@dataclass
class FrameRequest:
    timestamp: float
    north_m: float
    east_m: float
    alt_agl_m: float
    yaw_rad: float
    frame_bgr: Optional[np.ndarray] = None


@dataclass
class VisualServoResult:
    timestamp: float
    inference_latency_ms: float
    safe_pad_found: bool
    obstacle_detected: bool
    obstacle_reason: str
    target_pixel_u: float
    target_pixel_v: float
    offset_north_m: float
    offset_east_m: float
    safe_area_ratio: float
    backend_used: str


class PerceptionWorker:
    """
    Thread-isolated producer-consumer perception engine.
    Guarantees zero blocking of the 10 Hz flight control state machine.
    """
    SERVO_KP = 0.65                 # Damped proportional gain for visual servoing
    MIN_SAFE_AREA_RATIO = 0.04      # Minimum fraction of frame that must be safe turf
    CENTER_HAZARD_RADIUS_PX = 140   # Central nadir radius (px) checked for dynamic obstacles

    def __init__(
        self,
        camera: Optional[SyntheticDownwardCamera] = None,
        onnx_path: Optional[Path] = None,
        use_gpu: bool = True,
    ):
        self.camera = camera or SyntheticDownwardCamera()
        self.onnx_path = Path(onnx_path) if onnx_path else DEFAULT_ONNX_PATH
        self.use_gpu = use_gpu

        self._frame_queue: queue.Queue[FrameRequest] = queue.Queue(maxsize=1)
        self._result_queue: queue.Queue[VisualServoResult] = queue.Queue(maxsize=1)
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self._ort_session = None
        self._input_name = None
        self.backend_name = "OpenCV-Morphological"
        self._init_onnx_session()

    def _init_onnx_session(self):
        """Initialize ONNX Runtime session (CUDAExecutionProvider -> CPU fallback)."""
        if not self.onnx_path.exists():
            logger.warning("ONNX_MODEL_NOT_FOUND", path=str(self.onnx_path), fallback="OpenCV-Morphological")
            return

        try:
            import onnxruntime as ort

            avail = ort.get_available_providers()
            providers = ["CPUExecutionProvider"]
            if self.use_gpu and "CUDAExecutionProvider" in avail:
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            ort.set_default_logger_severity(3)  # Suppress C++ EP fallback warnings
            self._ort_session = ort.InferenceSession(str(self.onnx_path), providers=providers)
            self._input_name = self._ort_session.get_inputs()[0].name
            active_ep = self._ort_session.get_providers()[0]
            self.backend_name = f"YOLOv8n-seg-ONNX({active_ep})"
            logger.info("PERCEPTION_ONNX_READY", backend=self.backend_name, model=str(self.onnx_path.name))
        except Exception as e:
            logger.warning("ONNX_INIT_FALLBACK", error=str(e))
            self._ort_session = None

    def start(self):
        """Start the background inference daemon thread."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._worker_loop, name="PerceptionWorker", daemon=True)
        self._thread.start()
        logger.info("PERCEPTION_WORKER_STARTED", backend=self.backend_name)

    def stop(self):
        """Stop the background worker thread gracefully."""
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.5)
        logger.info("PERCEPTION_WORKER_STOPPED")

    def submit_pose(
        self,
        north_m: float,
        east_m: float,
        alt_agl_m: float,
        yaw_rad: float = 0.0,
        frame_bgr: Optional[np.ndarray] = None,
    ) -> bool:
        """
        Non-blocking submission from the 10 Hz control loop.
        Drops any stale un-processed frame so the worker always processes the newest pose.
        """
        req = FrameRequest(
            timestamp=time.time(),
            north_m=north_m,
            east_m=east_m,
            alt_agl_m=alt_agl_m,
            yaw_rad=yaw_rad,
            frame_bgr=frame_bgr,
        )
        try:
            self._frame_queue.get_nowait()
        except queue.Empty:
            pass
        try:
            self._frame_queue.put_nowait(req)
            return True
        except queue.Full:
            return False

    def get_latest_result(self) -> Optional[VisualServoResult]:
        """Non-blocking poll called by the 10 Hz state machine."""
        try:
            return self._result_queue.get_nowait()
        except queue.Empty:
            return None

    def process_single_frame_sync(self, req: FrameRequest) -> VisualServoResult:
        """
        Executes one complete perception pass (ONNX YOLOv8n-seg + Safe-Zone Contour + Pinhole Servo Math).
        Can be called directly in unit tests or by the background worker thread.
        """
        t0 = time.perf_counter()

        frame = (
            req.frame_bgr
            if req.frame_bgr is not None
            else self.camera.render_frame(req.north_m, req.east_m, req.alt_agl_m, req.yaw_rad)
        )

        obstacle_detected = False
        obstacle_reason = ""

        # ── Step 1: Run YOLOv8n-seg ONNX Forward Pass (if session active) ──
        if self._ort_session is not None:
            blob = (frame.astype(np.float32) / 255.0).transpose(2, 0, 1)[np.newaxis, ...]
            outputs = self._ort_session.run(None, {self._input_name: blob})
            # outputs[0] has shape (1, 116, 8400): 4 box coords + 80 class scores + 32 mask coeffs
            preds = outputs[0][0]  # (116, 8400)
            class_scores = preds[4:84, :]
            max_scores = np.max(class_scores, axis=0)
            # Check if high-confidence COCO obstacle (person=0, car=2, motorcycle=3, truck=7) is near center
            obstacle_mask = max_scores > 0.45
            if np.any(obstacle_mask):
                boxes_cx = preds[0, obstacle_mask]
                boxes_cy = preds[1, obstacle_mask]
                dists_from_center = np.hypot(boxes_cx - 320.0, boxes_cy - 320.0)
                if np.any(dists_from_center < self.CENTER_HAZARD_RADIUS_PX):
                    obstacle_detected = True
                    obstacle_reason = "YOLOv8n-seg detected obstacle in landing footprint"

        # ── Step 2: High-Contrast Dynamic Hazard Check (catches synthetic red/dark vehicle/debris) ──
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Detect bright red hazard markers injected by SyntheticDownwardCamera.add_dynamic_obstacle
        red_mask = cv2.inRange(hsv, np.array([0, 120, 100]), np.array([12, 255, 255]))
        center_crop = red_mask[
            320 - self.CENTER_HAZARD_RADIUS_PX : 320 + self.CENTER_HAZARD_RADIUS_PX,
            320 - self.CENTER_HAZARD_RADIUS_PX : 320 + self.CENTER_HAZARD_RADIUS_PX,
        ]
        if np.count_nonzero(center_crop) > 80:
            obstacle_detected = True
            obstacle_reason = "Visual segmentation detected unmapped dynamic obstacle on landing pad"

        # ── Step 3: Safe Landing Pad Segmentation & Centroid Extraction ──
        # Preferred safe turf rendered in green HSV band (Hue 35..85)
        safe_mask = cv2.inRange(hsv, np.array([35, 40, 60]), np.array([85, 255, 255]))
        kernel = np.ones((5, 5), np.uint8)
        safe_mask = cv2.morphologyEx(safe_mask, cv2.MORPH_OPEN, kernel)
        safe_mask = cv2.morphologyEx(safe_mask, cv2.MORPH_CLOSE, kernel)

        safe_area_ratio = float(np.count_nonzero(safe_mask)) / float(safe_mask.size)
        contours, _ = cv2.findContours(safe_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        target_u, target_v = 320.0, 320.0
        safe_pad_found = False

        if contours and safe_area_ratio >= self.MIN_SAFE_AREA_RATIO:
            largest_contour = max(contours, key=cv2.contourArea)
            moments = cv2.moments(largest_contour)
            if moments["m00"] > 100.0:
                target_u = float(moments["m10"] / moments["m00"])
                target_v = float(moments["m01"] / moments["m00"])
                safe_pad_found = True

        # Convert pixel offset from image center (320, 320) -> damped NED metric correction
        raw_dn, raw_de = SyntheticDownwardCamera.pixel_to_ned_offset(
            target_u, target_v, req.alt_agl_m, req.yaw_rad
        )
        servo_dn = self.SERVO_KP * raw_dn if safe_pad_found else 0.0
        servo_de = self.SERVO_KP * raw_de if safe_pad_found else 0.0

        latency_ms = (time.perf_counter() - t0) * 1000.0

        return VisualServoResult(
            timestamp=time.time(),
            inference_latency_ms=round(latency_ms, 2),
            safe_pad_found=safe_pad_found,
            obstacle_detected=obstacle_detected,
            obstacle_reason=obstacle_reason,
            target_pixel_u=round(target_u, 1),
            target_pixel_v=round(target_v, 1),
            offset_north_m=round(servo_dn, 3),
            offset_east_m=round(servo_de, 3),
            safe_area_ratio=round(safe_area_ratio, 3),
            backend_used=self.backend_name,
        )

    def _worker_loop(self):
        while not self._stop_event.is_set():
            try:
                req = self._frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                res = self.process_single_frame_sync(req)
                # Publish latest result (drop oldest if unread)
                try:
                    self._result_queue.get_nowait()
                except queue.Empty:
                    pass
                self._result_queue.put_nowait(res)
            except Exception as e:
                logger.error("PERCEPTION_WORKER_ERROR", error=str(e))