"""
Pose-aware Synthetic Downward Aerial Camera Generator for SITL & Visual Servoing.
Renders an overhead terrain map anchored to the 60x50 safety grid (300m x 250m)
and extracts perspective-scaled 640x640 downward camera frames based on real-time
drone (north_m, east_m, alt_agl_m, yaw_rad) pose.
"""
import math
from pathlib import Path
from typing import Tuple, Optional, List, Dict

import cv2
import numpy as np

GRID_COLS = 60
GRID_ROWS = 50
TILE_SIZE_M = 5.0


class SyntheticDownwardCamera:
    """
    Simulates a downward-facing RGB camera mounted on the drone belly.
      - Image size: 640x640 px
      - Horizontal/Vertical FOV: 60.0 degrees
      - Origin (north_m=0, east_m=0) corresponds to grid cell (origin_x=30, origin_y=25)
    """
    IMG_SIZE_PX = 640
    FOV_DEG = 60.0
    WORLD_PX_PER_M = 8  # 8 pixels per meter -> 300m x 250m world = 2400 x 2000 px canvas

    def __init__(
        self,
        prob_matrix: Optional[np.ndarray] = None,
        origin_grid_x: int = 30,
        origin_grid_y: int = 25,
    ):
        self.origin_grid_x = origin_grid_x
        self.origin_grid_y = origin_grid_y
        self.world_width_m = GRID_COLS * TILE_SIZE_M   # 300.0 m
        self.world_height_m = GRID_ROWS * TILE_SIZE_M  # 250.0 m

        self.canvas_w = int(self.world_width_m * self.WORLD_PX_PER_M)   # 2400 px
        self.canvas_h = int(self.world_height_m * self.WORLD_PX_PER_M)  # 2000 px

        self.dynamic_obstacles: List[Dict] = []  # List of {"north_m": float, "east_m": float, "radius_m": float}
        self.world_canvas = self._build_world_canvas(prob_matrix)

    def _build_world_canvas(self, prob_matrix: Optional[np.ndarray]) -> np.ndarray:
        """Synthesize a deterministic, textured overhead RGB terrain map from the 60x50 safety grid."""
        rng = np.random.default_rng(42)
        if prob_matrix is None:
            prob_matrix = np.full((GRID_ROWS, GRID_COLS), 0.75, dtype=float)

        canvas = np.zeros((self.canvas_h, self.canvas_w, 3), dtype=np.uint8)
        tile_px = int(TILE_SIZE_M * self.WORLD_PX_PER_M)  # 40 px per 5m tile

        for gy in range(GRID_ROWS):
            for gx in range(GRID_COLS):
                p = float(prob_matrix[gy, gx])
                # Note: In image canvas, row 0 is South or North; we map gy directly to y0..y1
                y0, y1 = gy * tile_px, (gy + 1) * tile_px
                x0, x1 = gx * tile_px, (gx + 1) * tile_px

                if p > 0.80:
                    # Preferred safe zone: smooth flat grass / landing turf (BGR)
                    base_bgr = np.array([65, 165, 75], dtype=np.int16)
                    noise = rng.integers(-8, 9, size=(tile_px, tile_px, 3), dtype=np.int16)
                    patch = np.clip(base_bgr + noise, 0, 255).astype(np.uint8)
                    # Draw subtle landing zone boundary marker
                    cv2.rectangle(patch, (2, 2), (tile_px - 3, tile_px - 3), (85, 200, 95), 1)
                elif p >= 0.50:
                    # Conditional zone: dry soil / light scrub
                    base_bgr = np.array([85, 140, 155], dtype=np.int16)
                    noise = rng.integers(-18, 19, size=(tile_px, tile_px, 3), dtype=np.int16)
                    patch = np.clip(base_bgr + noise, 0, 255).astype(np.uint8)
                else:
                    # Restricted / hazardous zone: jagged rocks, steep slope shadows & obstacles
                    base_bgr = np.array([50, 55, 70], dtype=np.int16)
                    noise = rng.integers(-35, 36, size=(tile_px, tile_px, 3), dtype=np.int16)
                    patch = np.clip(base_bgr + noise, 0, 255).astype(np.uint8)
                    # Draw synthetic boulders / trees (high edge density)
                    for _ in range(4):
                        cx = int(rng.integers(6, tile_px - 6))
                        cy = int(rng.integers(6, tile_px - 6))
                        rad = int(rng.integers(3, 7))
                        cv2.circle(patch, (cx, cy), rad, (25, 30, 40), -1)

                canvas[y0:y1, x0:x1] = patch

        return canvas

    def add_dynamic_obstacle(self, north_m: float, east_m: float, radius_m: float = 2.0):
        """Inject a high-contrast unmapped obstacle (e.g., vehicle/debris) at a specific NED offset."""
        self.dynamic_obstacles.append({
            "north_m": float(north_m),
            "east_m": float(east_m),
            "radius_m": float(radius_m),
        })

    def clear_dynamic_obstacles(self):
        self.dynamic_obstacles.clear()

    @classmethod
    def compute_gsd_m_per_px(cls, alt_agl_m: float) -> float:
        """Compute Ground Sample Distance (meters per pixel) at a given AGL altitude."""
        eff_alt = max(float(alt_agl_m), 1.0)
        fov_rad = math.radians(cls.FOV_DEG)
        footprint_m = 2.0 * eff_alt * math.tan(fov_rad / 2.0)
        return footprint_m / cls.IMG_SIZE_PX

    @classmethod
    def pixel_to_ned_offset(
        cls, u_px: float, v_px: float, alt_agl_m: float, yaw_rad: float = 0.0
    ) -> Tuple[float, float]:
        """
        Convert pixel coordinates (u, v) in the 640x640 downward image into
        Earth-fixed (delta_north_m, delta_east_m) offsets relative to the drone's nadir point.
        """
        gsd = cls.compute_gsd_m_per_px(alt_agl_m)
        cx = cls.IMG_SIZE_PX / 2.0
        cy = cls.IMG_SIZE_PX / 2.0

        # In downward camera image: +u is Right (+Body Y), -v (up from center) is Forward (+Body X)
        body_forward_m = (cy - float(u_px if False else v_px)) * gsd
        body_right_m = (float(u_px) - cx) * gsd

        cos_y = math.cos(yaw_rad)
        sin_y = math.sin(yaw_rad)
        delta_north_m = body_forward_m * cos_y - body_right_m * sin_y
        delta_east_m = body_forward_m * sin_y + body_right_m * cos_y
        return delta_north_m, delta_east_m

    def render_frame(
        self, north_m: float = 0.0, east_m: float = 0.0, alt_agl_m: float = 20.0, yaw_rad: float = 0.0
    ) -> np.ndarray:
        """
        Render a 640x640 BGR downward camera frame for the drone's instantaneous pose.
        Note: Grid Y increases with North_m, Grid X increases with East_m.
        """
        world_x_m = (self.origin_grid_x * TILE_SIZE_M) + float(east_m)
        world_y_m = (self.origin_grid_y * TILE_SIZE_M) + float(north_m)

        center_px_x = int(np.clip(world_x_m * self.WORLD_PX_PER_M, 0, self.canvas_w - 1))
        center_px_y = int(np.clip(world_y_m * self.WORLD_PX_PER_M, 0, self.canvas_h - 1))

        eff_alt = max(float(alt_agl_m), 1.5)
        footprint_m = 2.0 * eff_alt * math.tan(math.radians(self.FOV_DEG / 2.0))
        half_crop_px = max(16, int((footprint_m * self.WORLD_PX_PER_M) / 2.0))

        # Work on a local copy if dynamic obstacles exist
        canvas = self.world_canvas
        if self.dynamic_obstacles:
            canvas = self.world_canvas.copy()
            for obs in self.dynamic_obstacles:
                ox_m = (self.origin_grid_x * TILE_SIZE_M) + obs["east_m"]
                oy_m = (self.origin_grid_y * TILE_SIZE_M) + obs["north_m"]
                opx = int(ox_m * self.WORLD_PX_PER_M)
                opy = int(oy_m * self.WORLD_PX_PER_M)
                orad = max(4, int(obs["radius_m"] * self.WORLD_PX_PER_M))
                # Draw high-contrast red/dark-gray vehicle/obstacle hazard
                cv2.circle(canvas, (opx, opy), orad, (20, 20, 180), -1)
                cv2.rectangle(
                    canvas, (opx - orad, opy - orad), (opx + orad, opy + orad), (10, 10, 220), 2
                )

        # Pad canvas to avoid boundary clipping near edges
        pad = half_crop_px + 4
        padded = cv2.copyMakeBorder(
            canvas, pad, pad, pad, pad, cv2.BORDER_REFLECT_101
        )
        cx_pad = center_px_x + pad
        cy_pad = center_px_y + pad

        crop = padded[
            cy_pad - half_crop_px : cy_pad + half_crop_px,
            cx_pad - half_crop_px : cx_pad + half_crop_px,
        ]
        # Flip vertically so +North (+world_y_m) points UP (row 0) in the downward camera frame
        crop = cv2.flip(crop, 0)

        frame = cv2.resize(crop, (self.IMG_SIZE_PX, self.IMG_SIZE_PX), interpolation=cv2.INTER_LINEAR)

        if abs(yaw_rad) > 1e-3:
            angle_deg = math.degrees(yaw_rad)
            rot_mat = cv2.getRotationMatrix2D(
                (self.IMG_SIZE_PX / 2.0, self.IMG_SIZE_PX / 2.0), angle_deg, 1.0
            )
            frame = cv2.warpAffine(
                frame, rot_mat, (self.IMG_SIZE_PX, self.IMG_SIZE_PX), borderMode=cv2.BORDER_REFLECT_101
            )

        return frame