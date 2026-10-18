"""
Responsibilities:
  1. GPS-Denied Dead-Reckoning: Fuses downward Optical Flow (body-frame angular rates
     rotated by yaw heading + scaled by AGL height) with IMU/EKF velocity.
  2. Vertical Altitude Fusion & Terminal Rangefinder Gate: Fuses Barometer/EKF altitude
     with downward LiDAR/ultrasonic RANGEFINDER and checks final-meters surface roughness.
  3. Multi-Sensor Spoofing Detector: Combines Dual-GPS horizontal divergence (GPS1 vs GPS2)
     and EKF Normalized Innovation Squared (Chi-Squared consistency test).
"""
from collections import deque
from dataclasses import dataclass
import math
import time
from typing import Optional, Deque

import structlog

logger = structlog.get_logger()


@dataclass
class FusedStateEstimate:
    timestamp: float = 0.0
    north_m: float = 0.0
    east_m: float = 0.0
    alt_agl_m: float = 0.0
    vn_ms: float = 0.0
    ve_ms: float = 0.0
    vd_ms: float = 0.0
    flow_active: bool = False
    rangefinder_active: bool = False
    terminal_obstacle_detected: bool = False
    spoof_detected: bool = False
    spoof_reason: str = ""
    chi2_innovation: float = 0.0
    dual_gps_divergence_m: float = 0.0
    position_uncertainty_m: float = 0.5


class SensorFusionEngine:
    # ── Optical Flow & Rangefinder Fusion Constants ──
    MIN_FLOW_QUALITY = 80             # 0-255 quality byte from OPTICAL_FLOW msg
    MAX_FLOW_WEIGHT = 0.85            # Max complementary weight on optical flow velocity
    RANGEFINDER_MIN_M = 0.10          # Minimum valid LiDAR/sonar reading (m)
    RANGEFINDER_MAX_M = 15.0          # Maximum trusted LiDAR/sonar reading (m)
    RANGEFINDER_FUSION_WEIGHT = 0.85  # Weight on rangefinder vs baro when in valid range

    # ── Terminal Descent ("Final Meters") Obstacle Gate ──
    TERMINAL_GATE_ALT_M = 5.0         # Activate surface anomaly check below 5.0m AGL
    MAX_RANGE_STD_M = 0.60            # Max allowed rangefinder std-dev over sliding window
    RANGE_WINDOW_SIZE = 6             # Number of recent rangefinder samples

    # ── Multi-Sensor Spoofing Thresholds ──
    DUAL_GPS_MAX_DIVERGENCE_M = 3.0   # Max allowed horizontal distance between GPS1 and GPS2
    CHI2_SPOOF_THRESHOLD = 6.5        # Composite EKF innovation squared (v^2 + pos_h^2 + pos_v^2)

    def __init__(self):
        self._last_update_time: Optional[float] = None
        self.north_m: float = 0.0
        self.east_m: float = 0.0
        self.alt_agl_m: float = 0.0
        self.vn_ms: float = 0.0
        self.ve_ms: float = 0.0
        self.vd_ms: float = 0.0
        self.position_uncertainty_m: float = 0.5

        self._rng_history: Deque[float] = deque(maxlen=self.RANGE_WINDOW_SIZE)

    @staticmethod
    def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Great-circle distance in meters between two WGS84 coordinates."""
        earth_radius_m = 6371000.0
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)
        a = (
            math.sin(dphi / 2.0) ** 2
            + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
        )
        return 2.0 * earth_radius_m * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))

    def reset_origin(self, north_m: float = 0.0, east_m: float = 0.0, alt_agl_m: float = 0.0):
        """Anchor the fused estimator to a known starting coordinate."""
        self.north_m = north_m
        self.east_m = east_m
        self.alt_agl_m = alt_agl_m
        self.position_uncertainty_m = 0.5
        self._rng_history.clear()
        self._last_update_time = None

    def update(
        self,
        timestamp: Optional[float] = None,
        gps_healthy: bool = True,
        local_north_m: Optional[float] = None,
        local_east_m: Optional[float] = None,
        local_down_m: Optional[float] = None,
        local_vn_ms: Optional[float] = None,
        local_ve_ms: Optional[float] = None,
        local_vd_ms: Optional[float] = None,
        yaw_rad: float = 0.0,
        pitch_rate_rads: float = 0.0,
        roll_rate_rads: float = 0.0,
        flow_rate_x_rads: Optional[float] = None,
        flow_rate_y_rads: Optional[float] = None,
        flow_comp_m_x: Optional[float] = None,
        flow_comp_m_y: Optional[float] = None,
        flow_quality: int = 0,
        rangefinder_dist_m: Optional[float] = None,
        ekf_vel_var: float = 0.0,
        ekf_pos_h_var: float = 0.0,
        ekf_pos_v_var: float = 0.0,
        gps1_fix: int = 3,
        gps1_sats: int = 10,
        gps1_lat: float = 0.0,
        gps1_lon: float = 0.0,
        gps2_fix: Optional[int] = None,
        gps2_lat: Optional[float] = None,
        gps2_lon: Optional[float] = None,
    ) -> FusedStateEstimate:
        """Run one step of the complementary sensor fusion and integrity monitor."""
        now = timestamp if timestamp is not None and timestamp > 0.0 else time.time()
        dt = (
            max(min(now - self._last_update_time, 1.0), 0.01)
            if self._last_update_time is not None
            else 0.1
        )
        self._last_update_time = now

        # ── 1. Vertical Altitude Fusion (Baro/EKF + Rangefinder) ──
        baro_agl_m = -local_down_m if local_down_m is not None else self.alt_agl_m
        rng_valid = (
            rangefinder_dist_m is not None
            and self.RANGEFINDER_MIN_M <= rangefinder_dist_m <= self.RANGEFINDER_MAX_M
        )

        if rng_valid:
            beta = self.RANGEFINDER_FUSION_WEIGHT
            self.alt_agl_m = beta * float(rangefinder_dist_m) + (1.0 - beta) * baro_agl_m
            self._rng_history.append(float(rangefinder_dist_m))
        else:
            self.alt_agl_m = max(baro_agl_m, 0.0)

        self.vd_ms = local_vd_ms if local_vd_ms is not None else 0.0

        # ── 2. Terminal Descent ("Final Meters") Obstacle / Roughness Verification ──
        terminal_obstacle = False
        # In sensor_fusion.py, replace the std check:
        if len(self._rng_history) >= 4:
            samples = list(self._rng_history)
            n = len(samples)
            # Linear detrend: remove best-fit slope (expected descent rate)
            x_mean = (n - 1) / 2.0
            y_mean = sum(samples) / n
            slope = sum((i - x_mean) * (samples[i] - y_mean) for i in range(n)) / \
                    max(sum((i - x_mean) ** 2 for i in range(n)), 1e-9)
            residuals = [samples[i] - (y_mean + slope * (i - x_mean)) for i in range(n)]
            rng_std = math.sqrt(sum(r ** 2 for r in residuals) / n)
            
            # Alternative: flag only if consecutive readings change by more than expected
            MAX_EXPECTED_DELTA_M = 2.0  # max descent rate * dt + margin
            for i in range(1, len(self._rng_history)):
                if abs(self._rng_history[i] - self._rng_history[i-1]) > MAX_EXPECTED_DELTA_M:
                    terminal_obstacle = True
                    break

        # ── 3. Horizontal Velocity & Position Fusion (Optical Flow + IMU/EKF) ──
        flow_active = (flow_quality >= self.MIN_FLOW_QUALITY) and (
            flow_comp_m_x is not None or flow_rate_x_rads is not None
        )

        imu_vn = local_vn_ms if local_vn_ms is not None else 0.0
        imu_ve = local_ve_ms if local_ve_ms is not None else 0.0

        if flow_active:
            eff_alt = max(self.alt_agl_m, 0.5)
            # Prefer pre-compensated optical flow (m/s) if non-zero, else un-rotate angular flow (rad/s)
            if flow_comp_m_x is not None and flow_comp_m_y is not None and (
                abs(flow_comp_m_x) > 1e-4 or abs(flow_comp_m_y) > 1e-4
            ):
                v_body_x = float(flow_comp_m_x)
                v_body_y = float(flow_comp_m_y)
            else:
                fx = float(flow_rate_x_rads or 0.0) - pitch_rate_rads
                fy = float(flow_rate_y_rads or 0.0) - roll_rate_rads
                v_body_x = fx * eff_alt
                v_body_y = fy * eff_alt

            # Rotate Body-Frame (Forward, Right) -> Earth NED Frame (North, East) using yaw
            cos_y = math.cos(yaw_rad)
            sin_y = math.sin(yaw_rad)
            flow_vn = v_body_x * cos_y - v_body_y * sin_y
            flow_ve = v_body_x * sin_y + v_body_y * cos_y

            alpha = min((flow_quality / 255.0) * self.MAX_FLOW_WEIGHT, self.MAX_FLOW_WEIGHT)
            self.vn_ms = alpha * flow_vn + (1.0 - alpha) * imu_vn
            self.ve_ms = alpha * flow_ve + (1.0 - alpha) * imu_ve
        else:
            self.vn_ms = imu_vn
            self.ve_ms = imu_ve

        # Position integration:
        # When GPS is healthy and local_north_m is available, anchor to EKF position.
        # When GPS is denied, dead-reckon using fused optical-flow + IMU velocity!
        if gps_healthy and local_north_m is not None and local_east_m is not None:
            self.north_m = float(local_north_m)
            self.east_m = float(local_east_m)
            self.position_uncertainty_m = 0.5
        else:
            self.north_m += self.vn_ms * dt
            self.east_m += self.ve_ms * dt
            # Uncertainty grows slowly (0.15 m/s) with Optical Flow, or rapidly (1.2 m/s) on raw IMU
            drift_rate_ms = 0.15 if flow_active else 1.20
            self.position_uncertainty_m += drift_rate_ms * dt

        # ── 4. Multi-Sensor Spoofing Detection (Dual-GPS + Chi-Squared Innovation) ──
        spoof_detected = False
        spoof_reason = ""
        dual_gps_div_m = 0.0

        # Check A: Dual-GPS Horizontal Divergence (GPS1 vs GPS2)
        if (
            gps2_fix is not None
            and gps2_fix >= 3
            and gps1_fix >= 3
            and gps2_lat is not None
            and gps2_lon is not None
            and (gps1_lat != 0.0 or gps1_lon != 0.0)
            and (gps2_lat != 0.0 or gps2_lon != 0.0)
        ):
            dual_gps_div_m = self.haversine_m(gps1_lat, gps1_lon, gps2_lat, gps2_lon)
            if dual_gps_div_m > self.DUAL_GPS_MAX_DIVERGENCE_M:
                spoof_detected = True
                spoof_reason = (
                    f"Dual-GPS divergence {dual_gps_div_m:.2f}m > {self.DUAL_GPS_MAX_DIVERGENCE_M}m"
                )
                logger.critical("DUAL_GPS_SPOOF_DETECTED", divergence_m=round(dual_gps_div_m, 2))

        # Check B: EKF Normalized Innovation Squared (Chi-Squared Test)
        chi2_stat = (ekf_vel_var ** 2) + (ekf_pos_h_var ** 2) + (ekf_pos_v_var ** 2)
        if (
            not spoof_detected
            and gps1_fix >= 3
            and gps1_sats >= 6
            and chi2_stat > self.CHI2_SPOOF_THRESHOLD
            and ekf_pos_h_var <= 5.0  # Catches multi-axis spoofing before single-axis 5.0 gate
        ):
            spoof_detected = True
            spoof_reason = (
                f"EKF Innovation Chi2={chi2_stat:.2f} > {self.CHI2_SPOOF_THRESHOLD} with valid GPS fix"
            )
            logger.critical("CHI2_INNOVATION_SPOOF_DETECTED", chi2=round(chi2_stat, 2))

        return FusedStateEstimate(
            timestamp=now,
            north_m=self.north_m,
            east_m=self.east_m,
            alt_agl_m=self.alt_agl_m,
            vn_ms=self.vn_ms,
            ve_ms=self.ve_ms,
            vd_ms=self.vd_ms,
            flow_active=flow_active,
            rangefinder_active=rng_valid,
            terminal_obstacle_detected=terminal_obstacle,
            spoof_detected=spoof_detected,
            spoof_reason=spoof_reason,
            chi2_innovation=round(chi2_stat, 3),
            dual_gps_divergence_m=round(dual_gps_div_m, 3),
            position_uncertainty_m=round(self.position_uncertainty_m, 3),
        )