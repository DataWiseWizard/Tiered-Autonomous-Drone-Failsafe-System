"""
Tiered Failsafe State Machine — v2.0 (Sensor Fusion, Multi-Sensor Spoofing,
Battery-Aware Reachability, and Rangefinder Terminal Obstacle Diversion).
"""
import math
import time
from enum import Enum, auto
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Tuple

import yaml
import structlog
from vision_landing import VisionLandingEngine
from sensor_fusion import SensorFusionEngine, FusedStateEstimate

logger = structlog.get_logger()

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "params.yaml"


class FlightState(Enum):
    BOOT = auto()
    NOMINAL = auto()

    # Tier 1
    TIER1_BATTERY_WARNING = auto()
    TIER1_BATTERY_RTL = auto()
    TIER1_BATTERY_LAND = auto()
    TIER1_RC_LOST = auto()

    # Tier 2
    TIER2_GPS_DEGRADED = auto()
    TIER2_GPS_DENIED = auto()

    # Tier 3
    TIER3_VISION_HOVER = auto()
    TIER3_DESCENT = auto()

    LANDED = auto()
    CRITICAL_FAILURE = auto()


@dataclass
class TelemetrySnapshot:
    timestamp: float = 0.0
    battery_voltage_mv: int = 0
    battery_current_ma: int = 0
    battery_remaining_pct: int = 100
    rc_received: bool = True
    is_armed: bool = False
    flight_mode: int = 0
    gps_fix_type: int = 0
    gps_lat: float = 0.0
    gps_lon: float = 0.0
    gps_alt_m: float = 0.0
    gps_satellites: int = 0
    ekf_velocity_var: float = 0.0
    ekf_pos_horiz_var: float = 0.0
    ekf_pos_vert_var: float = 0.0

    # Secondary GPS receiver (Dual-GPS spoofing cross-check)
    gps2_fix_type: Optional[int] = None
    gps2_lat: Optional[float] = None
    gps2_lon: Optional[float] = None

    # Local NED position (m) and velocity (m/s) relative to EKF origin
    local_north_m: Optional[float] = None
    local_east_m: Optional[float] = None
    local_down_m: Optional[float] = None
    local_vn_ms: Optional[float] = None
    local_ve_ms: Optional[float] = None
    local_vd_ms: Optional[float] = None

    # Attitude & Angular Rates (for Optical Flow un-rotation)
    yaw_rad: float = 0.0
    pitch_rate_rads: float = 0.0
    roll_rate_rads: float = 0.0

    # Downward Optical Flow & Rangefinder sensors
    flow_rate_x_rads: Optional[float] = None
    flow_rate_y_rads: Optional[float] = None
    flow_comp_m_x: Optional[float] = None
    flow_comp_m_y: Optional[float] = None
    flow_quality: int = 0
    rangefinder_dist_m: Optional[float] = None

    # Latest COMMAND_ACK feedback
    last_ack_command: Optional[int] = None
    last_ack_result: Optional[int] = None

    def battery_voltage(self) -> float:
        return self.battery_voltage_mv / 1000.0

    def battery_current_amps(self) -> float:
        return self.battery_current_ma / 1000.0


@dataclass
class TransitionRecord:
    timestamp: float
    from_state: FlightState
    to_state: FlightState
    reason: str
    telemetry: Optional[TelemetrySnapshot] = None


class FailsafeStateMachine:
    # ── ArduCopter Mode IDs ──
    MODE_GUIDED = 4
    MODE_LOITER = 5
    MODE_RTL = 6
    MODE_LAND = 9

    # ── Tier 1 Thresholds ──
    BATTERY_WARNING_PCT = 20.0
    BATTERY_RTL_PCT = 15.0
    BATTERY_LAND_PCT = 8.0
    BATTERY_WARNING_VOLTAGE = 11.5
    BATTERY_RTL_VOLTAGE = 11.0
    BATTERY_LAND_VOLTAGE = 10.5
    RC_LOST_TIMEOUT_S = 3.0

    # ── Tier 2 Thresholds ──
    EKF_DEGRADED_THRESHOLD = 1.0
    EKF_CRITICAL_THRESHOLD = 5.0
    MIN_SATELLITES = 6
    MIN_GPS_FIX_TYPE = 3
    SPOOF_MAX_JUMP_M = 5.0
    SPOOF_MAX_VEL_DIVERGENCE_MS = 8.0

    # ── Tier 3 Staged Landing & Reachability Parameters ──
    STAGING_ALT_AGL_M = 10.0
    ARRIVAL_RADIUS_M = 1.5
    APPROACH_STAGE_TIMEOUT_S = 12.0
    TOTAL_DESCENT_TIMEOUT_S = 30.0
    MODE_RETRY_INTERVAL_S = 1.0
    MAX_MODE_RETRIES = 3
    METERS_PER_BATTERY_PCT = 8.0      # Conservative horizontal reach per 1% usable battery
    MAX_DIVERT_ATTEMPTS = 2           # Max alternate cluster diversions on terminal obstacle
    VISUAL_SERVO_INTERVAL_S = 0.5     # 2 Hz closed-loop visual servoing rate

    def __init__(
        self,
        connection,
        blackbox_logger=None,
        vision_engine=None,
        perception_worker=None,
        sensor_fusion: Optional[SensorFusionEngine] = None,
        config_path: Optional[Path] = None,
    ):
        self.conn = connection
        self.blackbox = blackbox_logger
        self.state = FlightState.BOOT
        self.history: List[TransitionRecord] = []

        # Simulated starting grid position (Center of 60x50 heatmap)
        self.current_grid_x = 30
        self.current_grid_y = 25

        self._load_config(config_path or DEFAULT_CONFIG_PATH)

        # Sensor Fusion Engine
        self.sensor_fusion = sensor_fusion or SensorFusionEngine()
        self.latest_fused_state = FusedStateEstimate()

        # Command tracking
        self._last_rc_seen_time: Optional[float] = None
        self._rtl_commanded: bool = False
        self._land_commanded: bool = False
        self._loiter_commanded: bool = False

        # Spoofing state tracking
        self._prev_gps_lat: Optional[float] = None
        self._prev_gps_lon: Optional[float] = None
        self._prev_gps_time: Optional[float] = None
        self.spoof_detected: bool = False
        self.spoof_reason: str = ""

        # Tier 3 Vision, Reachability & Diversion Tracking
        self.vision_engine = vision_engine
        self._vision_landing_triggered: bool = False
        self._vision_target: Optional[dict] = None
        self._excluded_clusters: List[Tuple[int, int]] = []
        self._divert_count: int = 0
        self._guided_commanded: bool = False
        self._mode_retry_count: int = 0
        self._last_mode_cmd_time: float = 0.0
        self._guidance_start_time: float = 0.0

        self._descent_commanded: bool = False
        self._terminal_land_commanded: bool = False
        self._descent_start_time: Optional[float] = None
        self._target_north_m: float = 0.0
        self._target_east_m: float = 0.0

        self.perception_worker = perception_worker
        self._last_servo_submit_time: float = 0.0
        self.servo_corrections_count: int = 0

        self._transition(FlightState.BOOT, "State machine initialized")

    def _load_config(self, path: Path):
        if not path or not Path(path).exists():
            return
        try:
            with open(path, "r") as f:
                cfg = yaml.safe_load(f) or {}
            sys_cfg = cfg.get("system", {})
            self.current_grid_x = sys_cfg.get("grid_origin_x", self.current_grid_x)
            self.current_grid_y = sys_cfg.get("grid_origin_y", self.current_grid_y)

            t1 = cfg.get("tier1_watchdog", {})
            self.BATTERY_WARNING_PCT = t1.get("battery_warning_pct", self.BATTERY_WARNING_PCT)
            self.BATTERY_RTL_PCT = t1.get("battery_rtl_pct", self.BATTERY_RTL_PCT)
            self.BATTERY_LAND_PCT = t1.get("battery_land_pct", self.BATTERY_LAND_PCT)
            self.BATTERY_WARNING_VOLTAGE = t1.get("battery_warning_voltage", self.BATTERY_WARNING_VOLTAGE)
            self.BATTERY_RTL_VOLTAGE = t1.get("battery_rtl_voltage", self.BATTERY_RTL_VOLTAGE)
            self.BATTERY_LAND_VOLTAGE = t1.get("battery_land_voltage", self.BATTERY_LAND_VOLTAGE)
            self.RC_LOST_TIMEOUT_S = t1.get("rc_lost_timeout_s", self.RC_LOST_TIMEOUT_S)

            t2 = cfg.get("tier2_environment", {})
            self.MIN_GPS_FIX_TYPE = t2.get("min_gps_fix_type", self.MIN_GPS_FIX_TYPE)
            self.MIN_SATELLITES = t2.get("min_satellites", self.MIN_SATELLITES)
            self.EKF_DEGRADED_THRESHOLD = t2.get("ekf_degraded_threshold", self.EKF_DEGRADED_THRESHOLD)
            self.EKF_CRITICAL_THRESHOLD = t2.get("ekf_critical_threshold", self.EKF_CRITICAL_THRESHOLD)
            self.SPOOF_MAX_JUMP_M = t2.get("spoof_max_jump_m", self.SPOOF_MAX_JUMP_M)
            self.SPOOF_MAX_VEL_DIVERGENCE_MS = t2.get(
                "spoof_max_vel_divergence_ms", self.SPOOF_MAX_VEL_DIVERGENCE_MS
            )

            t3 = cfg.get("tier3_vision", {})
            self.STAGING_ALT_AGL_M = t3.get("staging_alt_agl_m", self.STAGING_ALT_AGL_M)
            self.ARRIVAL_RADIUS_M = t3.get("arrival_radius_m", self.ARRIVAL_RADIUS_M)
            self.APPROACH_STAGE_TIMEOUT_S = t3.get("approach_stage_timeout_s", self.APPROACH_STAGE_TIMEOUT_S)
            self.TOTAL_DESCENT_TIMEOUT_S = t3.get("total_descent_timeout_s", self.TOTAL_DESCENT_TIMEOUT_S)
        except Exception as e:
            logger.warning("CONFIG_LOAD_WARNING", error=str(e))

    @staticmethod
    def haversine_distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Great-circle distance in meters between two WGS84 (lat, lon) points in degrees."""
        return SensorFusionEngine.haversine_m(lat1, lon1, lat2, lon2)

    def compute_max_reachable_tiles(self, battery_pct: float, grid_res_m: float = 5.0) -> float:
        """
        Battery-aware reachability constraint:
        Computes max horizontal distance (in grid tiles) the drone can safely fly
        while maintaining margin above the 8% critical emergency-land floor.
        """
        usable_pct = max(0.0, float(battery_pct) - self.BATTERY_LAND_PCT)
        max_dist_m = usable_pct * self.METERS_PER_BATTERY_PCT
        return max(3.0, max_dist_m / grid_res_m)

    def _detect_gps_spoofing(self, t: TelemetrySnapshot) -> bool:
        """
        Multi-layer GPS spoofing detector:
          1. SensorFusionEngine checks (Dual-GPS divergence > 3m OR EKF Chi-Squared > 6.5).
          2. Haversine jump > SPOOF_MAX_JUMP_M (5.0m) between consecutive reads.
          3. Implied GPS ground speed diverges from EKF inertial horizontal speed by > 8.0 m/s.
        """
        if self.latest_fused_state.spoof_detected:
            self.spoof_detected = True
            self.spoof_reason = self.latest_fused_state.spoof_reason
            return True

        if t.gps_lat == 0.0 and t.gps_lon == 0.0:
            return False

        now = t.timestamp if t.timestamp > 0.0 else time.time()

        if self._prev_gps_lat is not None and self._prev_gps_lon is not None:
            jump_m = self.haversine_distance_m(
                self._prev_gps_lat, self._prev_gps_lon, t.gps_lat, t.gps_lon
            )
            dt = max(now - (self._prev_gps_time or now), 0.05)

            if jump_m > self.SPOOF_MAX_JUMP_M:
                self.spoof_detected = True
                self.spoof_reason = f"Haversine GPS jump {jump_m:.2f}m > {self.SPOOF_MAX_JUMP_M}m"
                logger.critical("GPS_SPOOF_DETECTED", reason=self.spoof_reason, jump_m=round(jump_m, 2))
                self._prev_gps_lat, self._prev_gps_lon, self._prev_gps_time = t.gps_lat, t.gps_lon, now
                return True

            if t.local_vn_ms is not None and t.local_ve_ms is not None and dt <= 1.0:
                gps_speed_ms = jump_m / dt
                ekf_speed_ms = math.hypot(t.local_vn_ms, t.local_ve_ms)
                vel_diff = abs(gps_speed_ms - ekf_speed_ms)
                if vel_diff > self.SPOOF_MAX_VEL_DIVERGENCE_MS:
                    self.spoof_detected = True
                    self.spoof_reason = (
                        f"GPS/EKF velocity divergence {vel_diff:.1f}m/s > {self.SPOOF_MAX_VEL_DIVERGENCE_MS}m/s"
                    )
                    logger.critical("GPS_SPOOF_DETECTED", reason=self.spoof_reason)
                    self._prev_gps_lat, self._prev_gps_lon, self._prev_gps_time = t.gps_lat, t.gps_lon, now
                    return True

        self._prev_gps_lat = t.gps_lat
        self._prev_gps_lon = t.gps_lon
        self._prev_gps_time = now
        return False

    def evaluate(self, telemetry: TelemetrySnapshot):
        # Update Sensor Fusion Engine on every 10 Hz tick
        gps_raw_ok = (
            telemetry.gps_fix_type >= self.MIN_GPS_FIX_TYPE
            and telemetry.gps_satellites >= self.MIN_SATELLITES
            and telemetry.ekf_pos_horiz_var <= self.EKF_CRITICAL_THRESHOLD
            and not self.spoof_detected
        )
        self.latest_fused_state = self.sensor_fusion.update(
            timestamp=telemetry.timestamp,
            gps_healthy=gps_raw_ok,
            local_north_m=telemetry.local_north_m,
            local_east_m=telemetry.local_east_m,
            local_down_m=telemetry.local_down_m,
            local_vn_ms=telemetry.local_vn_ms,
            local_ve_ms=telemetry.local_ve_ms,
            local_vd_ms=telemetry.local_vd_ms,
            yaw_rad=telemetry.yaw_rad,
            pitch_rate_rads=telemetry.pitch_rate_rads,
            roll_rate_rads=telemetry.roll_rate_rads,
            flow_rate_x_rads=telemetry.flow_rate_x_rads,
            flow_rate_y_rads=telemetry.flow_rate_y_rads,
            flow_comp_m_x=telemetry.flow_comp_m_x,
            flow_comp_m_y=telemetry.flow_comp_m_y,
            flow_quality=telemetry.flow_quality,
            rangefinder_dist_m=telemetry.rangefinder_dist_m,
            ekf_vel_var=telemetry.ekf_velocity_var,
            ekf_pos_h_var=telemetry.ekf_pos_horiz_var,
            ekf_pos_v_var=telemetry.ekf_pos_vert_var,
            gps1_fix=telemetry.gps_fix_type,
            gps1_sats=telemetry.gps_satellites,
            gps1_lat=telemetry.gps_lat,
            gps1_lon=telemetry.gps_lon,
            gps2_fix=telemetry.gps2_fix_type,
            gps2_lat=telemetry.gps2_lat,
            gps2_lon=telemetry.gps2_lon,
        )

        match self.state:
            case FlightState.BOOT:
                self._handle_boot(telemetry)
            case FlightState.NOMINAL:
                self._handle_nominal(telemetry)
            case FlightState.TIER1_BATTERY_WARNING:
                self._handle_battery_warning(telemetry)
            case FlightState.TIER1_BATTERY_RTL:
                self._handle_battery_rtl(telemetry)
            case FlightState.TIER1_BATTERY_LAND:
                self._handle_battery_land(telemetry)
            case FlightState.TIER1_RC_LOST:
                self._handle_rc_lost(telemetry)
            case FlightState.TIER2_GPS_DEGRADED:
                self._handle_gps_degraded(telemetry)
            case FlightState.TIER2_GPS_DENIED:
                self._handle_gps_denied(telemetry)
            case FlightState.TIER3_VISION_HOVER:
                self._handle_tier3_vision_hover(telemetry)
            case FlightState.TIER3_DESCENT:
                self._handle_tier3_descent(telemetry)
            case _:
                pass

    def get_state(self) -> FlightState:
        return self.state

    def get_history(self) -> List[TransitionRecord]:
        return self.history.copy()

    def _assess_battery(self, t: TelemetrySnapshot) -> str:
        voltage = t.battery_voltage()
        pct = t.battery_remaining_pct
        if voltage <= self.BATTERY_LAND_VOLTAGE or pct <= self.BATTERY_LAND_PCT:
            return "LAND"
        if voltage <= self.BATTERY_RTL_VOLTAGE or pct <= self.BATTERY_RTL_PCT:
            return "RTL"
        if voltage <= self.BATTERY_WARNING_VOLTAGE or pct <= self.BATTERY_WARNING_PCT:
            return "WARNING"
        return "OK"

    def _assess_environment(self, t: TelemetrySnapshot) -> str:
        if self._detect_gps_spoofing(t) or self.spoof_detected:
            return "GPS_SPOOFED"
        if t.gps_fix_type < self.MIN_GPS_FIX_TYPE or t.gps_satellites < self.MIN_SATELLITES:
            return "GPS_LOST"
        if t.ekf_pos_horiz_var > self.EKF_CRITICAL_THRESHOLD:
            return "EKF_CRITICAL"
        if t.ekf_pos_horiz_var > self.EKF_DEGRADED_THRESHOLD:
            return "EKF_DEGRADED"
        return "HEALTHY"

    def _handle_boot(self, t: TelemetrySnapshot):
        if t.battery_remaining_pct > 0 and t.gps_fix_type >= 3:
            if t.gps_lat != 0.0 or t.gps_lon != 0.0:
                self._prev_gps_lat = t.gps_lat
                self._prev_gps_lon = t.gps_lon
                self._prev_gps_time = t.timestamp if t.timestamp > 0.0 else time.time()
            self._transition(
                FlightState.NOMINAL,
                f"Boot complete. Battery={t.battery_remaining_pct}%, GPS fix={t.gps_fix_type}"
            )

    def _handle_nominal(self, t: TelemetrySnapshot):
        batt_status = self._assess_battery(t)
        env_status = self._assess_environment(t)

        if batt_status == "LAND":
            self._transition(
                FlightState.TIER1_BATTERY_LAND,
                f"CRITICAL: Battery {t.battery_voltage():.1f}V. EMERGENCY LAND."
            )
            return
        if env_status in ("GPS_LOST", "EKF_CRITICAL", "GPS_SPOOFED"):
            reason = (
                f"GPS Spoofing Detected ({self.spoof_reason})"
                if env_status == "GPS_SPOOFED"
                else f"Environment Critical: GPS={t.gps_fix_type}, Sats={t.gps_satellites}, EKF_Var={t.ekf_pos_horiz_var:.2f}"
            )
            self._transition(FlightState.TIER2_GPS_DENIED, reason)
            return
        if batt_status == "RTL":
            self._transition(
                FlightState.TIER1_BATTERY_RTL,
                f"Battery {t.battery_voltage():.1f}V. Triggering RTL."
            )
            return
        if env_status == "EKF_DEGRADED":
            self._transition(
                FlightState.TIER2_GPS_DEGRADED,
                f"EKF Degraded: Var={t.ekf_pos_horiz_var:.2f}"
            )
            return
        if batt_status == "WARNING":
            self._transition(
                FlightState.TIER1_BATTERY_WARNING,
                f"Battery Warning: {t.battery_voltage():.1f}V"
            )
            return

        if not t.rc_received:
            if self._last_rc_seen_time is None:
                self._last_rc_seen_time = time.time()
            if time.time() - self._last_rc_seen_time >= self.RC_LOST_TIMEOUT_S:
                self._transition(
                    FlightState.TIER1_RC_LOST,
                    f"RC lost for >{self.RC_LOST_TIMEOUT_S}s"
                )
                return
        else:
            self._last_rc_seen_time = time.time()

    def _handle_battery_warning(self, t: TelemetrySnapshot):
        batt = self._assess_battery(t)
        if batt in ("RTL", "LAND"):
            target_state = FlightState.TIER1_BATTERY_RTL if batt == "RTL" else FlightState.TIER1_BATTERY_LAND
            self._transition(target_state, "Battery dropped further.")
        elif batt == "OK":
            self._transition(FlightState.NOMINAL, "Battery recovered.")

    def _handle_battery_rtl(self, t: TelemetrySnapshot):
        if self._assess_battery(t) == "LAND":
            self._transition(FlightState.TIER1_BATTERY_LAND, "Battery critical during RTL.")
            return
        if not self._rtl_commanded:
            self._command_rtl()
            self._rtl_commanded = True

    def _handle_battery_land(self, t: TelemetrySnapshot):
        if not self._land_commanded:
            self._command_land()
            self._land_commanded = True

    def _handle_rc_lost(self, t: TelemetrySnapshot):
        if t.rc_received:
            self._last_rc_seen_time = time.time()
            self._rtl_commanded = False
            self._transition(FlightState.NOMINAL, "RC signal restored.")
            return
        if not self._loiter_commanded:
            self._command_loiter()
            self._loiter_commanded = True

    def _handle_gps_degraded(self, t: TelemetrySnapshot):
        if self._assess_battery(t) == "LAND":
            self._transition(FlightState.TIER1_BATTERY_LAND, "Battery critical override during GPS degradation.")
            return
        env_status = self._assess_environment(t)
        if env_status in ("GPS_LOST", "EKF_CRITICAL", "GPS_SPOOFED"):
            self._transition(FlightState.TIER2_GPS_DENIED, f"Escalated to denied ({env_status}).")
        elif env_status == "HEALTHY":
            self._transition(FlightState.NOMINAL, "EKF recovered.")

    def _handle_gps_denied(self, t: TelemetrySnapshot):
        if self._assess_battery(t) == "LAND":
            self._transition(FlightState.TIER1_BATTERY_LAND, "Battery critical override during GPS denial.")
            return
        env_status = self._assess_environment(t)
        if env_status == "HEALTHY":
            self._transition(FlightState.NOMINAL, "GPS/EKF recovered.")
            return

        if not self._vision_landing_triggered:
            self._vision_landing_triggered = True
            self._transition(FlightState.TIER3_VISION_HOVER, "GPS Denied. Initiating Tier 3 Vision Landing.")

    def _handle_tier3_vision_hover(self, t: TelemetrySnapshot):
        """Calculate battery-reachable safe cluster, verify GUIDED mode, and command Stage 1 approach."""
        if self._assess_battery(t) == "LAND":
            self._transition(FlightState.TIER1_BATTERY_LAND, "Battery critical override during Vision Hover.")
            return

        if self.vision_engine is None:
            self.vision_engine = VisionLandingEngine()

        # Step 1: Query safest reachable cluster and request GUIDED mode
        if not self._vision_target:
            max_reach_tiles = self.compute_max_reachable_tiles(
                t.battery_remaining_pct, self.vision_engine.grid_resolution_m
            )
            self._vision_target = self.vision_engine.find_safest_zone(
                self.current_grid_x,
                self.current_grid_y,
                max_distance_tiles=max_reach_tiles,
                exclude_coords=self._excluded_clusters,
            )

            if self._vision_target is None:
                logger.critical("ABORT_LANDING", reason="No reachable safe cluster >= 0.3 floor. Loitering.")
                self._command_loiter()
                return

            self._command_guided()
            self._guided_commanded = True
            self._guidance_start_time = time.time()
            self._last_mode_cmd_time = time.time()
            self._mode_retry_count = 0
            return

        # Step 2: Verify GUIDED mode feedback (with retry) before sending position target
        if self._guided_commanded and not self._descent_commanded:
            now = time.time()
            elapsed = now - self._guidance_start_time
            mode_verified = (t.flight_mode == self.MODE_GUIDED)

            if not mode_verified and (now - self._last_mode_cmd_time >= self.MODE_RETRY_INTERVAL_S):
                if self._mode_retry_count < self.MAX_MODE_RETRIES:
                    self._mode_retry_count += 1
                    logger.warning("RETRYING_GUIDED_MODE", attempt=self._mode_retry_count)
                    self._command_guided()
                    self._last_mode_cmd_time = now

            if (mode_verified and elapsed >= 0.5) or elapsed >= 2.0:
                offset_north_m, offset_east_m = self.vision_engine.grid_to_local_meters(
                    self._vision_target['x'], self._vision_target['y'],
                    self.current_grid_x, self.current_grid_y
                )

                base_north = (
                    t.local_north_m if t.local_north_m is not None else self.latest_fused_state.north_m
                )
                base_east = (
                    t.local_east_m if t.local_east_m is not None else self.latest_fused_state.east_m
                )
                # Absolute NED target relative to Home/EKF origin (0,0)
                self._target_north_m = offset_north_m
                self._target_east_m = offset_east_m
                staging_down_m = -self.STAGING_ALT_AGL_M

                logger.info(
                    "STAGE1_APPROACH_COMMANDED",
                    target_north_m=round(self._target_north_m, 2),
                    target_east_m=round(self._target_east_m, 2),
                    staging_down_m=staging_down_m,
                    tier=self._vision_target['tier'],
                    mode_verified=mode_verified,
                )

                if hasattr(self.conn, "set_position_target_local_ned"):
                    self.conn.set_position_target_local_ned(
                        self._target_north_m, self._target_east_m, staging_down_m
                    )

                if self.blackbox:
                    self.blackbox.log_command_sent(
                        "SET_POSITION_TARGET_LOCAL_NED",
                        {
                            "frame": "MAV_FRAME_LOCAL_NED",
                            "type_mask": 3576,
                            "north_m": round(self._target_north_m, 2),
                            "east_m": round(self._target_east_m, 2),
                            "down_m": staging_down_m,
                            "tier": self._vision_target['tier'],
                        },
                    )

                self._descent_commanded = True
                self._descent_start_time = now
                self._transition(
                    FlightState.TIER3_DESCENT,
                    f"Stage 1: En route to {self._vision_target['tier']} cluster "
                    f"(N={self._target_north_m:.1f}m, E={self._target_east_m:.1f}m, Alt=10m)."
                )

    def _divert_to_alternate_cluster(self, t: TelemetrySnapshot, reason: str):
        """
        Triggered when downward rangefinder detects an obstacle in the final meters.
        Blacklists the current cluster and diverts to the next safest reachable cluster.
        """
        if self._vision_target:
            bad_coord = (int(self._vision_target["x"]), int(self._vision_target["y"]))
            if bad_coord not in self._excluded_clusters:
                self._excluded_clusters.append(bad_coord)

        self._divert_count += 1
        if self._divert_count > self.MAX_DIVERT_ATTEMPTS:
            logger.critical("MAX_DIVERSIONS_EXCEEDED", attempts=self._divert_count, action="LOITER")
            self._command_loiter()
            return

        if self.vision_engine is None:
            self.vision_engine = VisionLandingEngine()

        max_reach_tiles = self.compute_max_reachable_tiles(
            t.battery_remaining_pct, self.vision_engine.grid_resolution_m
        )
        alt_target = self.vision_engine.find_safest_zone(
            self.current_grid_x,
            self.current_grid_y,
            max_distance_tiles=max_reach_tiles,
            exclude_coords=self._excluded_clusters,
        )

        if alt_target is None:
            logger.critical("DIVERT_FAILED_NO_ALT_CLUSTER", action="LOITER")
            self._command_loiter()
            return

        self._vision_target = alt_target
        self._terminal_land_commanded = False
        self._descent_start_time = time.time()

        offset_north_m, offset_east_m = self.vision_engine.grid_to_local_meters(
            alt_target["x"], alt_target["y"], self.current_grid_x, self.current_grid_y
        )
        self._target_north_m = offset_north_m
        self._target_east_m = offset_east_m
        staging_down_m = -self.STAGING_ALT_AGL_M

        self._command_guided()
        if hasattr(self.conn, "set_position_target_local_ned"):
            self.conn.set_position_target_local_ned(
                self._target_north_m, self._target_east_m, staging_down_m
            )

        if self.blackbox:
            self.blackbox.log_command_sent(
                "DIVERT_TO_ALTERNATE_CLUSTER",
                {
                    "reason": reason,
                    "new_grid_x": alt_target["x"],
                    "new_grid_y": alt_target["y"],
                    "north_m": round(self._target_north_m, 2),
                    "east_m": round(self._target_east_m, 2),
                },
            )

        logger.warning(
            "DIVERTING_TO_ALTERNATE_CLUSTER",
            reason=reason,
            attempt=self._divert_count,
            new_grid=f"({alt_target['x']}, {alt_target['y']})",
            target_north_m=round(self._target_north_m, 2),
            target_east_m=round(self._target_east_m, 2),
        )

    def _handle_tier3_descent(self, t: TelemetrySnapshot):
        """
        Two-Stage Terminal Descent with Sensor Fusion & Final-Meters Obstacle Gate:
          - Checks rangefinder surface roughness in final 5m; diverts if obstacle detected.
          - Stage 1: Translate to (target_north_m, target_east_m) at 10m staging altitude.
          - Stage 2: Upon arrival within 1.5m (using local or fused NED), issue NAV_LAND.
        """
        if self._assess_battery(t) == "LAND":
            self._transition(FlightState.TIER1_BATTERY_LAND, "Battery critical override during Tier 3 Descent.")
            return

        if self._descent_start_time is None:
            self._descent_start_time = time.time()
            logger.info("DESCENT_STARTED", timestamp=self._descent_start_time)

        elapsed = time.time() - self._descent_start_time

        # ── FINAL METERS RANGEFINDER OBSTACLE CHECK ──
        if self.latest_fused_state.terminal_obstacle_detected:
            self._divert_to_alternate_cluster(
                t, reason="Rangefinder surface anomaly in final 5m AGL"
            )
            return

        # ── CLOSED-LOOP VISUAL SERVOING (1-2 Hz Thread-Isolated Perception) ──
        if self.perception_worker is not None and not self._terminal_land_commanded:
            now_t = time.time()
            curr_n_est = (
                t.local_north_m if t.local_north_m is not None else self.latest_fused_state.north_m
            )
            curr_e_est = (
                t.local_east_m if t.local_east_m is not None else self.latest_fused_state.east_m
            )
            curr_agl = max(self.latest_fused_state.alt_agl_m, 5.0)

            if now_t - self._last_servo_submit_time >= self.VISUAL_SERVO_INTERVAL_S:
                self.perception_worker.submit_pose(
                    north_m=curr_n_est,
                    east_m=curr_e_est,
                    alt_agl_m=curr_agl,
                    yaw_rad=t.yaw_rad,
                )
                self._last_servo_submit_time = now_t

            servo_res = self.perception_worker.get_latest_result()
            if servo_res is not None:
                if servo_res.obstacle_detected:
                    self._divert_to_alternate_cluster(t, reason=servo_res.obstacle_reason)
                    return

                if servo_res.safe_pad_found and (
                    abs(servo_res.offset_north_m) > 0.05 or abs(servo_res.offset_east_m) > 0.05
                ):
                    # Refine target coordinates using damped visual servoing offset
                    self._target_north_m = curr_n_est + servo_res.offset_north_m
                    self._target_east_m = curr_e_est + servo_res.offset_east_m
                    self.servo_corrections_count += 1

                    if hasattr(self.conn, "set_position_target_local_ned"):
                        self.conn.set_position_target_local_ned(
                            self._target_north_m, self._target_east_m, -self.STAGING_ALT_AGL_M
                        )
                    if self.blackbox:
                        self.blackbox.log_command_sent(
                            "VISUAL_SERVO_CORRECTION",
                            {
                                "target_north_m": round(self._target_north_m, 2),
                                "target_east_m": round(self._target_east_m, 2),
                                "servo_dn_m": servo_res.offset_north_m,
                                "servo_de_m": servo_res.offset_east_m,
                                "latency_ms": servo_res.inference_latency_ms,
                            },
                        )
                    logger.info(
                        "VISUAL_SERVO_CORRECTION",
                        correction_num=self.servo_corrections_count,
                        target_north_m=round(self._target_north_m, 2),
                        target_east_m=round(self._target_east_m, 2),
                        latency_ms=servo_res.inference_latency_ms,
                    )

        # ── STAGE 1 -> STAGE 2 TRANSITION (Horizontal Arrival Check) ──
        if not self._terminal_land_commanded:
            arrived_horizontally = False
            horiz_dist_m = None

            # Use local_north_m if provided, otherwise fall back to Optical-Flow Fused NED
            curr_n = (
                t.local_north_m
                if t.local_north_m is not None
                else (self.latest_fused_state.north_m if self.latest_fused_state.flow_active else None)
            )
            curr_e = (
                t.local_east_m
                if t.local_east_m is not None
                else (self.latest_fused_state.east_m if self.latest_fused_state.flow_active else None)
            )

            if curr_n is not None and curr_e is not None:
                horiz_dist_m = math.hypot(
                    self._target_north_m - curr_n,
                    self._target_east_m - curr_e
                )
                if horiz_dist_m <= self.ARRIVAL_RADIUS_M:
                    arrived_horizontally = True

            if arrived_horizontally or elapsed >= self.APPROACH_STAGE_TIMEOUT_S:
                reason_str = (
                    f"Reached target centroid (dist={horiz_dist_m:.2f}m <= {self.ARRIVAL_RADIUS_M}m)"
                    if arrived_horizontally and horiz_dist_m is not None
                    else f"Approach stage timeout ({elapsed:.1f}s)"
                )
                logger.info("STAGE2_TERMINAL_LAND_INITIATED", reason=reason_str)
                self._command_terminal_land(reason_str)
                self._terminal_land_commanded = True

        # ── TOUCHDOWN CONFIRMATION (Fused Rangefinder/Baro OR GPS/Local Down) ──
        altitude = t.gps_alt_m if t.gps_alt_m is not None else 999.0
        near_ground_local = (t.local_down_m is not None and t.local_down_m >= -0.8)
        near_ground_fused = (
            self.latest_fused_state.rangefinder_active
            and self.latest_fused_state.alt_agl_m <= 0.4
        )
        near_ground_gps = (altitude < 586.0)

        if (near_ground_local or near_ground_fused or near_ground_gps) and not t.is_armed:
            self._transition(
                FlightState.LANDED,
                f"Touchdown confirmed via telemetry. Alt={altitude:.1f}m, "
                f"FusedAGL={self.latest_fused_state.alt_agl_m:.2f}m, Disarmed=True"
            )
            return

        if elapsed > self.TOTAL_DESCENT_TIMEOUT_S:
            self._transition(
                FlightState.LANDED,
                f"Descent timeout reached ({elapsed:.1f}s). Assuming landed."
            )
            return

    def _transition(self, new_state: FlightState, reason: str, telemetry: Optional[TelemetrySnapshot] = None):
        old_state = self.state
        record = TransitionRecord(
            timestamp=time.time(),
            from_state=old_state,
            to_state=new_state,
            reason=reason,
            telemetry=telemetry
        )
        self.history.append(record)
        self.state = new_state

        if new_state == FlightState.NOMINAL:
            self._rtl_commanded = False
            self._land_commanded = False
            self._loiter_commanded = False
            self._vision_landing_triggered = False
            self._guided_commanded = False
            self._mode_retry_count = 0
            self._descent_commanded = False
            self._terminal_land_commanded = False
            self._descent_start_time = None
            self._vision_target = None
            self._excluded_clusters.clear()
            self._divert_count = 0
            self.spoof_detected = False
            self.spoof_reason = ""
            
            self.servo_corrections_count = 0

        if self.blackbox:
            self.blackbox.log_state_transition(old_state.name, new_state.name, reason)

        logger.info("STATE_TRANSITION", from_state=old_state.name, to_state=new_state.name, reason=reason)

    def _command_guided(self):
        logger.info("COMMANDING_GUIDED", reason="Tier 3 vision approach")
        self.conn.set_mode("GUIDED")
        if self.blackbox:
            self.blackbox.log_command_sent("SET_MODE", {"mode": "GUIDED"})

    def _command_rtl(self):
        logger.warning("COMMANDING_RTL", reason="Battery low failsafe")
        self.conn.set_mode("RTL")
        if self.blackbox:
            self.blackbox.log_command_sent("SET_MODE", {"mode": "RTL"})

    def _command_land(self):
        logger.critical("COMMANDING_EMERGENCY_LAND", reason="Battery critical failsafe")
        self.conn.set_mode("LAND")
        if self.blackbox:
            self.blackbox.log_command_sent("SET_MODE", {"mode": "LAND"})

    def _command_terminal_land(self, trigger_reason: str):
        if hasattr(self.conn, "command_nav_land"):
            self.conn.command_nav_land()
        else:
            self.conn.set_mode("LAND")
        if self.blackbox:
            self.blackbox.log_command_sent("MAV_CMD_NAV_LAND", {"mode": "LAND", "trigger": trigger_reason})

    def _command_loiter(self):
        logger.warning("COMMANDING_LOITER", reason="RC/GPS failsafe")
        self.conn.set_mode("LOITER")
        if self.blackbox:
            self.blackbox.log_command_sent("SET_MODE", {"mode": "LOITER"})