"""
state_machine.py
Tiered Failsafe State Machine — Phase 3 (Full Autonomy)
"""
import time
from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional, List

import structlog
from vision_landing import VisionLandingEngine

logger = structlog.get_logger()

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
    
    def battery_voltage(self) -> float:
        return self.battery_voltage_mv / 1000.0

@dataclass
class TransitionRecord:
    timestamp: float
    from_state: FlightState
    to_state: FlightState
    reason: str
    telemetry: Optional[TelemetrySnapshot] = None

class FailsafeStateMachine:
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

    def __init__(self, connection, blackbox_logger=None):
        self.conn = connection
        self.blackbox = blackbox_logger
        self.state = FlightState.BOOT
        self.history: List[TransitionRecord] = []
        
        # Command tracking
        self._last_rc_seen_time: Optional[float] = None
        self._rtl_commanded: bool = False
        self._land_commanded: bool = False
        self._loiter_commanded: bool = False
        
        # Tier 3 Vision Tracking
        self.vision_engine = VisionLandingEngine()
        self._vision_landing_triggered: bool = False
        self._vision_target: Optional[dict] = None
        self._guided_commanded: bool = False
        self._descent_commanded: bool = False
        self._guidance_start_time: float = 0.0
        self._descent_start_time = None
        
        # Simulated starting grid position (Center of your 60x50 heatmap)
        self.current_grid_x = 30
        self.current_grid_y = 25 
        
        self._transition(FlightState.BOOT, "State machine initialized")
    
    def evaluate(self, telemetry: TelemetrySnapshot):
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
    
    def get_state(self) -> FlightState: return self.state
    def get_history(self) -> List[TransitionRecord]: return self.history.copy()
    
    def _assess_battery(self, t: TelemetrySnapshot) -> str:
        voltage = t.battery_voltage()
        pct = t.battery_remaining_pct
        if voltage <= self.BATTERY_LAND_VOLTAGE or pct <= self.BATTERY_LAND_PCT: return "LAND"
        if voltage <= self.BATTERY_RTL_VOLTAGE or pct <= self.BATTERY_RTL_PCT: return "RTL"
        if voltage <= self.BATTERY_WARNING_VOLTAGE or pct <= self.BATTERY_WARNING_PCT: return "WARNING"
        return "OK"

    def _assess_environment(self, t: TelemetrySnapshot) -> str:
        if t.gps_fix_type < self.MIN_GPS_FIX_TYPE or t.gps_satellites < self.MIN_SATELLITES:
            return "GPS_LOST"
        if t.ekf_pos_horiz_var > self.EKF_CRITICAL_THRESHOLD:
            return "EKF_CRITICAL"
        if t.ekf_pos_horiz_var > self.EKF_DEGRADED_THRESHOLD:
            return "EKF_DEGRADED"
        return "HEALTHY"
    
    def _handle_boot(self, t: TelemetrySnapshot):
        if t.battery_remaining_pct > 0 and t.gps_fix_type >= 3:
            self._transition(FlightState.NOMINAL, f"Boot complete. Battery={t.battery_remaining_pct}%, GPS fix={t.gps_fix_type}")
    
    def _handle_nominal(self, t: TelemetrySnapshot):
        batt_status = self._assess_battery(t)
        env_status = self._assess_environment(t)
        
        if batt_status == "LAND":
            self._transition(FlightState.TIER1_BATTERY_LAND, f"CRITICAL: Battery {t.battery_voltage():.1f}V. EMERGENCY LAND.")
            return
        if env_status in ("GPS_LOST", "EKF_CRITICAL"):
            self._transition(FlightState.TIER2_GPS_DENIED, f"Environment Critical: GPS={t.gps_fix_type}, Sats={t.gps_satellites}, EKF_Var={t.ekf_pos_horiz_var:.2f}")
            return
        if batt_status == "RTL":
            self._transition(FlightState.TIER1_BATTERY_RTL, f"Battery {t.battery_voltage():.1f}V. Triggering RTL.")
            return
        if env_status == "EKF_DEGRADED":
            self._transition(FlightState.TIER2_GPS_DEGRADED, f"EKF Degraded: Var={t.ekf_pos_horiz_var:.2f}")
            return
        if batt_status == "WARNING":
            self._transition(FlightState.TIER1_BATTERY_WARNING, f"Battery Warning: {t.battery_voltage():.1f}V")
            return
        
        if not t.rc_received:
            if self._last_rc_seen_time is None: self._last_rc_seen_time = time.time()
            if time.time() - self._last_rc_seen_time >= self.RC_LOST_TIMEOUT_S:
                self._transition(FlightState.TIER1_RC_LOST, f"RC lost for >{self.RC_LOST_TIMEOUT_S}s")
                return
        else:
            self._last_rc_seen_time = time.time()

    def _handle_battery_warning(self, t: TelemetrySnapshot):
        if self._assess_battery(t) in ("RTL", "LAND"):
            self._transition(FlightState.TIER1_BATTERY_RTL if self._assess_battery(t) == "RTL" else FlightState.TIER1_BATTERY_LAND, "Battery dropped further.")
        elif self._assess_battery(t) == "OK":
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
        env_status = self._assess_environment(t)
        if env_status in ("GPS_LOST", "EKF_CRITICAL"):
            self._transition(FlightState.TIER2_GPS_DENIED, "EKF degraded to critical.")
        elif env_status == "HEALTHY":
            self._transition(FlightState.NOMINAL, "EKF recovered.")

    def _handle_gps_denied(self, t: TelemetrySnapshot):
        env_status = self._assess_environment(t)
        if env_status == "HEALTHY":
            self._transition(FlightState.NOMINAL, "GPS/EKF recovered.")
            return
            
        if not self._vision_landing_triggered:
            self._vision_landing_triggered = True
            self._transition(FlightState.TIER3_VISION_HOVER, "GPS Denied. Initiating Tier 3 Vision Landing.")

    def _handle_tier3_vision_hover(self, t: TelemetrySnapshot):
        """Calculate safe zone, switch to GUIDED, and fly to it."""
        if not self._vision_target:
            self._vision_target = self.vision_engine.find_safest_zone(self.current_grid_x, self.current_grid_y)
            
            if self._vision_target is None:
                logger.critical("ABORT_LANDING", reason="No safe zone. Loitering.")
                self.conn.set_mode("LOITER")
                return
                
            self.conn.set_mode("GUIDED")
            self._guided_commanded = True
            self._guidance_start_time = time.time()
            
        elif self._guided_commanded and not self._descent_commanded:
            # Wait 2 seconds for GUIDED mode to engage before sending coordinates
            if time.time() - self._guidance_start_time > 2.0:
                north_m, east_m = self.vision_engine.grid_to_local_meters(
                    self._vision_target['x'], self._vision_target['y'], self.current_grid_x, self.current_grid_y
                )
                logger.info("FLYING_TO_TARGET", north=north_m, east=east_m, tier=self._vision_target['tier'])
                # Command drone to fly to offset at 10m altitude (Down is negative in NED)
                self.conn.set_position_target_local_ned(north_m, east_m, -10.0)
                self._descent_commanded = True
                self._transition(FlightState.TIER3_DESCENT, f"En route to {self._vision_target['tier']} zone.")

    def _handle_tier3_descent(self, t: TelemetrySnapshot):
        """Monitor descent progress. Transition to LANDED when on ground."""
        
        # Initialize descent timer if this is the first run in this state
        if not hasattr(self, '_descent_start_time') or self._descent_start_time is None:
            self._descent_start_time = time.time()
            logger.info("DESCENT_STARTED", timestamp=self._descent_start_time)
        
        elapsed = time.time() - self._descent_start_time
        
        # Check altitude IF it is available (GPS might be dead, so alt might be None)
        altitude = t.gps_alt_m if t.gps_alt_m is not None else 999.0
        
        # Condition 1: Altitude confirms touchdown (if alt data is valid)
        # Note: In SITL, home altitude is ~584m. Touchdown is ~584m. 
        # We check if we are within 2 meters of home altitude (approx 584m)
        # OR if absolute altitude is very low (depending on SITL frame)
        if altitude < 586.0 and not t.is_armed:
            self._transition(FlightState.LANDED,
                            f"Touchdown confirmed via telemetry. Alt={altitude:.1f}m")
            return
            
        # Condition 2: Timeout fallback (If sensors are dead, rely on time)
        # A descent from 20-40m takes about 15-20 seconds at 1-2 m/s.
        # If 30 seconds pass, we assume the landing sequence is complete.
        if elapsed > 30.0:
            self._transition(FlightState.LANDED,
                            f"Descent timeout reached ({elapsed:.1f}s). Assuming landed.")
            return
    
    def _transition(self, new_state: FlightState, reason: str, telemetry: Optional[TelemetrySnapshot] = None):
        old_state = self.state
        record = TransitionRecord(timestamp=time.time(), from_state=old_state, to_state=new_state, reason=reason, telemetry=telemetry)
        self.history.append(record)
        self.state = new_state
        
        # Reset command flags on recovery
        if new_state == FlightState.NOMINAL:
            self._rtl_commanded = False
            self._land_commanded = False
            self._loiter_commanded = False
            self._vision_landing_triggered = False
            self._guided_commanded = False
            self._descent_commanded = False
            self._vision_target = None

        if self.blackbox:
            self.blackbox.log_state_transition(old_state.name, new_state.name, reason)

        logger.info("STATE_TRANSITION", from_state=old_state.name, to_state=new_state.name, reason=reason)
    
    def _command_rtl(self):
        logger.warning("COMMANDING_RTL", reason="Battery low failsafe")
        self.conn.set_mode("RTL")
    
    def _command_land(self):
        logger.critical("COMMANDING_EMERGENCY_LAND", reason="Battery critical failsafe")
        self.conn.set_mode("LAND")
    
    def _command_loiter(self):
        logger.warning("COMMANDING_LOITER", reason="RC/GPS failsafe")
        self.conn.set_mode("LOITER")