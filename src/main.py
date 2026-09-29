"""
main.py
Tiered Drone Failsafe — Main execution loop.
Phase 3: Full Tiered Autonomy (Telemetry + GPS/EKF + Vision Landing)
"""
import time
import signal
import sys
import structlog

# Ensure these imports match your actual file names
from mavlink_connection import MAVLinkConnection
from blackbox import BlackBoxLogger
from state_machine import FailsafeStateMachine, TelemetrySnapshot, FlightState

# ─────────────────────────────────────────────
# LOGGING CONFIG
# ─────────────────────────────────────────────
structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer(colors=True)
    ]
)
logger = structlog.get_logger()

# ─────────────────────────────────────────────
# TELEMETRY COLLECTOR
# ─────────────────────────────────────────────
class TelemetryCollector:
    """
    Collects raw MAVLink messages, converts them to dictionaries,
    and assembles them into TelemetrySnapshot objects.
    """
    def __init__(self, connection: MAVLinkConnection):
        self.conn = connection
        
        # Store everything as dictionaries to prevent attribute errors
        self._latest = {
            'heartbeat': None,
            'sys_status': None,
            'gps_raw_int': None,
            'ekf_status': None,
        }
        
        # Register handlers
        self.conn.register_handler("HEARTBEAT", self._on_heartbeat)
        self.conn.register_handler("SYS_STATUS", self._on_sys_status)
        self.conn.register_handler("GPS_RAW_INT", self._on_gps_raw_int)
        self.conn.register_handler("EKF_STATUS_REPORT", self._on_ekf_status)

    # --- Callbacks: Convert raw MAVLink msgs to dicts immediately ---
    def _on_heartbeat(self, msg):
        self._latest['heartbeat'] = {
            'is_armed': bool(msg.base_mode & 128),
            'custom_mode': msg.custom_mode
        }

    def _on_sys_status(self, msg):
        self._latest['sys_status'] = {
            'battery_voltage_mv': msg.voltage_battery,
            'battery_current_ma': msg.current_battery,
            'battery_remaining_pct': msg.battery_remaining
        }

    def _on_gps_raw_int(self, msg):
        self._latest['gps_raw_int'] = {
            'fix_type': msg.fix_type,
            'lat_deg': msg.lat / 1e7 if msg.lat else 0.0,
            'lon_deg': msg.lon / 1e7 if msg.lon else 0.0,
            'alt_m': msg.alt / 1000.0 if msg.alt else 0.0,
            'satellites_visible': msg.satellites_visible,
        }

    def _on_ekf_status(self, msg):
        self._latest['ekf_status'] = {
            'velocity_variance': msg.velocity_variance,
            'pos_horiz_variance': msg.pos_horiz_variance,
            'pos_vert_variance': msg.pos_vert_variance,
        }

    def get_snapshot(self) -> TelemetrySnapshot:
        """Assemble current telemetry into a snapshot safely."""
        snapshot = TelemetrySnapshot(timestamp=time.time())
        
        # All of these are now guaranteed to be dictionaries or None
        hb = self._latest.get('heartbeat')
        if hb:
            snapshot.is_armed = hb.get('is_armed', False)
            snapshot.flight_mode = hb.get('custom_mode', 0)
            snapshot.rc_received = True # SITL default
            
        ss = self._latest.get('sys_status')
        if ss:
            snapshot.battery_voltage_mv = ss.get('battery_voltage_mv', 0)
            snapshot.battery_current_ma = ss.get('battery_current_ma', 0)
            snapshot.battery_remaining_pct = ss.get('battery_remaining_pct', 100)
            
        gps = self._latest.get('gps_raw_int')
        if gps:
            snapshot.gps_fix_type = gps.get('fix_type', 0)
            snapshot.gps_lat = gps.get('lat_deg', 0.0)
            snapshot.gps_lon = gps.get('lon_deg', 0.0)
            snapshot.gps_alt_m = gps.get('alt_m', 0.0)
            snapshot.gps_satellites = gps.get('satellites_visible', 0)
            
        ekf = self._latest.get('ekf_status')
        if ekf:
            snapshot.ekf_velocity_var = ekf.get('velocity_variance', 0.0)
            snapshot.ekf_pos_horiz_var = ekf.get('pos_horiz_variance', 0.0)
            snapshot.ekf_pos_vert_var = ekf.get('pos_vert_variance', 0.0)
            
        return snapshot

# ─────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────
LOOP_RATE_HZ = 10  # Evaluate state machine at 10 Hz
LOOP_INTERVAL_S = 1.0 / LOOP_RATE_HZ

def print_status(snapshot: TelemetrySnapshot, state: FlightState):
    """Print current status to console (1 Hz)."""
    print("\n" + "═" * 70)
    print(f"  STATE: {state.name}")
    print(f"  Battery: {snapshot.battery_voltage():.1f}V | {snapshot.battery_remaining_pct}%")
    
    env_health = "HEALTHY"
    if snapshot.gps_fix_type < 3: env_health = "NO_FIX"
    elif snapshot.ekf_pos_horiz_var > 5.0: env_health = "CRITICAL 🚨"
    elif snapshot.ekf_pos_horiz_var > 1.0: env_health = "DEGRADED ⚠️"
    
    print(f"  GPS: fix={snapshot.gps_fix_type} | sats={snapshot.gps_satellites} | alt={snapshot.gps_alt_m:.1f}m")
    print(f"  ENV: {env_health} | EKF PosH Var: {snapshot.ekf_pos_horiz_var:.3f}")
    print(f"  Armed: {'YES' if snapshot.is_armed else 'NO'} | RC: {'OK' if snapshot.rc_received else 'LOST'}")
    print("═" * 70)

def main():
    logger.info("STARTING_TIERED_FAILSAFE", phase="Phase 3: Full Tiered Autonomy")
    
    # ── Connect to SITL ──
    conn = MAVLinkConnection("udp:127.0.0.1:14550")
    if not conn.connect():
        logger.error("CONNECTION_FAILED", hint="Is SITL running?")
        sys.exit(1)
    # ── Initialize Black Box ──
    blackbox = BlackBoxLogger(log_dir="logs")
    
    # ── Initialize components ──
    # Note: FailsafeStateMachine initializes VisionLandingEngine internally
    state_machine = FailsafeStateMachine(conn, blackbox_logger=blackbox)
    collector = TelemetryCollector(conn)
    
    logger.info("SYSTEM_READY",
               loop_rate_hz=LOOP_RATE_HZ,
               battery_rtl_threshold=FailsafeStateMachine.BATTERY_RTL_PCT,
               battery_land_threshold=FailsafeStateMachine.BATTERY_LAND_PCT)
    
    # ── Graceful shutdown ──
    running = True
    def signal_handler(sig, frame):
        nonlocal running
        logger.info("SHUTDOWN_SIGNAL_RECEIVED")
        running = False
    signal.signal(signal.SIGINT, signal_handler)
    
    # ── Main loop ──
    last_print_time = 0.0
    
    try:
        while running and conn.is_alive():
            loop_start = time.time()
            
            # 1. Read all pending MAVLink messages
            conn.receive_msgs_batch(timeout=0.05)
            
            # 2. Assemble telemetry snapshot
            snapshot = collector.get_snapshot()
            
            # 3. Evaluate state machine
            state_machine.evaluate(snapshot)
            
            # 4. Print status at 1 Hz (for readability)
            if time.time() - last_print_time >= 1.0:
                print_status(snapshot, state_machine.get_state())
                
                # Log telemetry to black box
                blackbox.log_telemetry_snapshot({
                    "battery_v": snapshot.battery_voltage(),
                    "battery_pct": snapshot.battery_remaining_pct,
                    "gps_fix": snapshot.gps_fix_type,
                    "gps_sats": snapshot.gps_satellites,
                    "gps_alt_m": snapshot.gps_alt_m,
                    "ekf_pos_h_var": snapshot.ekf_pos_horiz_var,
                    "is_armed": snapshot.is_armed
                }, state_machine.get_state().name)
                
                last_print_time = time.time()
                
            # 5. Maintain loop rate
            elapsed = time.time() - loop_start
            sleep_time = LOOP_INTERVAL_S - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
                
    except Exception as e:
        logger.error("UNHANDLED_EXCEPTION", error=str(e))
    finally:
        conn.disconnect()
        blackbox.close() # <--- ADD THIS
        logger.info("SYSTEM_SHUTDOWN", total_transitions=len(state_machine.get_history()))
        
        # Print transition history to console as well
        print("\n" + "═" * 70)
        print("  FLIGHT STATE HISTORY (BLACK BOX)")
        print("═" * 70)
        for record in state_machine.get_history():
            ts = time.strftime("%H:%M:%S", time.localtime(record.timestamp))
            print(f"  [{ts}] {record.from_state.name} → "
                  f"{record.to_state.name} | {record.reason}")
        print("═" * 70)

if __name__ == "__main__":
    main()