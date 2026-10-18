"""
main.py
Tiered Drone Failsafe — Main execution loop (v2.0 Sensor Fusion + Tiered Autonomy).
"""
import time
import signal
import sys
import structlog

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
    Collects raw MAVLink messages (including v2.0 Optical Flow, Rangefinder,
    Attitude, and Dual-GPS), converts them to dictionaries, and assembles
    them into TelemetrySnapshot objects.
    """
    def __init__(self, connection: MAVLinkConnection):
        self.conn = connection

        self._latest = {
            'heartbeat': None,
            'sys_status': None,
            'gps_raw_int': None,
            'gps2_raw': None,
            'ekf_status': None,
            'local_pos': None,
            'attitude': None,
            'optical_flow': None,
            'rangefinder': None,
            'last_ack': None,
        }

        # Register core + sensor fusion MAVLink handlers
        self.conn.register_handler("HEARTBEAT", self._on_heartbeat)
        self.conn.register_handler("SYS_STATUS", self._on_sys_status)
        self.conn.register_handler("GPS_RAW_INT", self._on_gps_raw_int)
        self.conn.register_handler("GPS2_RAW", self._on_gps2_raw)
        self.conn.register_handler("EKF_STATUS_REPORT", self._on_ekf_status)
        self.conn.register_handler("LOCAL_POSITION_NED", self._on_local_pos)
        self.conn.register_handler("ATTITUDE", self._on_attitude)
        self.conn.register_handler("OPTICAL_FLOW", self._on_optical_flow)
        self.conn.register_handler("RANGEFINDER", self._on_rangefinder)
        self.conn.register_handler("COMMAND_ACK", self._on_command_ack)

    def _on_heartbeat(self, msg):
        self._latest['heartbeat'] = {
            'is_armed': bool(msg.base_mode & 128),
            'custom_mode': msg.custom_mode
        }

    def _on_sys_status(self, msg):
        # MAVLink SYS_STATUS.current_battery is in centi-amps (10 mA), -1 if unsupported
        raw_current_ca = msg.current_battery
        current_ma = (raw_current_ca * 10) if raw_current_ca >= 0 else 0

        self._latest['sys_status'] = {
            'battery_voltage_mv': msg.voltage_battery,
            'battery_current_ma': current_ma,
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

    def _on_gps2_raw(self, msg):
        self._latest['gps2_raw'] = {
            'fix_type': msg.fix_type,
            'lat_deg': msg.lat / 1e7 if msg.lat else 0.0,
            'lon_deg': msg.lon / 1e7 if msg.lon else 0.0,
            'satellites_visible': msg.satellites_visible,
        }

    def _on_ekf_status(self, msg):
        self._latest['ekf_status'] = {
            'velocity_variance': msg.velocity_variance,
            'pos_horiz_variance': msg.pos_horiz_variance,
            'pos_vert_variance': msg.pos_vert_variance,
        }

    def _on_local_pos(self, msg):
        self._latest['local_pos'] = {
            'north_m': float(msg.x),
            'east_m': float(msg.y),
            'down_m': float(msg.z),
            'vn_ms': float(msg.vx),
            've_ms': float(msg.vy),
            'vd_ms': float(msg.vz),
        }

    def _on_attitude(self, msg):
        self._latest['attitude'] = {
            'roll_rad': float(msg.roll),
            'pitch_rad': float(msg.pitch),
            'yaw_rad': float(msg.yaw),
            'rollspeed_rads': float(msg.rollspeed),
            'pitchspeed_rads': float(msg.pitchspeed),
            'yawspeed_rads': float(msg.yawspeed),
        }

    def _on_optical_flow(self, msg):
        self._latest['optical_flow'] = {
            'flow_rate_x_rads': float(getattr(msg, 'flow_rate_x', 0.0)),
            'flow_rate_y_rads': float(getattr(msg, 'flow_rate_y', 0.0)),
            'flow_comp_m_x': float(getattr(msg, 'flow_comp_m_x', 0.0)),
            'flow_comp_m_y': float(getattr(msg, 'flow_comp_m_y', 0.0)),
            'quality': int(getattr(msg, 'quality', 0)),
        }

    def _on_rangefinder(self, msg):
        self._latest['rangefinder'] = {
            'distance_m': float(msg.distance),
        }

    def _on_command_ack(self, msg):
        self._latest['last_ack'] = {
            'command': int(msg.command),
            'result': int(msg.result),
        }

    def get_snapshot(self) -> TelemetrySnapshot:
        """Assemble current telemetry into a TelemetrySnapshot safely."""
        snapshot = TelemetrySnapshot(timestamp=time.time())

        hb = self._latest.get('heartbeat')
        if hb:
            snapshot.is_armed = hb.get('is_armed', False)
            snapshot.flight_mode = hb.get('custom_mode', 0)
            snapshot.rc_received = True

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

        gps2 = self._latest.get('gps2_raw')
        if gps2:
            snapshot.gps2_fix_type = gps2.get('fix_type')
            snapshot.gps2_lat = gps2.get('lat_deg')
            snapshot.gps2_lon = gps2.get('lon_deg')

        ekf = self._latest.get('ekf_status')
        if ekf:
            snapshot.ekf_velocity_var = ekf.get('velocity_variance', 0.0)
            snapshot.ekf_pos_horiz_var = ekf.get('pos_horiz_variance', 0.0)
            snapshot.ekf_pos_vert_var = ekf.get('pos_vert_variance', 0.0)

        lpos = self._latest.get('local_pos')
        if lpos:
            snapshot.local_north_m = lpos.get('north_m')
            snapshot.local_east_m = lpos.get('east_m')
            snapshot.local_down_m = lpos.get('down_m')
            snapshot.local_vn_ms = lpos.get('vn_ms')
            snapshot.local_ve_ms = lpos.get('ve_ms')
            snapshot.local_vd_ms = lpos.get('vd_ms')

        att = self._latest.get('attitude')
        if att:
            snapshot.yaw_rad = att.get('yaw_rad', 0.0)
            snapshot.pitch_rate_rads = att.get('pitchspeed_rads', 0.0)
            snapshot.roll_rate_rads = att.get('rollspeed_rads', 0.0)

        flow = self._latest.get('optical_flow')
        if flow:
            snapshot.flow_rate_x_rads = flow.get('flow_rate_x_rads')
            snapshot.flow_rate_y_rads = flow.get('flow_rate_y_rads')
            snapshot.flow_comp_m_x = flow.get('flow_comp_m_x')
            snapshot.flow_comp_m_y = flow.get('flow_comp_m_y')
            snapshot.flow_quality = flow.get('quality', 0)

        rng = self._latest.get('rangefinder')
        if rng:
            snapshot.rangefinder_dist_m = rng.get('distance_m')

        ack = self._latest.get('last_ack')
        if ack:
            snapshot.last_ack_command = ack.get('command')
            snapshot.last_ack_result = ack.get('result')

        return snapshot


# ─────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────
LOOP_RATE_HZ = 10  # Evaluate state machine at 10 Hz
LOOP_INTERVAL_S = 1.0 / LOOP_RATE_HZ


def print_status(snapshot: TelemetrySnapshot, state: FlightState, sm: FailsafeStateMachine):
    """Print current status (including fused state estimate) to console at 1 Hz."""
    fused = sm.latest_fused_state
    print("\n" + "═" * 74)
    print(f"  STATE: {state.name}")
    print(
        f"  Battery: {snapshot.battery_voltage():.1f}V | "
        f"{snapshot.battery_current_amps():.1f}A | {snapshot.battery_remaining_pct}%"
    )

    env_health = "HEALTHY"
    if sm.spoof_detected:
        env_health = f"SPOOFED 🚨 ({sm.spoof_reason})"
    elif snapshot.gps_fix_type < 3:
        env_health = "NO_FIX"
    elif snapshot.ekf_pos_horiz_var > 5.0:
        env_health = "CRITICAL 🚨"
    elif snapshot.ekf_pos_horiz_var > 1.0:
        env_health = "DEGRADED ⚠️"

    print(
        f"  GPS: fix={snapshot.gps_fix_type} | sats={snapshot.gps_satellites} | "
        f"alt={snapshot.gps_alt_m:.1f}m"
    )
    print(
        f"  ENV: {env_health} | EKF PosH Var: {snapshot.ekf_pos_horiz_var:.3f} | "
        f"Chi2: {fused.chi2_innovation:.2f}"
    )
    print(
        f"  FUSED NED: N={fused.north_m:.1f}m, E={fused.east_m:.1f}m, "
        f"AGL={fused.alt_agl_m:.1f}m (±{fused.position_uncertainty_m:.2f}m) | "
        f"Flow={'ON' if fused.flow_active else 'OFF'} | Rng={'ON' if fused.rangefinder_active else 'OFF'}"
    )
    print(f"  Armed: {'YES' if snapshot.is_armed else 'NO'} | RC: {'OK' if snapshot.rc_received else 'LOST'}")
    print("═" * 74)


def main():
    logger.info("STARTING_TIERED_FAILSAFE", phase="v2.0: Sensor Fusion & Vision Autonomy")

    conn = MAVLinkConnection("udp:127.0.0.1:14550")
    if not conn.connect():
        logger.error("CONNECTION_FAILED", hint="Is SITL running?")
        sys.exit(1)

    blackbox = BlackBoxLogger(log_dir="logs")
    state_machine = FailsafeStateMachine(conn, blackbox_logger=blackbox)
    collector = TelemetryCollector(conn)

    logger.info(
        "SYSTEM_READY",
        loop_rate_hz=LOOP_RATE_HZ,
        battery_rtl_threshold=state_machine.BATTERY_RTL_PCT,
        battery_land_threshold=state_machine.BATTERY_LAND_PCT,
    )

    running = True

    def signal_handler(sig, frame):
        nonlocal running
        logger.info("SHUTDOWN_SIGNAL_RECEIVED")
        running = False

    signal.signal(signal.SIGINT, signal_handler)

    last_print_time = 0.0

    try:
        while running and conn.is_alive():
            loop_start = time.time()

            conn.receive_msgs_batch(timeout=0.05)
            snapshot = collector.get_snapshot()
            state_machine.evaluate(snapshot)

            if time.time() - last_print_time >= 1.0:
                print_status(snapshot, state_machine.get_state(), state_machine)
                fused = state_machine.latest_fused_state

                blackbox.log_telemetry_snapshot({
                    "battery_v": snapshot.battery_voltage(),
                    "battery_a": snapshot.battery_current_amps(),
                    "battery_pct": snapshot.battery_remaining_pct,
                    "gps_fix": snapshot.gps_fix_type,
                    "gps_sats": snapshot.gps_satellites,
                    "gps_alt_m": snapshot.gps_alt_m,
                    "ekf_pos_h_var": snapshot.ekf_pos_horiz_var,
                    "chi2_innovation": fused.chi2_innovation,
                    "fused_north_m": round(fused.north_m, 2),
                    "fused_east_m": round(fused.east_m, 2),
                    "fused_agl_m": round(fused.alt_agl_m, 2),
                    "flow_active": fused.flow_active,
                    "rangefinder_active": fused.rangefinder_active,
                    "is_armed": snapshot.is_armed,
                }, state_machine.get_state().name)

                last_print_time = time.time()

            elapsed = time.time() - loop_start
            sleep_time = LOOP_INTERVAL_S - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except Exception as e:
        logger.error("UNHANDLED_EXCEPTION", error=str(e))
    finally:
        conn.disconnect()
        blackbox.close()
        logger.info("SYSTEM_SHUTDOWN", total_transitions=len(state_machine.get_history()))

        print("\n" + "═" * 70)
        print("  FLIGHT STATE HISTORY (BLACK BOX)")
        print("═" * 70)
        for record in state_machine.get_history():
            ts = time.strftime("%H:%M:%S", time.localtime(record.timestamp))
            print(f"  [{ts}] {record.from_state.name} → {record.to_state.name} | {record.reason}")
        print("═" * 70)


if __name__ == "__main__":
    main()