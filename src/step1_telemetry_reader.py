import time
import structlog
from mavlink_connection import MAVLinkConnection

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer()
    ]
)
logger = structlog.get_logger()

class TelemetryReader:
    """Reads and displays critical Tier 1 & Tier 2 telemetry."""
    
    def __init__(self, connection_string: str = "udp:127.0.0.1:14550"):
        self.conn = MAVLinkConnection(connection_string)
        self.latest_data = {
            'heartbeat': None, 'sys_status': None, 
            'gps_raw_int': None, 'ekf_status': None
        }
    
    def start(self):
        if not self.conn.connect():
            logger.error("Failed to connect. Is SITL running?")
            return
        
        # Register callbacks for specific MAVLink messages
        self.conn.register_handler("HEARTBEAT", self._on_heartbeat)
        self.conn.register_handler("SYS_STATUS", self._on_sys_status)
        self.conn.register_handler("GPS_RAW_INT", self._on_gps_raw_int)
        self.conn.register_handler("EKF_STATUS_REPORT", self._on_ekf_status)
        
        logger.info("Telemetry reader active. Press Ctrl+C to stop.")
        
        try:
            while self.conn.is_alive():
                self.conn.receive_msgs_batch(timeout=0.1)
                self._print_telemetry_summary()
                time.sleep(1.0) # Print at 1Hz for readability
        except KeyboardInterrupt:
            logger.info("Shutting down...")
        finally:
            self.conn.disconnect()
    
    # --- Message Handlers ---
    def _on_heartbeat(self, msg):
        self.latest_data['heartbeat'] = {
            'type': msg.type,
            'is_armed': bool(msg.base_mode & 128), # Bitmask check for armed state
        }
    
    def _on_sys_status(self, msg):
        self.latest_data['sys_status'] = {
            'battery_voltage_mv': msg.voltage_battery,
            'battery_remaining_pct': msg.battery_remaining
        }
    
    def _on_gps_raw_int(self, msg):
        self.latest_data['gps_raw_int'] = {
            'fix_type': msg.fix_type,
            'lat_deg': msg.lat / 1e7,
            'lon_deg': msg.lon / 1e7,
            'alt_m': msg.alt / 1000.0,
            'satellites_visible': msg.satellites_visible,
        }
    
    def _on_ekf_status(self, msg):
        self.latest_data['ekf_status'] = {
            'velocity_variance': msg.velocity_variance,
            'pos_horiz_variance': msg.pos_horiz_variance,
            'pos_vert_variance': msg.pos_vert_variance,
            'compass_variance': msg.compass_variance
        }
    
    def _print_telemetry_summary(self):
        hb = self.latest_data.get('heartbeat')
        sys_st = self.latest_data.get('sys_status')
        gps = self.latest_data.get('gps_raw_int')
        ekf = self.latest_data.get('ekf_status')
        
        print("\n" + "="*70)
        print(f"{'TIERED FAILSAFE TELEMETRY (STEP 1 VERIFICATION)':^70}")
        print("="*70)
        
        if hb:
            armed_str = "ARMED" if hb['is_armed'] else "DISARMED"
            print(f"  Vehicle: Type={hb['type']} | Status={armed_str}")
        else:
            print("  Vehicle: NO HEARTBEAT")
        
        if sys_st:
            batt_v = sys_st['battery_voltage_mv'] / 1000.0
            batt_pct = sys_st['battery_remaining_pct']
            status = "OK" if batt_pct > 15 else "LOW" if batt_pct > 8 else "CRITICAL"
            print(f"  Battery: {batt_v:.1f}V | {batt_pct}% | [{status}]")
        else:
            print("  Battery: NO DATA")
        
        if gps:
            fix_names = {0: "NO_GPS", 1: "NO_FIX", 2: "2D_FIX", 3: "3D_FIX", 4: "DGPS", 5: "RTK_FLOAT", 6: "RTK_FIXED"}
            fix_str = fix_names.get(gps['fix_type'], f"UNKNOWN")
            print(f"  GPS: {fix_str} | Sats={gps['satellites_visible']} | "
                  f"Pos=({gps['lat_deg']:.6f}, {gps['lon_deg']:.6f}) | Alt={gps['alt_m']:.1f}m")
        else:
            print("  GPS: NO DATA")
        
        if ekf:
            pos_var = ekf['pos_horiz_variance']
            if pos_var < 0.5: health = "EXCELLENT"
            elif pos_var < 1.0: health = "GOOD"
            elif pos_var < 5.0: health = "DEGRADED ⚠️"
            else: health = "DENIED 🚨"
            print(f"  EKF: Vel={ekf['velocity_variance']:.2f} | PosH={pos_var:.2f} | Health=[{health}]")
        else:
            print("  EKF: NO DATA")
            
        print("="*70)

if __name__ == "__main__":
    reader = TelemetryReader("udp:127.0.0.1:14550")
    reader.start()