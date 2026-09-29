import time
from typing import Callable, Dict
from enum import Enum
from pymavlink import mavutil
import structlog

logger = structlog.get_logger()

class ConnectionStatus(Enum):
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    LINK_LOST = "link_lost"

class MAVLinkConnection:
    """Production-grade MAVLink connection manager."""
    
    HEARTBEAT_TIMEOUT_S = 5.0
    CONNECT_RETRY_LIMIT = 5
    CONNECT_RETRY_DELAY_S = 2.0
    
    def __init__(self, connection_string: str = "udp:127.0.0.1:14550", baudrate: int = 57600):
        self.connection_string = connection_string
        self.baudrate = baudrate
        self.status = ConnectionStatus.DISCONNECTED
        self._mav = None
        self.target_system = 1
        self.target_component = 1
        self.last_heartbeat_time = 0.0
        self._msg_handlers: Dict[str, list[Callable]] = {}

    def connect(self) -> bool:
        self.status = ConnectionStatus.CONNECTING
        for attempt in range(1, self.CONNECT_RETRY_LIMIT + 1):
            try:
                logger.info("Connecting to MAVLink", connection=self.connection_string, attempt=attempt)
                self._mav = mavutil.mavlink_connection(
                    self.connection_string,
                    baud=self.baudrate,
                    source_system=255,      # GCS System ID
                    source_component=190    # MAV_COMP_ID_MISSIONPLANNER
                )
                
                logger.info("Waiting for heartbeat...")
                heartbeat = self._mav.wait_heartbeat(timeout=10)
                if heartbeat is None:
                    raise ConnectionError("No heartbeat received in 10s")
                
                self.target_system = heartbeat.get_srcSystem()
                self.target_component = heartbeat.get_srcComponent()
                self.status = ConnectionStatus.CONNECTED
                self.last_heartbeat_time = time.time()
                
                logger.info("MAVLink connected", 
                          target_system=self.target_system, 
                          vehicle_type=heartbeat.type,
                          autopilot=heartbeat.autopilot)
                return True
                
            except Exception as e:
                logger.warning("Connection attempt failed", attempt=attempt, error=str(e))
                if attempt < self.CONNECT_RETRY_LIMIT:
                    time.sleep(self.CONNECT_RETRY_DELAY_S)
        
        self.status = ConnectionStatus.DISCONNECTED
        logger.error("All connection attempts failed")
        return False

    def is_alive(self) -> bool:
        if self.status != ConnectionStatus.CONNECTED:
            return False
        if time.time() - self.last_heartbeat_time > self.HEARTBEAT_TIMEOUT_S:
            self.status = ConnectionStatus.LINK_LOST
            logger.warning("Link lost: heartbeat timeout")
            return False
        return True

    def register_handler(self, msg_type: str, callback: Callable):
        if msg_type not in self._msg_handlers:
            self._msg_handlers[msg_type] = []
        self._msg_handlers[msg_type].append(callback)

    def receive_msgs_batch(self, timeout: float = 0.1):
        """Non-blocking batch receive for high-frequency loops."""
        if not self._mav or self.status != ConnectionStatus.CONNECTED:
            return
        
        deadline = time.time() + timeout
        while time.time() < deadline:
            msg = self._mav.recv_match(blocking=False)
            if msg is None:
                time.sleep(0.01)
                continue
            
            msg_type = msg.get_type()
            if msg_type == "HEARTBEAT":
                self.last_heartbeat_time = time.time()
            
            if msg_type in self._msg_handlers:
                for cb in self._msg_handlers[msg_type]:
                    try:
                        cb(msg)
                    except Exception as e:
                        logger.error("Handler error", msg_type=msg_type, error=str(e))

    def set_mode(self, mode: str) -> bool:
        """Set flight mode (e.g., 'RTL', 'LOITER', 'GUIDED', 'LAND')."""
        if not self._mav: return False
        
        COPTER_MODES = {
            "STABILIZE": 0, "ACRO": 1, "ALT_HOLD": 2, "AUTO": 3, "GUIDED": 4,
            "LOITER": 5, "RTL": 6, "CIRCLE": 7, "LAND": 9, "DRIFT": 11,
            "SPORT": 13, "FLIP": 14, "AUTOTUNE": 15, "POS_HOLD": 16, "BRAKE": 17
        }
        mode_id = COPTER_MODES.get(mode.upper(), 0)
        
        self._mav.mav.set_mode_send(
            self.target_system,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            mode_id
        )
        logger.info("Mode command sent", mode=mode)
        return True


    def set_position_target_local_ned(self, north_m: float, east_m: float, down_m: float):
        """Command drone to fly to a local NED offset in GUIDED mode."""
        if not self._mav: return
        
        # Bitmask to ignore velocity and acceleration, only use position
        # 0b0000111111000111 = 4039 in decimal
        type_mask = 0b0000111111000111 
        
        self._mav.mav.set_position_target_local_ned_send(
            0,  # time_boot_ms
            self.target_system,
            self.target_component,
            mavutil.mavlink.MAV_FRAME_BODY_NED,
            type_mask,
            east_m,   # x -> East in NED, but MAV_FRAME_BODY_NED uses X=Forward(North), Y=Right(East)
            north_m,  # y -> North
            down_m,   # z -> Down
            0, 0, 0,  # vx, vy, vz
            0, 0, 0,  # afx, afy, afz
            0, 0      # yaw, yaw_rate
        )
        logger.info("POSITION_TARGET_SENT", north=north_m, east=east_m, down=down_m)

    def disconnect(self):
        if self._mav:
            self._mav.close()
        self.status = ConnectionStatus.DISCONNECTED
        logger.info("Disconnected")