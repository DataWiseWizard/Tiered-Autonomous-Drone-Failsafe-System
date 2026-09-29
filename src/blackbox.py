"""
blackbox.py
Flight Data Recorder (Black Box) for Tiered Failsafe System.
Writes structured JSON logs to disk for post-flight analysis.
"""
import json
import time
import os
from typing import Dict, Any
from pathlib import Path

class BlackBoxLogger:
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Create a unique log file for this flight session
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"flight_{timestamp}.jsonl" # JSON Lines format
        
        self.session_start = time.time()
        self.event_count = 0
        
        # Write session header
        self._write_event({
            "event": "SESSION_START",
            "timestamp": self.session_start,
            "log_file": str(self.log_file)
        })
        
    def _write_event(self, data: Dict[str, Any]):
        """Append a single JSON event to the log file."""
        try:
            with open(self.log_file, "a") as f:
                # Add relative time since session start for easy timeline analysis
                data["rel_time_s"] = round(time.time() - self.session_start, 3)
                f.write(json.dumps(data) + "\n")
                self.event_count += 1
        except Exception as e:
            print(f"[BLACKBOX ERROR] Failed to write log: {e}")

    def log_state_transition(self, from_state: str, to_state: str, reason: str):
        """Log a state machine transition."""
        self._write_event({
            "event": "STATE_TRANSITION",
            "from_state": from_state,
            "to_state": to_state,
            "reason": reason
        })

    def log_telemetry_snapshot(self, snapshot_dict: Dict[str, Any], state: str):
        """Log a telemetry snapshot (call this at a lower frequency, e.g., 1Hz)."""
        self._write_event({
            "event": "TELEMETRY_SNAPSHOT",
            "current_state": state,
            "data": snapshot_dict
        })

    def log_command_sent(self, command: str, params: Dict[str, Any] = None):
        """Log a command sent to the flight controller."""
        self._write_event({
            "event": "COMMAND_SENT",
            "command": command,
            "params": params or {}
        })

    def close(self):
        """Finalize the log file."""
        self._write_event({
            "event": "SESSION_END",
            "total_events": self.event_count,
            "duration_s": round(time.time() - self.session_start, 2)
        })
        print(f"\n[BLACKBOX] Flight log saved to: {self.log_file}")