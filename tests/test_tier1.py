"""
Unit tests for Tier 1 state machine logic.
Tests BOTH percentage-based and voltage-based battery failsafes.
"""
import sys
from pathlib import Path

# Automatically add the 'src/' directory to Python's import path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from state_machine import FailsafeStateMachine, TelemetrySnapshot, FlightState


class MockConnection:
    def __init__(self):
        self.commands_sent = []
    
    def set_mode(self, mode: str):
        self.commands_sent.append(mode)
        print(f"  [MOCK] Command sent: SET_MODE → {mode}")


def make_telemetry(battery_pct=100, voltage_mv=12600, gps_fix=3, gps_sats=10, rc=True):
    """Helper to create telemetry snapshots with healthy default GPS (fix=3, sats=10)."""
    return TelemetrySnapshot(
        battery_remaining_pct=battery_pct,
        battery_voltage_mv=voltage_mv,
        gps_fix_type=gps_fix,
        gps_satellites=gps_sats,
        rc_received=rc
    )


def boot_to_nominal(sm, conn):
    """Helper: boot the state machine to NOMINAL."""
    t = make_telemetry()
    sm.evaluate(t)
    assert sm.state == FlightState.NOMINAL, f"Expected NOMINAL, got {sm.state}"


# ── Percentage-based tests ──

def test_battery_warning_by_pct():
    print("\n── Test: Battery Warning (by percentage) ──")
    conn = MockConnection()
    sm = FailsafeStateMachine(conn)
    boot_to_nominal(sm, conn)
    
    t = make_telemetry(battery_pct=18)  # Below 20% warning
    sm.evaluate(t)
    assert sm.state == FlightState.TIER1_BATTERY_WARNING
    assert len(conn.commands_sent) == 0
    print("  ✓ PASSED")


def test_battery_rtl_by_pct():
    print("\n── Test: Battery RTL (by percentage) ──")
    conn = MockConnection()
    sm = FailsafeStateMachine(conn)
    boot_to_nominal(sm, conn)
    
    t = make_telemetry(battery_pct=14)  # Below 15% RTL
    sm.evaluate(t)
    assert sm.state == FlightState.TIER1_BATTERY_RTL
    
    sm.evaluate(t)  # Second evaluate sends the command
    assert "RTL" in conn.commands_sent
    print("  ✓ PASSED")


def test_battery_land_by_pct():
    print("\n── Test: Battery Emergency Land (by percentage) ──")
    conn = MockConnection()
    sm = FailsafeStateMachine(conn)
    boot_to_nominal(sm, conn)
    
    t = make_telemetry(battery_pct=7)  # Below 8% LAND
    sm.evaluate(t)
    assert sm.state == FlightState.TIER1_BATTERY_LAND
    
    sm.evaluate(t)
    assert "LAND" in conn.commands_sent
    print("  ✓ PASSED")


# ── Voltage-based tests ──

def test_battery_warning_by_voltage():
    print("\n── Test: Battery Warning (by voltage) ──")
    conn = MockConnection()
    sm = FailsafeStateMachine(conn)
    boot_to_nominal(sm, conn)
    
    # 11.2V is below WARNING threshold (11.5V) but pct is 100%
    t = make_telemetry(battery_pct=100, voltage_mv=11200)
    sm.evaluate(t)
    assert sm.state == FlightState.TIER1_BATTERY_WARNING
    print("  ✓ PASSED")


def test_battery_rtl_by_voltage():
    print("\n── Test: Battery RTL (by voltage) ──")
    conn = MockConnection()
    sm = FailsafeStateMachine(conn)
    boot_to_nominal(sm, conn)
    
    # 10.8V is below RTL threshold (11.0V) but pct is 100%
    t = make_telemetry(battery_pct=100, voltage_mv=10800)
    sm.evaluate(t)
    assert sm.state == FlightState.TIER1_BATTERY_RTL
    
    sm.evaluate(t)
    assert "RTL" in conn.commands_sent
    print("  ✓ PASSED")


def test_battery_land_by_voltage():
    print("\n── Test: Battery Emergency Land (by voltage) ──")
    conn = MockConnection()
    sm = FailsafeStateMachine(conn)
    boot_to_nominal(sm, conn)
    
    # 10.0V is below LAND threshold (10.5V) but pct is 100%
    t = make_telemetry(battery_pct=100, voltage_mv=10000)
    sm.evaluate(t)
    assert sm.state == FlightState.TIER1_BATTERY_LAND
    
    sm.evaluate(t)
    assert "LAND" in conn.commands_sent
    print("  ✓ PASSED")


# ── RC tests ──

def test_rc_lost():
    print("\n── Test: RC Lost ──")
    import time
    
    conn = MockConnection()
    sm = FailsafeStateMachine(conn)
    boot_to_nominal(sm, conn)
    
    t = make_telemetry(rc=False)
    sm._last_rc_seen_time = time.time() - 4.0
    sm.evaluate(t)
    assert sm.state == FlightState.TIER1_RC_LOST
    
    sm.evaluate(t)
    assert "LOITER" in conn.commands_sent
    print("  ✓ PASSED")


def test_rc_recovery():
    print("\n── Test: RC Recovery ──")
    import time
    
    conn = MockConnection()
    sm = FailsafeStateMachine(conn)
    boot_to_nominal(sm, conn)
    
    t = make_telemetry(rc=False)
    sm._last_rc_seen_time = time.time() - 4.0
    sm.evaluate(t)
    assert sm.state == FlightState.TIER1_RC_LOST
    
    t.rc_received = True
    sm.evaluate(t)
    assert sm.state == FlightState.NOMINAL
    print("  ✓ PASSED")


if __name__ == "__main__":
    print("═" * 50)
    print("  TIER 1 STATE MACHINE — UNIT TESTS")
    print("═" * 50)
    
    test_battery_warning_by_pct()
    test_battery_rtl_by_pct()
    test_battery_land_by_pct()
    test_battery_warning_by_voltage()
    test_battery_rtl_by_voltage()
    test_battery_land_by_voltage()
    test_rc_lost()
    test_rc_recovery()
    
    print("\n" + "═" * 50)
    print("  ALL 8 TESTS PASSED ✓")
    print("═" * 50)