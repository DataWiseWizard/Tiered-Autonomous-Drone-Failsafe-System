"""
Unit tests for Tier 2 (Environmental Validator & GPS Spoofing Heuristic)
and Tier 3 (Connected-Component Cluster Selection & Two-Stage Vision Landing).
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from state_machine import FailsafeStateMachine, TelemetrySnapshot, FlightState
from vision_landing import VisionLandingEngine

SHARED_VISION_ENGINE = VisionLandingEngine()


class MockConnection:
    def __init__(self):
        self.commands_sent = []
        self.position_targets = []
        self.nav_land_called = False

    def set_mode(self, mode: str):
        self.commands_sent.append(mode)

    def set_position_target_local_ned(self, north_m: float, east_m: float, down_m: float):
        self.position_targets.append((north_m, east_m, down_m))

    def command_nav_land(self):
        self.nav_land_called = True
        self.commands_sent.append("LAND")


def make_telemetry(
    battery_pct=100,
    voltage_mv=12600,
    gps_fix=3,
    gps_sats=10,
    ekf_horiz_var=0.2,
    lat=-35.363262,
    lon=149.165237,
    alt_m=604.0,
    flight_mode=0,
    is_armed=True,
    local_north_m=0.0,
    local_east_m=0.0,
    local_down_m=-20.0,
    timestamp=None,
):
    return TelemetrySnapshot(
        timestamp=timestamp or time.time(),
        battery_remaining_pct=battery_pct,
        battery_voltage_mv=voltage_mv,
        gps_fix_type=gps_fix,
        gps_satellites=gps_sats,
        ekf_pos_horiz_var=ekf_horiz_var,
        gps_lat=lat,
        gps_lon=lon,
        gps_alt_m=alt_m,
        flight_mode=flight_mode,
        is_armed=is_armed,
        local_north_m=local_north_m,
        local_east_m=local_east_m,
        local_down_m=local_down_m,
    )


def boot_to_nominal(sm):
    t = make_telemetry()
    sm.evaluate(t)
    assert sm.state == FlightState.NOMINAL


def test_ekf_degraded_and_recovery():
    print("\n── Test 1: EKF Degraded (>1.0) & Recovery to NOMINAL ──")
    conn = MockConnection()
    sm = FailsafeStateMachine(conn, vision_engine=SHARED_VISION_ENGINE)
    boot_to_nominal(sm)

    # EKF variance 1.8 -> TIER2_GPS_DEGRADED
    sm.evaluate(make_telemetry(ekf_horiz_var=1.8))
    assert sm.state == FlightState.TIER2_GPS_DEGRADED

    # EKF recovers to 0.4 -> NOMINAL
    sm.evaluate(make_telemetry(ekf_horiz_var=0.4))
    assert sm.state == FlightState.NOMINAL
    print("  ✓ PASSED")


def test_ekf_degraded_to_denied_escalation():
    print("\n── Test 2: EKF Degraded (>1.0) -> Critical (>5.0) Escalation ──")
    conn = MockConnection()
    sm = FailsafeStateMachine(conn, vision_engine=SHARED_VISION_ENGINE)
    boot_to_nominal(sm)

    sm.evaluate(make_telemetry(ekf_horiz_var=2.2))
    assert sm.state == FlightState.TIER2_GPS_DEGRADED

    sm.evaluate(make_telemetry(ekf_horiz_var=5.8))
    assert sm.state == FlightState.TIER2_GPS_DENIED
    print("  ✓ PASSED")


def test_haversine_gps_spoof_detection():
    print("\n── Test 3: Haversine GPS Position-Jump Spoofing Heuristic (>5m) ──")
    conn = MockConnection()
    sm = FailsafeStateMachine(conn, vision_engine=SHARED_VISION_ENGINE)
    t0 = time.time()
    sm.evaluate(make_telemetry(lat=-35.363262, lon=149.165237, timestamp=t0))
    assert sm.state == FlightState.NOMINAL

    # Inject a 0.0001 deg latitude spoof jump (~11.12 meters in 0.1s) while fix=3 and sats=12
    sm.evaluate(make_telemetry(lat=-35.363162, lon=149.165237, gps_sats=12, timestamp=t0 + 0.1))
    assert sm.spoof_detected is True
    assert sm.state == FlightState.TIER2_GPS_DENIED
    print(f"  ✓ PASSED (Trigger: {sm.spoof_reason})")


def test_connected_component_cluster_properties():
    print("\n── Test 4: Connected-Component Cluster Minimum Area & Confidence Gate ──")
    cluster = SHARED_VISION_ENGINE.find_safest_zone(30, 25)
    assert cluster is not None
    assert cluster["area_tiles"] >= SHARED_VISION_ENGINE.MIN_CLUSTER_TILES
    assert cluster["area_m2"] >= 75.0
    assert cluster["prob"] > 0.80
    assert cluster["mean_conf"] >= SHARED_VISION_ENGINE.MIN_CONFIDENCE_SCORE
    print(f"  ✓ PASSED (Cluster ID={cluster['cluster_id']}, Area={cluster['area_m2']}m^2, Prob={cluster['prob']:.3f})")


def test_full_tier2_to_tier3_two_stage_cascade():
    print("\n── Test 5: Full Tier 2 -> Tier 3 Two-Stage Landing Cascade ──")
    conn = MockConnection()
    sm = FailsafeStateMachine(conn, vision_engine=SHARED_VISION_ENGINE)
    boot_to_nominal(sm)

    # 1. GPS lost -> TIER2_GPS_DENIED
    t_denied = make_telemetry(gps_fix=0, gps_sats=0)
    sm.evaluate(t_denied)
    assert sm.state == FlightState.TIER2_GPS_DENIED

    # 2. Next tick -> TIER3_VISION_HOVER
    sm.evaluate(t_denied)
    assert sm.state == FlightState.TIER3_VISION_HOVER

    # 3. Evaluate in hover: selects cluster and commands GUIDED mode
    sm.evaluate(t_denied)
    assert "GUIDED" in conn.commands_sent

    # 4. Simulate mode feedback (GUIDED=4) after 0.6s settle -> sends Stage 1 position target (-10m)
    sm._guidance_start_time = time.time() - 0.6
    t_guided = make_telemetry(gps_fix=0, gps_sats=0, flight_mode=4, local_north_m=0.0, local_east_m=0.0)
    sm.evaluate(t_guided)
    assert sm.state == FlightState.TIER3_DESCENT
    assert len(conn.position_targets) == 1
    target_n, target_e, target_d = conn.position_targets[0]
    assert target_d == -10.0

    # 5. Simulate drone arriving within 1.5m of (target_n, target_e) -> triggers Stage 2 MAV_CMD_NAV_LAND
    t_arrived = make_telemetry(
        gps_fix=0, gps_sats=0, flight_mode=4,
        local_north_m=target_n + 0.5, local_east_m=target_e - 0.5, local_down_m=-10.0
    )
    sm.evaluate(t_arrived)
    assert conn.nav_land_called is True
    assert "LAND" in conn.commands_sent

    # 6. Simulate touchdown (local_down_m near 0 and disarmed) -> LANDED
    t_touchdown = make_telemetry(
        gps_fix=0, gps_sats=0, flight_mode=9, is_armed=False,
        local_north_m=target_n, local_east_m=target_e, local_down_m=-0.1, alt_m=584.2
    )
    sm.evaluate(t_touchdown)
    assert sm.state == FlightState.LANDED
    print("  ✓ PASSED")


def test_battery_critical_overrides_tier3():
    print("\n── Test 6: Battery Critical (8%) Overrides Active Tier 3 Vision Hover ──")
    conn = MockConnection()
    sm = FailsafeStateMachine(conn, vision_engine=SHARED_VISION_ENGINE)
    boot_to_nominal(sm)

    sm.evaluate(make_telemetry(gps_fix=0, gps_sats=0))
    sm.evaluate(make_telemetry(gps_fix=0, gps_sats=0))
    assert sm.state == FlightState.TIER3_VISION_HOVER

    # Drop battery to 7% / 10.1V during vision hover -> immediate global override to TIER1_BATTERY_LAND
    sm.evaluate(make_telemetry(gps_fix=0, gps_sats=0, battery_pct=7, voltage_mv=10100))
    assert sm.state == FlightState.TIER1_BATTERY_LAND
    print("  ✓ PASSED")


if __name__ == "__main__":
    print("═" * 60)
    print("  TIER 2 & TIER 3 — SPOOFING, CLUSTERING & CASCADE TESTS")
    print("═" * 60)

    test_ekf_degraded_and_recovery()
    test_ekf_degraded_to_denied_escalation()
    test_haversine_gps_spoof_detection()
    test_connected_component_cluster_properties()
    test_full_tier2_to_tier3_two_stage_cascade()
    test_battery_critical_overrides_tier3()

    print("\n" + "═" * 60)
    print("  ALL 6 TIER 2/3 TESTS PASSED ✓")
    print("═" * 60)