"""
Sprint 2 Unit Tests:
  1. Optical-Flow + Yaw Rotation + Rangefinder GPS-denied dead-reckoning
  2. Dual-GPS horizontal divergence spoofing detection (>3.0m)
  3. EKF Normalized Innovation Squared (Chi-Squared > 6.5) spoofing detection
  4. Terminal-descent Rangefinder obstacle detection & diversion to alternate cluster
  5. Battery-aware reachability constraint filtering
"""
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sensor_fusion import SensorFusionEngine
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


def test_optical_flow_yaw_rotation_and_dead_reckoning():
    print("\n── Test 1: Optical Flow Yaw Rotation (90 deg East) & Dead-Reckoning ──")
    sf = SensorFusionEngine()
    t0 = 1000.0
    sf.reset_origin(0.0, 0.0, 10.0)

    # Drone is yawed 90 degrees East (pi/2 rad) and moving Forward in Body X at 2.0 m/s.
    # In Earth NED frame, Forward when facing East = +East velocity (vn ~ 0, ve > 0)!
    est = sf.update(
        timestamp=t0,
        gps_healthy=False,
        local_down_m=-10.0,
        rangefinder_dist_m=10.0,
        yaw_rad=math.pi / 2.0,
        flow_comp_m_x=2.0,
        flow_comp_m_y=0.0,
        flow_quality=255,
    )
    assert est.flow_active is True
    assert abs(est.vn_ms) < 1e-4, f"Expected vn ~ 0, got {est.vn_ms}"
    assert abs(est.ve_ms - 1.70) < 0.05, f"Expected ve ~ 1.70 m/s, got {est.ve_ms}"
    assert est.east_m > 0.15
    print(f"  ✓ PASSED (Yaw=90° -> Body Vx=2.0m/s mapped to NED Ve={est.ve_ms:.2f}m/s, East={est.east_m:.3f}m)")


def test_dual_gps_divergence_spoof_detection():
    print("\n── Test 2: Dual-GPS Divergence Spoofing Detection (>3.0m) ──")
    conn = MockConnection()
    sm = FailsafeStateMachine(conn, vision_engine=SHARED_VISION_ENGINE)

    t0 = time.time()
    # Boot to NOMINAL with healthy battery (100% / 12.6V) and matching GPS1/GPS2
    sm.evaluate(TelemetrySnapshot(
        timestamp=t0, battery_remaining_pct=100, battery_voltage_mv=12600,
        gps_fix_type=3, gps_satellites=10,
        gps_lat=-35.363262, gps_lon=149.165237,
        gps2_fix_type=3, gps2_lat=-35.363262, gps2_lon=149.165237,
    ))
    assert sm.state == FlightState.NOMINAL

    # GPS1 stays fixed while GPS2 diverges by ~11.1m -> Dual-GPS spoof trigger
    sm.evaluate(TelemetrySnapshot(
        timestamp=t0 + 0.1, battery_remaining_pct=100, battery_voltage_mv=12600,
        gps_fix_type=3, gps_satellites=10,
        gps_lat=-35.363262, gps_lon=149.165237,
        gps2_fix_type=3, gps2_lat=-35.363162, gps2_lon=149.165237,
    ))
    assert sm.spoof_detected is True
    assert sm.state == FlightState.TIER2_GPS_DENIED
    print(f"  ✓ PASSED (Trigger: {sm.spoof_reason})")


def test_chi2_innovation_spoof_detection():
    print("\n── Test 3: EKF Chi-Squared Innovation Consistency Spoofing Detection ──")
    conn = MockConnection()
    sm = FailsafeStateMachine(conn, vision_engine=SHARED_VISION_ENGINE)

    sm.evaluate(TelemetrySnapshot(battery_remaining_pct=100, battery_voltage_mv=12600, gps_fix_type=3, gps_satellites=10))
    assert sm.state == FlightState.NOMINAL

    sm.evaluate(TelemetrySnapshot(
        battery_remaining_pct=100, battery_voltage_mv=12600,
        gps_fix_type=3, gps_satellites=10,
        ekf_velocity_var=1.8, ekf_pos_horiz_var=1.8, ekf_pos_vert_var=1.5,
    ))
    assert sm.spoof_detected is True
    assert sm.state == FlightState.TIER2_GPS_DENIED
    print(f"  ✓ PASSED (Chi2={sm.latest_fused_state.chi2_innovation} -> {sm.spoof_reason})")


def test_terminal_rangefinder_obstacle_diversion():
    print("\n── Test 4: Rangefinder Final-Meters Obstacle Detection & Cluster Diversion ──")
    conn = MockConnection()
    sm = FailsafeStateMachine(conn, vision_engine=SHARED_VISION_ENGINE)

    # Boot -> GPS Denied -> Vision Hover -> Stage 1 Approach
    sm.evaluate(TelemetrySnapshot(battery_remaining_pct=100, battery_voltage_mv=12600, gps_fix_type=3, gps_satellites=10))
    sm.evaluate(TelemetrySnapshot(battery_remaining_pct=100, battery_voltage_mv=12600, gps_fix_type=0, gps_satellites=0))
    sm.evaluate(TelemetrySnapshot(battery_remaining_pct=100, battery_voltage_mv=12600, gps_fix_type=0, gps_satellites=0))
    sm.evaluate(TelemetrySnapshot(battery_remaining_pct=100, battery_voltage_mv=12600, gps_fix_type=0, gps_satellites=0))
    sm._guidance_start_time = time.time() - 1.0
    sm.evaluate(TelemetrySnapshot(battery_remaining_pct=100, battery_voltage_mv=12600, gps_fix_type=0, gps_satellites=0, flight_mode=4))
    assert sm.state == FlightState.TIER3_DESCENT

    first_cluster = (sm._vision_target["x"], sm._vision_target["y"])

    # Feed jagged readings that clearly cross the terminal obstacle threshold.
    # 2.5m peak-to-peak oscillation = rocky/uneven surface (detrended std ≈ 1.07 > 0.60)
    for rng in (3.0, 0.5, 2.8, 0.3):
        sm.evaluate(TelemetrySnapshot(
            battery_remaining_pct=100, battery_voltage_mv=12600,
            gps_fix_type=0, gps_satellites=0, flight_mode=4, is_armed=True,
            gps_alt_m=594.0, local_down_m=-3.5, rangefinder_dist_m=rng,
        ))

    second_cluster = (sm._vision_target["x"], sm._vision_target["y"])
    assert first_cluster != second_cluster, "Expected diversion to a new cluster!"
    assert first_cluster in sm._excluded_clusters
    assert len(conn.position_targets) == 2
    print(f"  ✓ PASSED (Diverted from primary cluster {first_cluster} -> alternate {second_cluster})")


def test_battery_aware_reachability():
    print("\n── Test 5: Battery-Aware Reachability Radius Calculation ──")
    conn = MockConnection()
    sm = FailsafeStateMachine(conn, vision_engine=SHARED_VISION_ENGINE)

    # At 10% battery (2% above 8% floor) -> 2 * 8m = 16m -> 3.2 tiles max reach
    reach_low = sm.compute_max_reachable_tiles(10.0, grid_res_m=5.0)
    # At 50% battery -> 42 * 8m = 336m -> 67.2 tiles max reach
    reach_high = sm.compute_max_reachable_tiles(50.0, grid_res_m=5.0)
    assert abs(reach_low - 3.2) < 1e-4
    assert reach_high > 60.0
    print(f"  ✓ PASSED (10% Batt -> {reach_low:.1f} tiles max reach; 50% Batt -> {reach_high:.1f} tiles)")


if __name__ == "__main__":
    print("═" * 68)
    print("  SPRINT 2: SENSOR FUSION, SPOOFING & OBSTACLE DIVERSION TESTS")
    print("═" * 68)

    test_optical_flow_yaw_rotation_and_dead_reckoning()
    test_dual_gps_divergence_spoof_detection()
    test_chi2_innovation_spoof_detection()
    test_terminal_rangefinder_obstacle_diversion()
    test_battery_aware_reachability()

    print("\n" + "═" * 68)
    print("  ALL 5 SPRINT 2 SENSOR FUSION TESTS PASSED ✓")
    print("═" * 68)