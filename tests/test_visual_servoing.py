"""
tests/test_visual_servoing.py
Sprint 3 Unit Tests:
  1. Pinhole camera GSD and pixel-to-NED coordinate projection math
  2. Thread-isolated PerceptionWorker non-blocking queue & ONNX inference latency
  3. Closed-loop visual servoing target refinement during TIER3_DESCENT
  4. Dynamic visual obstacle detection on landing pad -> automatic cluster diversion
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from synthetic_camera import SyntheticDownwardCamera
from perception_worker import PerceptionWorker
from state_machine import FailsafeStateMachine, TelemetrySnapshot, FlightState
from vision_landing import VisionLandingEngine

SHARED_VISION_ENGINE = VisionLandingEngine()
SHARED_CAMERA = SyntheticDownwardCamera(prob_matrix=SHARED_VISION_ENGINE.prob_matrix)


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


def test_pinhole_camera_projection_math():
    print("\n── Test 1: Pinhole Camera GSD & Pixel-to-NED Projection Math ──")
    gsd_20m = SyntheticDownwardCamera.compute_gsd_m_per_px(20.0)
    gsd_10m = SyntheticDownwardCamera.compute_gsd_m_per_px(10.0)
    assert abs(gsd_20m - 2.0 * gsd_10m) < 1e-6, "GSD must scale linearly with altitude!"

    # Pixel (u=420, v=220) -> +100px East, +100px North from center (320, 320)
    dn, de = SyntheticDownwardCamera.pixel_to_ned_offset(420.0, 220.0, alt_agl_m=20.0, yaw_rad=0.0)
    assert dn > 3.5 and de > 3.5
    print(f"  ✓ PASSED (GSD@20m={gsd_20m*100:.2f}cm/px, GSD@10m={gsd_10m*100:.2f}cm/px, Offset=({dn:.2f}m N, {de:.2f}m E))")


def test_thread_isolated_perception_worker():
    print("\n── Test 2: Thread-Isolated ONNX PerceptionWorker Queue & Latency ──")
    worker = PerceptionWorker(camera=SHARED_CAMERA, use_gpu=True)
    worker.start()
    try:
        # Submit pose over the known Preferred cluster at (N=-50m, E=15m)
        t_submit = time.perf_counter()
        submitted = worker.submit_pose(north_m=-49.0, east_m=14.0, alt_agl_m=15.0)
        submit_time_ms = (time.perf_counter() - t_submit) * 1000.0
        assert submitted is True
        assert submit_time_ms < 2.0, f"Queue submission blocked for {submit_time_ms}ms!"

        # Wait up to 1.0s for background thread to produce result
        res = None
        deadline = time.time() + 1.5
        while time.time() < deadline and res is None:
            res = worker.get_latest_result()
            time.sleep(0.01)

        assert res is not None, "Worker did not produce a VisualServoResult!"
        assert res.safe_pad_found is True
        assert res.obstacle_detected is False
        print(
            f"  ✓ PASSED (Submit={submit_time_ms:.3f}ms [non-blocking], "
            f"Worker Latency={res.inference_latency_ms:.2f}ms, Backend={res.backend_used})"
        )
    finally:
        worker.stop()


def test_closed_loop_visual_servoing_in_state_machine():
    print("\n── Test 3: Closed-Loop Visual Servoing Target Refinement in TIER3_DESCENT ──")
    conn = MockConnection()
    worker = PerceptionWorker(camera=SHARED_CAMERA, use_gpu=True)
    worker.start()

    try:
        sm = FailsafeStateMachine(
            conn, vision_engine=SHARED_VISION_ENGINE, perception_worker=worker
        )
        # Boot -> GPS Denied -> Hover -> Stage 1 Descent
        sm.evaluate(TelemetrySnapshot(battery_remaining_pct=100, battery_voltage_mv=12600, gps_fix_type=3, gps_satellites=10))
        sm.evaluate(TelemetrySnapshot(battery_remaining_pct=100, battery_voltage_mv=12600, gps_fix_type=0, gps_satellites=0))
        sm.evaluate(TelemetrySnapshot(battery_remaining_pct=100, battery_voltage_mv=12600, gps_fix_type=0, gps_satellites=0))
        sm.evaluate(TelemetrySnapshot(battery_remaining_pct=100, battery_voltage_mv=12600, gps_fix_type=0, gps_satellites=0))
        sm._guidance_start_time = time.time() - 1.0
        sm.evaluate(TelemetrySnapshot(
            battery_remaining_pct=100, battery_voltage_mv=12600,
            gps_fix_type=0, gps_satellites=0, flight_mode=4, is_armed=True,
            local_north_m=-46.0, local_east_m=12.0, local_down_m=-15.0, gps_alt_m=599.0
        ))
        assert sm.state == FlightState.TIER3_DESCENT

             # Trigger visual servo submission during descent
        sm._last_servo_submit_time = 0.0
        sm.evaluate(TelemetrySnapshot(
            battery_remaining_pct=100, battery_voltage_mv=12600,
            gps_fix_type=0, gps_satellites=0, flight_mode=4, is_armed=True,
            local_north_m=-48.0, local_east_m=13.5, local_down_m=-12.0, gps_alt_m=596.0
        ))
        # Poll: keep evaluating until servo correction is applied (up to 2s)
        for _ in range(20):
            time.sleep(0.1)
            sm._last_servo_submit_time = 0.0  # Force re-submission each poll
            sm.evaluate(TelemetrySnapshot(
                battery_remaining_pct=100, battery_voltage_mv=12600,
                gps_fix_type=0, gps_satellites=0, flight_mode=4, is_armed=True,
                local_north_m=-48.0, local_east_m=13.5, local_down_m=-12.0, gps_alt_m=596.0
            ))
            if sm.servo_corrections_count >= 1:
                break
        assert sm.servo_corrections_count >= 1
        assert len(conn.position_targets) >= 2
        print(f"  ✓ PASSED (Closed-loop servo corrections applied={sm.servo_corrections_count})")
    finally:
        worker.stop()


def test_visual_obstacle_triggers_cluster_diversion():
    print("\n── Test 4: Dynamic Visual Obstacle on Pad -> Automatic Cluster Diversion ──")
    conn = MockConnection()
    SHARED_CAMERA.clear_dynamic_obstacles()
    # Inject a dynamic obstacle right on top of the primary cluster at (N=-50m, E=15m)
    SHARED_CAMERA.add_dynamic_obstacle(north_m=-50.0, east_m=15.0, radius_m=2.5)
    worker = PerceptionWorker(camera=SHARED_CAMERA, use_gpu=True)
    worker.start()
    try:
        sm = FailsafeStateMachine(
            conn, vision_engine=SHARED_VISION_ENGINE, perception_worker=worker
        )
        sm.evaluate(TelemetrySnapshot(battery_remaining_pct=100, battery_voltage_mv=12600, gps_fix_type=3, gps_satellites=10))
        sm.evaluate(TelemetrySnapshot(battery_remaining_pct=100, battery_voltage_mv=12600, gps_fix_type=0, gps_satellites=0))
        sm.evaluate(TelemetrySnapshot(battery_remaining_pct=100, battery_voltage_mv=12600, gps_fix_type=0, gps_satellites=0))
        sm.evaluate(TelemetrySnapshot(battery_remaining_pct=100, battery_voltage_mv=12600, gps_fix_type=0, gps_satellites=0))
        sm._guidance_start_time = time.time() - 1.0
        # ── KEY FIX: Position drone ~3.6m AWAY from target (N=-50, E=15) ──
        # so it does NOT immediately trigger Stage 2 terminal landing.
        # Distance = sqrt(3² + 2²) = 3.6m > 1.5m arrival radius.
        # Camera at 12m AGL covers ±7m, so obstacle at target is still in FOV.
        sm.evaluate(TelemetrySnapshot(
            battery_remaining_pct=100, battery_voltage_mv=12600,
            gps_fix_type=0, gps_satellites=0, flight_mode=4, is_armed=True,
            local_north_m=-47.0, local_east_m=13.0, local_down_m=-12.0, gps_alt_m=596.0
        ))
        assert sm.state == FlightState.TIER3_DESCENT
        first_cluster = (sm._vision_target["x"], sm._vision_target["y"])

        # Poll: keep evaluating until perception worker detects obstacle and diverts
        sm._last_servo_submit_time = 0.0
        second_cluster = first_cluster
        for _ in range(20):
            time.sleep(0.1)
            sm._last_servo_submit_time = 0.0  # Force re-submission each poll
            sm.evaluate(TelemetrySnapshot(
                battery_remaining_pct=100, battery_voltage_mv=12600,
                gps_fix_type=0, gps_satellites=0, flight_mode=4, is_armed=True,
                local_north_m=-47.0, local_east_m=13.0, local_down_m=-12.0, gps_alt_m=596.0
            ))
            second_cluster = (sm._vision_target["x"], sm._vision_target["y"])
            if second_cluster != first_cluster:
                break

        assert first_cluster != second_cluster, "Expected visual obstacle diversion to alternate cluster!"
        print(f"  ✓ PASSED (Visual obstacle on {first_cluster} -> diverted to {second_cluster})")
    finally:
        SHARED_CAMERA.clear_dynamic_obstacles()
        worker.stop()


if __name__ == "__main__":
    print("═" * 70)
    print("  SPRINT 3: THREAD-ISOLATED ONNX PERCEPTION & VISUAL SERVOING TESTS")
    print("═" * 70)

    test_pinhole_camera_projection_math()
    test_thread_isolated_perception_worker()
    test_closed_loop_visual_servoing_in_state_machine()
    test_visual_obstacle_triggers_cluster_diversion()

    print("\n" + "═" * 70)
    print("  ALL 4 SPRINT 3 VISUAL SERVOING TESTS PASSED ✓")
    print("═" * 70)