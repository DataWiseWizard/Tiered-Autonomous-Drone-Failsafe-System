"""
v2.0: Exports YOLOv8n-seg -> ONNX
and generates a hardware latency/throughput benchmark table on NVIDIA RTX 4060.
"""
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from synthetic_camera import SyntheticDownwardCamera
from vision_landing import VisionLandingEngine

MODELS_DIR = PROJECT_ROOT / "models"
DOCS_DIR = PROJECT_ROOT / "docs"


def benchmark_fn(fn, warmup: int = 10, runs: int = 50):
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    latencies_ms = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    arr = np.array(latencies_ms)
    mean_ms = float(np.mean(arr))
    p50_ms = float(np.percentile(arr, 50))
    p95_ms = float(np.percentile(arr, 95))
    fps = 1000.0 / mean_ms if mean_ms > 0 else 0.0
    return mean_ms, p50_ms, p95_ms, fps


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    print("═" * 72)
    print("  SPRINT 3: YOLOv8n-seg -> ONNX / TensorRT EXPORT & RTX 4060 BENCHMARK")
    print("═" * 72)

    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"  Detected Compute Device: {device_name}")

    # 1. Generate a realistic 640x640 downward aerial test frame
    ve = VisionLandingEngine()
    cam = SyntheticDownwardCamera(prob_matrix=ve.prob_matrix)
    sample_frame = cam.render_frame(north_m=-50.0, east_m=15.0, alt_agl_m=15.0)

    # 2. Load YOLOv8n-seg & Export to ONNX
    from ultralytics import YOLO

    pt_path = MODELS_DIR / "yolov8n-seg.pt"
    model = YOLO(str(pt_path))

    onnx_path = MODELS_DIR / "yolov8n-seg.onnx"
    if not onnx_path.exists():
        print("\n[1/3] Exporting YOLOv8n-seg to ONNX (640x640)...")
        exported = model.export(format="onnx", imgsz=640, simplify=True)
        if Path(exported).exists() and Path(exported) != onnx_path:
            shutil.move(str(exported), str(onnx_path))
    print(f"  ✓ ONNX Model Ready: {onnx_path}")

    results_table = []

    # 3. Benchmark PyTorch FP32 & FP16 (CUDA if available)
    if torch.cuda.is_available():
        print("\n[2/3] Benchmarking PyTorch FP32 & FP16 (CUDA)...")
        mean_ms, p50, p95, fps = benchmark_fn(
            lambda: model.predict(sample_frame, imgsz=640, device=0, half=False, verbose=False)
        )
        results_table.append(("PyTorch FP32 (CUDA)", "FP32", mean_ms, p50, p95, fps))

        mean_ms, p50, p95, fps = benchmark_fn(
            lambda: model.predict(sample_frame, imgsz=640, device=0, half=True, verbose=False)
        )
        results_table.append(("PyTorch AMP FP16 (CUDA)", "FP16", mean_ms, p50, p95, fps))

    # 4. Benchmark ONNX Runtime (CPU & CUDA/TensorRT Execution Providers)
    import onnxruntime as ort

    available_eps = ort.get_available_providers()
    print(f"\n[3/3] Benchmarking ONNX Runtime (Available Providers: {available_eps})...")

    blob = sample_frame.astype(np.float32) / 255.0
    blob = np.transpose(blob, (2, 0, 1))[np.newaxis, ...]  # (1, 3, 640, 640)

    # ONNX CPU (Used by Hugging Face Space)
    sess_cpu = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = sess_cpu.get_inputs()[0].name
    mean_ms, p50, p95, fps = benchmark_fn(lambda: sess_cpu.run(None, {input_name: blob}), warmup=5, runs=25)
    results_table.append(("ONNX Runtime (CPU - HF Space)", "FP32", mean_ms, p50, p95, fps))

    # ONNX CUDA Execution Provider
    if "CUDAExecutionProvider" in available_eps:
        try:
            sess_cuda = ort.InferenceSession(
                str(onnx_path), providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
            )
            mean_ms, p50, p95, fps = benchmark_fn(lambda: sess_cuda.run(None, {input_name: blob}))
            results_table.append(("ONNX Runtime (CUDA EP)", "FP32", mean_ms, p50, p95, fps))
        except Exception as e:
            print(f"  [Note] ONNX CUDA EP fallback: {e}")

    # Print & Save Markdown Benchmark Table
    md_lines = [
        f"# Edge CV Inference Benchmark — `{device_name}`",
        "",
        "| Runtime / Backend | Precision | Mean Latency (ms) | Median P50 (ms) | Tail P95 (ms) | Throughput (FPS) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ]
    for backend, prec, mean_ms, p50, p95, fps in results_table:
        md_lines.append(
            f"| **{backend}** | `{prec}` | {mean_ms:.2f} ms | {p50:.2f} ms | {p95:.2f} ms | **{fps:.1f} FPS** |"
        )

    md_text = "\n".join(md_lines)
    print("\n" + md_text + "\n")

    out_md = DOCS_DIR / "yolo_benchmark.md"
    out_md.write_text(md_text + "\n")
    print(f"  ✓ Benchmark saved to: {out_md}")
    print("═" * 72)


if __name__ == "__main__":
    main()