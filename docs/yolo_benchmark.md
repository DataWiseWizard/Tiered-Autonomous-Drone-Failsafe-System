# Edge CV Inference Benchmark — `NVIDIA GeForce RTX 4060 Laptop GPU`

| Runtime / Backend | Precision | Mean Latency (ms) | Median P50 (ms) | Tail P95 (ms) | Throughput (FPS) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **PyTorch FP32 (CUDA)** | `FP32` | 13.89 ms | 13.97 ms | 14.60 ms | **72.0 FPS** |
| **PyTorch AMP FP16 (CUDA)** | `FP16` | 11.52 ms | 11.43 ms | 13.09 ms | **86.8 FPS** |
| **ONNX Runtime (CPU - HF Space)** | `FP32` | 25.63 ms | 25.54 ms | 26.22 ms | **39.0 FPS** |
| **ONNX Runtime (CUDA EP)** | `FP32` | 11.59 ms | 11.49 ms | 11.93 ms | **86.3 FPS** |
