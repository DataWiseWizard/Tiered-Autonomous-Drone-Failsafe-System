# Monte Carlo Fault-Injection & Ablation Benchmark

**Trials per Configuration:** `100` randomized fault-injection flights (25% Total GPS Denial, 25% Haversine Jump Spoofing, 25% Dual-GPS Divergence Spoofing, 25% EKF $\chi^2$ Innovation Spoofing) across randomized starting coordinates and wind drift.

| Architecture Configuration | Trials ($N$) | Safe Pad Success Rate ($\le 2.5\text{ m}$) | Mean Radial Error | $\text{CEP}_{50}$ (Median) | $\text{CEP}_{95}$ (95th Pct) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **v1.0 Open-Loop (IMU Only)** | 100 | **95.0%** | 0.00 m | **0.00 m** | 0.00 m |
| **v2.0 Sensor Fusion (Flow + Rangefinder)** | 100 | **95.0%** | 0.00 m | **0.00 m** | 0.00 m |
| **v2.0 Full Stack (Fusion + Visual Servoing)** | 100 | **96.0%** | 0.46 m | **0.42 m** | 0.91 m |
