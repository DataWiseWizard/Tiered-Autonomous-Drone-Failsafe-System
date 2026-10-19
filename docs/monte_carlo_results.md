# Monte Carlo Fault-Injection & Ablation Benchmark

**Trials per Configuration:** `100` randomized fault-injection flights (25% Total GPS Denial, 25% Haversine Jump Spoofing, 25% Dual-GPS Divergence Spoofing, 25% EKF $\chi^2$ Innovation Spoofing) across randomized starting coordinates and wind drift.

| Architecture Configuration | Trials ($N$) | Safe Pad Success Rate ($\le 2.5\text{ m}$) | Mean Radial Error | $\text{CEP}_{50}$ (Median) | $\text{CEP}_{95}$ (95th Pct) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **v1.0 Open-Loop (IMU Only)** | 100 | **11.0%** | 7.29 m | **6.88 m** | 16.12 m |
| **v2.0 Sensor Fusion (Flow + Rangefinder)** | 100 | **88.0%** | 0.45 m | **0.38 m** | 0.83 m |
| **v2.0 Full Stack (Fusion + Visual Servoing)** | 100 | **82.0%** | 0.57 m | **0.37 m** | 0.67 m |
