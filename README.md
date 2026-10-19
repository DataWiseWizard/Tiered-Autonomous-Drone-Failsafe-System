# 🛸 Tiered Autonomous Drone Failsafe Hierarchy & Edge CV Guidance

**A resilient, closed-loop autonomy and sensor-fusion safety stack for multirotor drones. When GPS is jammed, spoofed, or denied, the drone doesn't crash—it estimates its state via optical flow, segments terrain via TensorRT/ONNX YOLOv8n-seg, selects a contiguous safe landing pad via connected-component clustering, and executes an autonomous visual-servoed descent.**

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://python.org)
[![ArduPilot](https://img.shields.io/badge/ArduPilot-Copter%204.8--dev-green.svg)](https://ardupilot.org)
[![TensorRT](https://img.shields.io/badge/TensorRT-INT8%2FFP16-orange.svg)](https://developer.nvidia.com/tensorrt)
[![Hugging Face Space](https://img.shields.io/badge/Hugging%20Face-Live%20Demo-yellow.svg)](https://huggingface.co/spaces/your-username/tiered-drone-failsafe-hub)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🎯 The Engineering Problem

Autonomous multirotor UAVs depend heavily on Global Navigation Satellite Systems (GNSS/GPS) for position holding and Return-to-Launch (RTL) recovery. In urban canyons, dense vegetation, near electronic jamming units, or during adversarial spoofing attacks, GPS can be abruptly lost or subtly hijacked. 

When GPS fails under traditional flight stacks:
1. **Dead-Reckoning Divergence:** Uncompensated inertial sensors ($\iint a \, dt^2$) accumulate quadratic position drift ($> 15\text{ m}$ within 10 seconds), leading to uncontrolled crashes.
2. **Blind Spot Landings:** Standard auto-land routines force the drone down onto whatever is directly beneath it—whether that is moving traffic, water, steep cliffs, or high-voltage lines.

---

## 🏗️ The Solution: Three-Tier Failsafe Architecture

The **Tiered Autonomous Drone Failsafe Hierarchy (TDFH)** operates as a deterministic, $10\text{ Hz}$ state machine running on a companion computer interfacing with ArduPilot via MAVLink v2[cite: 1, 2]:

| Tier | Trigger Condition | Escalation / Mitigation Action |
| :--- | :--- | :--- |
| **Tier 1: Telemetry** | Battery $< 15\%$ or $< 11.0\text{V}$<br>RC Signal Loss $> 3.0\text{ s}$ | Triggers autonomous `RTL` or `LOITER` hold<br>**Global Override:** Battery $< 8\%$ forces emergency `LAND`[cite: 1, 6] |
| **Tier 2: Environmental** | GPS fix lost / EKF horizontal variance $> 1.0$<br>Dual-GPS divergence $> 3\text{ m}$ / $\chi^2$ innovation $> 6.5$ | Validates navigation integrity; detects GPS spoofing vs denial; escalates to Tier 3[cite: 1, 5, 6] |
| **Tier 3: Vision Landing** | Confirmed GPS denial / EKF variance $> 5.0$ | Triggers emergency hover $\rightarrow$ queries connected-component safety grid $\rightarrow$ executes staged visual-servoed descent[cite: 1, 4] |

---

## 🧭 System Architecture & Data Flow

```mermaid
flowchart TD
    A[ArduPilot Copter SITL / Pixhawk] -->|MAVLink v2 UDP 14550| B[Companion Computer: 10Hz Python Stack]

    B --> C{Tier 1: Telemetry Watchdog}
    C -->|Battery OK, RC OK| D{Tier 2: Environmental Validator}
    C -->|Battery < 15%| E[RTL Command]
    C -->|Battery < 8%| F[Emergency LAND - Global Override]
    C -->|RC Lost > 3s| G[LOITER Position Hold]

    D -->|EKF Normal, GPS 3D Fix| H[Nominal Flight]
    D -->|EKF Var > 1.0| I[GPS Degraded Alert]
    D -->|No Fix / EKF Var > 5.0 / Spoofing Detected| J[Tier 3: Emergency Vision Mode]

    I -->|Recovers| H
    I -->|Persists| J

    J --> K[Sensor Fusion Engine: Optical Flow + Rangefinder + Baro]
    K --> L[Query 60x50 Safety Grid: Logistic Regression Pipeline]
    L --> M[Connected-Component Cluster Selection (min-area >= 75 m²)]
    M --> N[Battery-Aware Reachability & Confidence Gating]

    N --> O[GUIDED Mode: Stage 1 Approach to Staging Alt (-10m AGL)]
    O --> P{Thread-Isolated Perception Worker: YOLOv8n-seg ONNX}
    P -->|Safe Pad Centroid| Q[1-2 Hz Damped Proportional Visual Servoing (Kp = 0.65)]
    P -->|Dynamic Obstacle Detected| R[Divert to Alternate Safe Cluster]

    Q --> S{Horizontal Arrival <= 1.5m OR Terminal Gate}
    S -->|Reached Centroid| T[Stage 2: MAV_CMD_NAV_LAND / Touchdown Disarm]
    T --> U[LANDED State & Blackbox JSONL Finalized]

    style J fill:#ff6b6b,stroke:#c0392b
    style M fill:#feca57,stroke:#f39c12
    style P fill:#48dbfb,stroke:#0abde3
    style U fill:#2ecc71,stroke:#27ae60

## 🧠 Tier 3: ML Terrain Scoring & Connected-Component ClusteringThe vision engine scores a $60 \times 50$ terrain grid ($300\text{ m} \times 250\text{ m}$) using an scikit-learn Logistic Regression pipeline ($C=0.01$, balanced class weights) trained on 3,000 tiles across 8 engineered physical features:   slope_deg: Terrain steepness (tipping risk).roughness: Local elevation variance.edge_density: Structural visual clutter proxy.ndvi_mean: Vegetation cover index from red/NIR reflectance.shadow_fraction: Occlusion risk and hidden obstacle proxy.brightness_std: Surface homogeneity measure.object_density: Discrete obstacle density.confidence_score: Upstream perception reliability gate ($\ge 0.50$).   Why Connected-Component Cluster Selection?Traditional portfolio projects pick the single closest grid tile with $p > 0.80$. In reality, a single safe pixel surrounded by rocks causes a crash due to EKF drift. Our engine builds a binary mask $M$, runs 4-connected component labeling (BFS flood-fill), enforces a minimum area constraint ($\ge 3\text{ contiguous tiles} = 75\text{ m}^2$), and computes the probability-weighted cluster centroid.   Deterministic spatial safety heatmap with Tier 3 decision boundaries ($0.50$ / $0.80$) and selected safe cluster centroid ($\star$). Regenerate via python tools/generate_heatmap.py.   🏎️ Edge Computer Vision & RTX 4060 BenchmarksWe exported yolov8n-seg.pt to ONNX and benchmarked real-time inference across CPU and GPU runtimes on an NVIDIA GeForce RTX 4060 Laptop GPU (640x640 input resolution):