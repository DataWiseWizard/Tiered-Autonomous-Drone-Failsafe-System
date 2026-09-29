# System Architecture

## State Machine Flow

```mermaid
stateDiagram-v2
    [*] --> BOOT: System Init
    BOOT --> NOMINAL: GPS Lock + Battery OK

    NOMINAL --> TIER1_BATTERY_WARNING: Battery < 20%
    NOMINAL --> TIER1_BATTERY_RTL: Battery < 15%
    NOMINAL --> TIER1_BATTERY_LAND: Battery < 8%
    NOMINAL --> TIER1_RC_LOST: RC Lost > 3s
    NOMINAL --> TIER2_GPS_DEGRADED: EKF Var > 1.0
    NOMINAL --> TIER2_GPS_DENIED: GPS Fix < 3 OR EKF Var > 5.0

    TIER1_BATTERY_WARNING --> NOMINAL: Battery Recovered
    TIER1_BATTERY_WARNING --> TIER1_BATTERY_RTL: Battery Dropped
    TIER1_BATTERY_RTL --> TIER1_BATTERY_LAND: Battery Critical
    TIER1_RC_LOST --> NOMINAL: RC Restored

    TIER2_GPS_DEGRADED --> NOMINAL: EKF Recovered
    TIER2_GPS_DEGRADED --> TIER2_GPS_DENIED: Degradation Persisted

    TIER2_GPS_DENIED --> TIER3_VISION_HOVER: Vision Landing Initiated
    TIER3_VISION_HOVER --> TIER3_DESCENT: Safe Zone Found + GUIDED Command
    TIER3_DESCENT --> LANDED: Touchdown Confirmed

    TIER1_BATTERY_LAND --> LANDED: Emergency Land Complete
    LANDED --> [*]

    note right of TIER3_VISION_HOVER
        Logistic Regression Model
        Pre-computed Safety Grid
        Centroid → NED Offset
        GUIDED Mode Descent
    end note

    note right of TIER2_GPS_DENIED
        EKF_STATUS_REPORT variance
        GPS_RAW_INT fix type + sats
        Position jump detection
    end note
```

## System Overview

```mermaid
flowchart TD
    A[ArduPilot SITL / Flight Controller] -->|MAVLink v2 UDP| B[Companion Computer]

    B --> C{Tier 1: Telemetry Watchdog}
    C -->|Battery OK, RC OK| D{Tier 2: GPS/EKF Validator}
    C -->|Battery < 15%| E[RTL Command]
    C -->|Battery < 8%| F[Emergency LAND]
    C -->|RC Lost > 3s| G[LOITER → RTL]

    D -->|EKF Normal, GPS ≥ 3D| H[Nominal Flight]
    D -->|EKF Variance High| I[GPS Degraded Alert]
    D -->|No Fix / Spoof Detected| J[Tier 3: Emergency Vision Mode]

    I -->|Recovers| H
    I -->|Degrades > 5s| J

    J --> K[Hover & Stabilize]
    K --> L[Query Safety Grid<br/>Logistic Regression Model]
    L --> M{Safe Zone Detected?}
    M -->|Yes, Prob > 0.8| N[Compute Centroid → NED Offset]
    M -->|No| O[Loiter & Abort]

    N --> P[GUIDED Mode:<br/>Fly to Target + Descend]
    P --> Q{Alt < Threshold OR Timeout?}
    Q -->|Yes| R[LANDED ✓]
    Q -->|No| P

    style J fill:#ff6b6b,stroke:#c0392b
    style L fill:#feca57,stroke:#f39c12
    style R fill:#2ecc71,stroke:#27ae60
```

## Data Pipeline (From Notebook to Flight Stack)

```mermaid
flowchart LR
    A[Google Sheets CSV<br/>3000 Terrain Tiles] --> B[Feature Extraction<br/>8 terrain features]
    B --> C[Spatial Anchoring<br/>grid_x, grid_y]
    C --> D[Logistic Regression<br/>Pipeline + StandardScaler]
    D --> E[Safety Probability Grid<br/>60×50 heatmap]
    E --> F[Zone Tiering<br/>Preferred > 0.8<br/>Conditional 0.5–0.8<br/>Restricted < 0.5]
    F --> G[VisionLandingEngine<br/>find_safest_zone()]
    G --> H[Centroid → NED Offset<br/>→ GUIDED Descent]

    style D fill:#3498db,stroke:#2980b9
    style G fill:#e74c3c,stroke:#c0392b
```