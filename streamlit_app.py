"""
Tiered Drone Failsafe Hub — Interactive Streamlit Dashboard
Sprint 4.2: Cluster Explorer + ONNX Segmentation Tester + Monte Carlo Replay
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

# ── Path Setup ──
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

try:
    from vision_landing import VisionLandingEngine
except ImportError:
    st.error("Could not import VisionLandingEngine. Ensure src/ is in your path.")

# ── Ultralytics for ONNX inference (handles YOLOv8-seg mask decoding) ──
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

# ── Page Config ──
st.set_page_config(
    page_title="Tiered Drone Failsafe Hub",
    layout="wide",
    page_icon="🚁",
)
st.title("🚁 Tiered Drone Failsafe (TDFH) Dashboard")
st.caption(
    "Sprint 4.2 — Interactive $60 \\times 50$ Cluster Explorer · "
    "ONNX Segmentation Tester · Monte Carlo Flight Replay"
)

# ── Tabs ──
tab1, tab2, tab3 = st.tabs([
    "🗺️ 1. Spatial Cluster Explorer",
    "🧠 2. ONNX Segmentation Tester",
    "📊 3. Monte Carlo Replay Dashboard",
])

# ══════════════════════════════════════════════════════════
# TAB 1: Spatial Cluster Explorer
# ══════════════════════════════════════════════════════════
with tab1:
    st.header("60×50 Terrain Probability & Safe Zone Clusters")

    if st.button("🔄 Load Terrain Grid", type="primary"):
        ve = VisionLandingEngine()

        # Probability heatmap
        fig = go.Figure(data=go.Heatmap(
            z=ve.prob_matrix,
            colorscale="RdYlGn",
            reversescale=True,
            x=np.arange(60),
            y=np.arange(50),
            colorbar=dict(title="Safe Probability"),
        ))

        # Overlay starting position (center)
        fig.add_trace(go.Scatter(
            x=[30], y=[25],
            mode="markers+text",
            marker=dict(color="blue", size=16, symbol="x"),
            text=["Start"], textposition="top center",
            name="Drone Start",
        ))

        # Find and overlay the safest reachable cluster
        target = ve.find_safest_zone(30, 25, max_distance_tiles=15)
        if target:
            fig.add_trace(go.Scatter(
                x=[target["x"]], y=[target["y"]],
                mode="markers+text",
                marker=dict(color="black", size=18, symbol="star"),
                text=[f"Target (Tier {target['tier']})"],
                textposition="bottom center",
                name="Safe Cluster Centroid",
            ))
            st.success(
                f"✅ Safest cluster found at grid ({target['x']}, {target['y']}) "
                f"— Tier {target['tier']}, confidence={target.get('confidence', 0):.2f}"
            )
        else:
            st.warning("⚠️ No reachable safe cluster found above 0.3 floor.")

        fig.update_layout(
            title="Connected-Component Safe Zone Selection (BFS Flood-Fill)",
            xaxis_title="Grid X (60 tiles × 5 m)",
            yaxis_title="Grid Y (50 tiles × 5 m)",
            height=650,
        )
        st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════
# TAB 2: ONNX Segmentation Tester
# ══════════════════════════════════════════════════════════
with tab2:
    st.header("YOLOv8n-seg ONNX Edge Inference")

    ONNX_PATH = PROJECT_ROOT / "models" / "yolov8n-seg.onnx"

    if not ULTRALYTICS_AVAILABLE:
        st.warning(
            "Install ultralytics (`pip install ultralytics`) to run "
            "YOLOv8-seg ONNX inference with automatic mask decoding."
        )
    elif not ONNX_PATH.exists():
        st.error(
            f"ONNX model not found at: `{ONNX_PATH}`. "
            "Please export your model first or update the path."
        )
    else:
        st.info(f"✅ Model loaded from: `{ONNX_PATH}`")
        model = YOLO(str(ONNX_PATH))

        # ── Image source: manual upload OR bundled sample ──
        source = st.radio(
            "Image source",
            ["Upload image", "Use bundled sample"],
            horizontal=True,
        )
        image = None
        if source == "Upload image":
            uploaded_file = st.file_uploader(
                "Upload an aerial/satellite image", type=["jpg", "jpeg", "png"]
            )
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
        else:
            sample_dir = PROJECT_ROOT / "data" / "sample_aerials"
            if sample_dir.exists():
                samples = sorted(
                    p for p in sample_dir.iterdir()
                    if p.suffix.lower() in (".jpg", ".jpeg", ".png")
                )
                if samples:
                    choice = st.selectbox("Sample image", [p.name for p in samples])
                    image = Image.open(sample_dir / choice)
                else:
                    st.info("No samples found in `data/sample_aerials/`.")
            else:
                st.info("Folder `data/sample_aerials/` not found.")

        if image is not None:
            col_a, col_b = st.columns(2)
            with col_a:
                st.image(image, caption="Original Input", use_column_width=True)

            with st.spinner("Running ONNX inference..."):
                results = model(image, verbose=False)
                res_plotted = results[0].plot()

            with col_b:
                st.image(
                    res_plotted,
                    caption="Segmentation Output (Masks + Bounding Boxes)",
                    use_column_width=True,
                )

            # Detection summary
            if results[0].boxes is not None and len(results[0].boxes) > 0:
                st.subheader("Detection Summary")
                boxes = results[0].boxes
                for i in range(len(boxes)):
                    conf = float(boxes.conf[i])
                    cls_id = int(boxes.cls[i])
                    cls_name = results[0].names.get(cls_id, f"class_{cls_id}")
                    st.text(f"  • {cls_name}: {conf:.2%} confidence")
            else:
                st.info("No COCO-class objects detected in this image.")
                
# ══════════════════════════════════════════════════════════
# TAB 3: Monte Carlo Replay Dashboard
# ══════════════════════════════════════════════════════════
with tab3:
    st.header("Statistical Fault-Injection Replay")

    jsonl_path = PROJECT_ROOT / "logs" / "monte_carlo_summary.jsonl"

    if not jsonl_path.exists():
        st.warning(
            "No `monte_carlo_summary.jsonl` found. "
            "Run the Monte Carlo harness first: `python -m tools.monte_carlo_harness`"
        )
    else:
        # Load JSONL
        data = []
        with open(jsonl_path, "r") as f:
            for line in f:
                data.append(json.loads(line))
        df = pd.DataFrame(data)

        # ── Interactive Filters ──
        col1, col2 = st.columns(2)
        with col1:
            modes = st.multiselect(
                "Architecture Configuration",
                df["mode"].unique(),
                default=df["mode"].unique(),
            )
        with col2:
            faults = st.multiselect(
                "Fault Type Injected",
                df["fault_type"].unique(),
                default=df["fault_type"].unique(),
            )

        filtered_df = df[
            (df["mode"].isin(modes)) & (df["fault_type"].isin(faults))
        ]

        if filtered_df.empty:
            st.warning("No trials match the selected filters.")
        else:
            # ── KPI Metrics ──
            m1, m2, m3, m4 = st.columns(4)
            m1.metric(
                "Safe Pad Success Rate",
                f"{filtered_df['landed_safely'].mean() * 100:.1f}%",
            )
            m2.metric(
                "Mean Radial Error",
                f"{filtered_df['radial_error_m'].mean():.2f} m",
            )
            m3.metric(
                "CEP₅₀ (Median)",
                f"{np.percentile(filtered_df['radial_error_m'], 50):.2f} m",
            )
            m4.metric(
                "CEP₉₅ (95th Pct)",
                f"{np.percentile(filtered_df['radial_error_m'], 95):.2f} m",
            )

            st.divider()

            # ── Visualizations ──
            col_a, col_b = st.columns(2)
            with col_a:
                fig_hist = px.histogram(
                    filtered_df,
                    x="radial_error_m",
                    color="mode",
                    barmode="overlay",
                    title="Radial Error Distribution (m)",
                    nbins=30,
                    labels={"radial_error_m": "Radial Error (m)", "mode": "Architecture"},
                )
                fig_hist.add_vline(
                    x=2.5, line_dash="dash", line_color="red",
                    annotation_text="2.5 m Safe Pad Limit",
                )
                st.plotly_chart(fig_hist, use_container_width=True)

            with col_b:
                fig_pie = px.pie(
                    filtered_df,
                    names="landed_safely",
                    title="Overall Landing Success vs Failure",
                    color="landed_safely",
                    color_discrete_map={True: "green", False: "red"},
                )
                st.plotly_chart(fig_pie, use_container_width=True)

            # ── Fault-type breakdown ──
            st.subheader("Success Rate by Fault Type")
            fault_summary = (
                filtered_df.groupby(["fault_type", "mode"])["landed_safely"]
                .mean()
                .reset_index()
            )
            fault_summary["landed_safely"] *= 100
            fig_bar = px.bar(
                fault_summary,
                x="fault_type",
                y="landed_safely",
                color="mode",
                barmode="group",
                title="Success Rate (%) by Fault Type & Architecture",
                labels={
                    "fault_type": "Fault Type",
                    "landed_safely": "Success Rate (%)",
                    "mode": "Architecture",
                },
            )
            st.plotly_chart(fig_bar, use_container_width=True)

            # ── Raw Data Explorer ──
            with st.expander("📋 View Raw Trial Data"):
                st.dataframe(filtered_df, use_container_width=True)