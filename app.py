"""
Tiered Drone Failsafe Hub — Gradio Dashboard
Hardened for Hugging Face Spaces (ZeroGPU & CPU Basic compatible)
"""
import json
import sys
from pathlib import Path

# ── 1. Patch huggingface_hub.HfFolder BEFORE importing gradio ──
import huggingface_hub
if not hasattr(huggingface_hub, "HfFolder"):
    class _HfFolderCompat:
        @classmethod
        def get_token(cls):
            return huggingface_hub.get_token()
    huggingface_hub.HfFolder = _HfFolderCompat

import gradio as gr
import gradio_client.utils as _gc_utils

# ── 2. Patch gradio_client bool schema bug ──
if hasattr(_gc_utils, "_json_schema_to_python_type"):
    _orig_schema_fn = _gc_utils._json_schema_to_python_type
    def _safe_json_schema_to_python_type(schema, defs=None):
        if isinstance(schema, bool):
            return "Any" if schema else "None"
        return _orig_schema_fn(schema, defs)
    _gc_utils._json_schema_to_python_type = _safe_json_schema_to_python_type

if hasattr(_gc_utils, "get_type"):
    _orig_get_type = _gc_utils.get_type
    def _safe_get_type(schema):
        if isinstance(schema, bool):
            return "Any"
        return _orig_get_type(schema)
    _gc_utils.get_type = _safe_get_type

try:
    import spaces
    gpu_decorator = spaces.GPU
except ImportError:
    def gpu_decorator(fn):
        return fn

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vision_landing import VisionLandingEngine

# Lazy-load YOLO safely
_yolo_model = None
def get_yolo_model():
    global _yolo_model
    if _yolo_model is None:
        try:
            from ultralytics import YOLO
            # Always load yolov8n-seg.pt natively to avoid ONNX Protobuf corruption errors on HF Spaces
            _yolo_model = YOLO("yolov8n-seg.pt")
        except Exception as e:
            print(f"[YOLO Load Error]: {e}")
    return _yolo_model

# ═══════════════════════════════════════════════════════
# TAB 1: Spatial Cluster Explorer
# ═══════════════════════════════════════════════════════
def load_cluster_explorer():
    try:
        ve = VisionLandingEngine()
        fig = go.Figure(data=go.Heatmap(
            z=ve.prob_matrix, colorscale="RdYlGn", reversescale=False,
            x=np.arange(60), y=np.arange(50),
            colorbar=dict(title="Safe Probability"),
        ))
        fig.add_trace(go.Scatter(
            x=[30], y=[25], mode="markers+text",
            marker=dict(color="blue", size=16, symbol="x"),
            text=["Start"], textposition="top center", name="Drone Start",
        ))
        target = ve.find_safest_zone(30, 25, max_distance_tiles=15)
        info_text = "⚠️ No reachable safe cluster found."
        if target:
            fig.add_trace(go.Scatter(
                x=[target["x"]], y=[target["y"]], mode="markers+text",
                marker=dict(color="black", size=18, symbol="star"),
                text=[f"Target ({target['tier']})"],
                textposition="bottom center", name="Safe Cluster",
            ))
            info_text = (
                f"✅ Safest cluster at grid ({target['x']}, {target['y']}) — "
                f"Tier: {target['tier']} | Safety Prob: {target.get('prob', 0):.3f} | "
                f"Area: {target.get('area_m2', 75):.0f} m² | Sensor Conf: {target.get('mean_conf', 0):.2f}"
            )
        fig.update_layout(
            title="Connected-Component Safe Zone Selection (BFS Flood-Fill)",
            xaxis_title="Grid X (60 tiles × 5 m)",
            yaxis_title="Grid Y (50 tiles × 5 m)",
            height=550,
        )
        return fig, info_text
    except Exception as e:
        fig = go.Figure()
        fig.update_layout(title=f"Error loading terrain grid: {str(e)}")
        return fig, f"❌ Error: {str(e)}"


# ═══════════════════════════════════════════════════════
# TAB 2: ONNX Segmentation Tester
# ═══════════════════════════════════════════════════════
@gpu_decorator
def run_segmentation(image_input):
    if image_input is None:
        return None, "No image provided."
    model = get_yolo_model()
    if model is None:
        return None, "❌ YOLO segmentation model could not be loaded."
    try:
        results = model(image_input, verbose=False)
        res_plotted = results[0].plot()
        summary_lines = []
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                conf = float(boxes.conf[i])
                cls_id = int(boxes.cls[i])
                cls_name = results[0].names.get(cls_id, f"class_{cls_id}")
                summary_lines.append(f"• {cls_name}: {conf:.2%}")
        else:
            summary_lines.append("No COCO-class obstacle detected — landing zone clear.")
        return Image.fromarray(res_plotted), "\n".join(summary_lines)
    except Exception as e:
        return image_input, f"❌ Inference Error: {str(e)}"


# ═══════════════════════════════════════════════════════
# TAB 3: Monte Carlo Replay Dashboard
# ═══════════════════════════════════════════════════════
def load_monte_carlo_data():
    jsonl_path = PROJECT_ROOT / "logs" / "monte_carlo_summary.jsonl"
    if not jsonl_path.exists():
        # Fallback dummy df if jsonl not found yet
        return pd.DataFrame([
            {"trial": 0, "mode": "v1.0 Open-Loop (IMU Only)", "fault_type": "GPS_DENIAL", "landed_safely": False, "radial_error_m": 8.5},
            {"trial": 1, "mode": "v2.0 Full Stack (Fusion + Visual Servoing)", "fault_type": "GPS_DENIAL", "landed_safely": True, "radial_error_m": 0.4},
        ])
    data = []
    with open(jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return pd.DataFrame(data) if data else None


def update_mc_dashboard(modes, faults):
    df = load_monte_carlo_data()
    if df is None or not modes or not faults:
        return "N/A", "N/A", "N/A", "N/A", None, None, pd.DataFrame()
    filtered = df[(df["mode"].isin(modes)) & (df["fault_type"].isin(faults))]
    if filtered.empty:
        return "N/A", "N/A", "N/A", "N/A", None, None, pd.DataFrame()

    success_rate = f"{filtered['landed_safely'].mean() * 100:.1f}%"
    mean_err = f"{filtered['radial_error_m'].mean():.2f} m"
    cep50 = f"{np.percentile(filtered['radial_error_m'], 50):.2f} m"
    cep95 = f"{np.percentile(filtered['radial_error_m'], 95):.2f} m"

    fig_hist = px.histogram(
        filtered, x="radial_error_m", color="mode", barmode="overlay",
        title="Radial Touchdown Error Distribution (m)", nbins=30,
        labels={"radial_error_m": "Radial Error (m)", "mode": "Architecture"},
    )
    fig_hist.add_vline(x=2.5, line_dash="dash", line_color="red", annotation_text="2.5m Safe Pad Limit")

    fig_bar = px.bar(
        filtered.groupby(["fault_type", "mode"])["landed_safely"].mean().reset_index()
                .assign(pct=lambda d: d["landed_safely"] * 100),
        x="fault_type", y="pct", color="mode", barmode="group",
        title="Landing Success Rate (%) by Fault Vector",
        labels={"fault_type": "Fault Type", "pct": "Success Rate (%)", "mode": "Architecture"},
    )

    return success_rate, mean_err, cep50, cep95, fig_hist, fig_bar, filtered


# ═══════════════════════════════════════════════════════
# BUILD GRADIO UI
# ═══════════════════════════════════════════════════════
with gr.Blocks(title="🚁 TDFH Dashboard") as demo:
    gr.Markdown(
        "# 🚁 Tiered Autonomous Drone Failsafe Hub\n"
        "**Interactive Portfolio Suite** — Connected-Component Cluster Explorer · "
        "YOLOv8n-seg Perception · Monte Carlo Fault-Injection Replay"
    )

    with gr.Tabs():
        # ── TAB 1 ──
        with gr.Tab("🗺️ Spatial Cluster Explorer"):
            cluster_plot = gr.Plot(label="60×50 Terrain Safety Probability Grid")
            cluster_info = gr.Textbox(label="Selected Contiguous Cluster Status", interactive=False)
            load_btn = gr.Button("🔄 Compute Safest Contiguous Landing Cluster", variant="primary")

        # ── TAB 2 ──
        with gr.Tab("🧠 ONNX Segmentation Tester"):
            with gr.Row():
                img_input = gr.Image(type="pil", label="Upload Downward Aerial Frame")
                img_output = gr.Image(type="pil", label="YOLOv8n-seg Output")
            detection_summary = gr.Textbox(label="Detection Summary", lines=4, interactive=False)
            sample_dir = PROJECT_ROOT / "data" / "sample_aerials"
            samples = (
                sorted([str(p) for p in sample_dir.glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
                if sample_dir.exists() else []
            )
            if samples:
                gr.Examples(examples=[[s] for s in samples], inputs=img_input, label="Bundled Aerial Samples")
            img_input.change(fn=run_segmentation, inputs=img_input, outputs=[img_output, detection_summary])

        # ── TAB 3 ──
        with gr.Tab("📊 Monte Carlo Replay"):
            df_full = load_monte_carlo_data()
            if df_full is not None:
                mode_choices = list(df_full["mode"].unique())
                fault_choices = list(df_full["fault_type"].unique())
            else:
                mode_choices, fault_choices = ["v1.0 Open-Loop (IMU Only)"], ["GPS_DENIAL"]

            with gr.Row():
                mc_modes = gr.CheckboxGroup(choices=mode_choices, value=mode_choices, label="Architecture")
                mc_faults = gr.CheckboxGroup(choices=fault_choices, value=fault_choices, label="Fault Type")

            with gr.Row():
                kpi1 = gr.Textbox(label="Safe Pad Success Rate (≤ 2.5m)", interactive=False)
                kpi2 = gr.Textbox(label="Mean Radial Error", interactive=False)
                kpi3 = gr.Textbox(label="CEP₅₀ (Median)", interactive=False)
                kpi4 = gr.Textbox(label="CEP₉₅ (95th Pct)", interactive=False)

            mc_hist = gr.Plot(label="Radial Error Distribution")
            mc_bar = gr.Plot(label="Success Rate by Fault Type")
            mc_table = gr.Dataframe(label="Raw Trial Data", interactive=False)

            for widget in [mc_modes, mc_faults]:
                widget.change(
                    fn=update_mc_dashboard,
                    inputs=[mc_modes, mc_faults],
                    outputs=[kpi1, kpi2, kpi3, kpi4, mc_hist, mc_bar, mc_table],
                )

    # Event bindings
    load_btn.click(fn=load_cluster_explorer, outputs=[cluster_plot, cluster_info])
    
    # Auto-load on page open
    demo.load(fn=load_cluster_explorer, outputs=[cluster_plot, cluster_info])
    demo.load(
        fn=update_mc_dashboard,
        inputs=[mc_modes, mc_faults],
        outputs=[kpi1, kpi2, kpi3, kpi4, mc_hist, mc_bar, mc_table],
    )

if __name__ == "__main__":
    demo.launch(show_api=False)