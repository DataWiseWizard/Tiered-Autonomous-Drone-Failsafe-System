"""
Tiered Drone Failsafe Hub — Gradio Dashboard
Drop-in replacement for Streamlit (HF ZeroGPU/CPU compatible)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vision_landing import VisionLandingEngine

import huggingface_hub
if not hasattr(huggingface_hub, "HfFolder"):
    class _HfFolderCompat:
        @classmethod
        def get_token(cls): return huggingface_hub.get_token()
    huggingface_hub.HfFolder = _HfFolderCompat

import gradio as gr


# Lazy-load YOLO only when Tab 2 is used
_yolo_model = None
def get_yolo_model():
    global _yolo_model
    if _yolo_model is None:
        from ultralytics import YOLO
        onnx_path = PROJECT_ROOT / "models" / "yolov8n-seg.onnx"
        if onnx_path.exists():
            _yolo_model = YOLO(str(onnx_path))
    return _yolo_model


# ═══════════════════════════════════════════════════════
# TAB 1: Spatial Cluster Explorer
# ═══════════════════════════════════════════════════════
def load_cluster_explorer():
    ve = VisionLandingEngine()
    fig = go.Figure(data=go.Heatmap(
        z=ve.prob_matrix, colorscale="RdYlGn", reversescale=True,
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
            text=[f"Target (Tier {target['tier']})"],
            textposition="bottom center", name="Safe Cluster",
        ))
        info_text = (f"✅ Safest cluster at grid ({target['x']}, {target['y']}) "
                        f"Tier: {target['tier']} | Safety Prob: {target.get('prob', 0):.3f} | "
                        f"Area: {target.get('area_m2', 75):.0f} m² | Sensor Conf: {target.get('mean_conf', 0):.2f}"
        )
    fig.update_layout(
        title="Connected-Component Safe Zone Selection (BFS Flood-Fill)",
        xaxis_title="Grid X (60 tiles × 5 m)", yaxis_title="Grid Y (50 tiles × 5 m)",
        height=650,
    )
    return fig, info_text


# ═══════════════════════════════════════════════════════
# TAB 2: ONNX Segmentation Tester
# ═══════════════════════════════════════════════════════
def run_segmentation(image_input):
    if image_input is None:
        return None, "No image provided."
    model = get_yolo_model()
    if model is None:
        return None, "❌ ONNX model not found at models/yolov8n-seg.onnx"
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
        summary_lines.append("No COCO-class objects detected.")
    return Image.fromarray(res_plotted), "\n".join(summary_lines)


# ═══════════════════════════════════════════════════════
# TAB 3: Monte Carlo Replay Dashboard
# ═══════════════════════════════════════════════════════
def load_monte_carlo_data():
    jsonl_path = PROJECT_ROOT / "logs" / "monte_carlo_summary.jsonl"
    if not jsonl_path.exists():
        return None
    data = []
    with open(jsonl_path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return pd.DataFrame(data)

def update_mc_dashboard(modes, faults):
    df = load_monte_carlo_data()
    if df is None:
        return {}, {}, {}, {}, None, None, None
    filtered = df[(df["mode"].isin(modes)) & (df["fault_type"].isin(faults))]
    if filtered.empty:
        return {"value": "N/A"}, {"value": "N/A"}, {"value": "N/A"}, {"value": "N/A"}, None, None, None

    success_rate = f"{filtered['landed_safely'].mean() * 100:.1f}%"
    mean_err = f"{filtered['radial_error_m'].mean():.2f} m"
    cep50 = f"{np.percentile(filtered['radial_error_m'], 50):.2f} m"
    cep95 = f"{np.percentile(filtered['radial_error_m'], 95):.2f} m"

    fig_hist = px.histogram(
        filtered, x="radial_error_m", color="mode", barmode="overlay",
        title="Radial Error Distribution (m)", nbins=30,
        labels={"radial_error_m": "Radial Error (m)", "mode": "Architecture"},
    )
    fig_hist.add_vline(x=2.5, line_dash="dash", line_color="red", annotation_text="2.5m Limit")

    fig_bar = px.bar(
        filtered.groupby(["fault_type", "mode"])["landed_safely"].mean().reset_index()
                .assign(pct=lambda d: d["landed_safely"] * 100),
        x="fault_type", y="pct", color="mode", barmode="group",
        title="Success Rate (%) by Fault Type",
        labels={"fault_type": "Fault Type", "pct": "Success Rate (%)", "mode": "Architecture"},
    )

    kpis = [
        {"label": "Success Rate", "value": success_rate},
        {"label": "Mean Radial Error", "value": mean_err},
        {"label": "CEP₅₀ (Median)", "value": cep50},
        {"label": "CEP₉₅ (95th Pct)", "value": cep95},
    ]
    return kpis[0], kpis[1], kpis[2], kpis[3], fig_hist, fig_bar, filtered


# ═══════════════════════════════════════════════════════
# BUILD GRADIO UI
# ═══════════════════════════════════════════════════════
with gr.Blocks(title="🚁 TDFH Dashboard", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🚁 Tiered Drone Failsafe Hub\n**Sprint 4.2** — Cluster Explorer · ONNX Segmentation · Monte Carlo Replay")

    with gr.Tabs():
        # ── TAB 1 ──
        with gr.Tab("🗺️ Spatial Cluster Explorer"):
            cluster_plot = gr.Plot(label="60×50 Terrain Heatmap")
            cluster_info = gr.Textbox(label="Cluster Status", interactive=False)
            gr.Button("🔄 Load Terrain Grid", variant="primary").click(
                fn=load_cluster_explorer, outputs=[cluster_plot, cluster_info]
            )

        # ── TAB 2 ──
        with gr.Tab("🧠 ONNX Segmentation Tester"):
            with gr.Row():
                img_input = gr.Image(type="pil", label="Upload Aerial Image")
                img_output = gr.Image(type="pil", label="Segmentation Output")
            detection_summary = gr.Textbox(label="Detection Summary", lines=4, interactive=False)
            sample_dir = PROJECT_ROOT / "data" / "sample_aerials"
            samples = sorted([str(p) for p in sample_dir.glob("*") if p.suffix.lower() in (".jpg",".jpeg",".png")]) if sample_dir.exists() else []
            if samples:
                gr.Examples(examples=[[s] for s in samples], inputs=img_input, label="Or use bundled sample")
            img_input.change(fn=run_segmentation, inputs=img_input, outputs=[img_output, detection_summary])

        # ── TAB 3 ──
        with gr.Tab("📊 Monte Carlo Replay"):
            df_full = load_monte_carlo_data()
            if df_full is not None:
                mode_choices = list(df_full["mode"].unique())
                fault_choices = list(df_full["fault_type"].unique())
            else:
                mode_choices, fault_choices = [], []

            with gr.Row():
                mc_modes = gr.CheckboxGroup(choices=mode_choices, value=mode_choices, label="Architecture")
                mc_faults = gr.CheckboxGroup(choices=fault_choices, value=fault_choices, label="Fault Type")

            with gr.Row():
                kpi1 = gr.JSON(label="Success Rate")
                kpi2 = gr.JSON(label="Mean Radial Error")
                kpi3 = gr.JSON(label="CEP₅₀")
                kpi4 = gr.JSON(label="CEP₉₅")

            mc_hist = gr.Plot(label="Radial Error Distribution")
            mc_bar = gr.Plot(label="Success Rate by Fault Type")
            mc_table = gr.Dataframe(label="Raw Trial Data", interactive=False)

            for widget in [mc_modes, mc_faults]:
                widget.change(
                    fn=update_mc_dashboard,
                    inputs=[mc_modes, mc_faults],
                    outputs=[kpi1, kpi2, kpi3, kpi4, mc_hist, mc_bar, mc_table],
                )
            # Auto-load on page open
            demo.load(
                fn=update_mc_dashboard,
                inputs=[mc_modes, mc_faults],
                outputs=[kpi1, kpi2, kpi3, kpi4, mc_hist, mc_bar, mc_table],
            )

if __name__ == "__main__":
    demo.launch()