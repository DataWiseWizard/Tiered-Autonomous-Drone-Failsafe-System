"""
tools/generate_heatmap.py
Regenerates docs/safety_heatmap.png from the live terrain dataset.

Reuses the exact same model pipeline as the flight stack
(src/vision_landing.py), so the figure in the README is
guaranteed to match what the drone actually uses in Tier 3.

Usage:
    python tools/generate_heatmap.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Headless backend — no GUI needed
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from vision_landing import VisionLandingEngine  # noqa: E402


def main() -> None:
    print("Training safety model and building probability grid...")
    engine = VisionLandingEngine()

    # Pivot to the 60x50 spatial grid (rows = grid_y, cols = grid_x)
    pivot = engine.df_grid.pivot(index="grid_y", columns="grid_x",
                                 values="safety_prob")
    z = pivot.values

    fig, ax = plt.subplots(figsize=(9, 6.5))
    im = ax.imshow(z, cmap="RdYlGn", vmin=0.0, vmax=1.0,
                   origin="upper", aspect="auto")

    # Decision boundaries actually used by the flight stack
    cs = ax.contour(z, levels=[0.5, 0.8], colors=("black", "blue"),
                    linewidths=1.2)
    ax.clabel(cs, fmt={0.5: "restricted < 0.5", 0.8: "preferred > 0.8"},
              fontsize=8)

    # Mark the exact target the Tier 3 state machine would select
    target = engine.find_safest_zone(30, 25)
    if target:
        ax.plot(target["x"], target["y"], marker="*", markersize=18,
                color="blue", markeredgecolor="white",
                label=f"Tier 3 target (p={target['prob']:.2f}, {target['tier']})")
        ax.legend(loc="upper right", fontsize=9)

    ax.set_title("Autonomous Drone Landing Zone: Deterministic Spatial Safety Heatmap")
    ax.set_xlabel("Grid X Coordinate (Longitude Equivalent)")
    ax.set_ylabel("Grid Y Coordinate (Latitude Equivalent)")
    fig.colorbar(im, ax=ax).set_label("Safety Probability")

    out = REPO_ROOT / "docs" / "safety_heatmap.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"[OK] Saved heatmap -> {out}")


if __name__ == "__main__":
    main()