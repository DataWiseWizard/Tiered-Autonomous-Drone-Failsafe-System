"""
Tier 3: Vision Landing Engine
Features:
  - Offline CSV caching (data/terrain_grid.csv) + joblib model persistence
  - Stratified 80/20 train/test split (no full-grid training leakage)
  - Connected-component cluster selection with minimum-area constraint
  - Sensor confidence_score gating and multi-cluster fallback hierarchy
"""
from collections import deque
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import structlog

logger = structlog.get_logger()

DATA_URL = "https://docs.google.com/spreadsheets/d/1tCQf9YVzj8zET1bjTlettAV5WfyeNpo4EBEjo5H1Z9Y/export?format=csv"
GRID_COLS = 60  # 60 columns x 50 rows = 3000 tiles
GRID_ROWS = 50

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOCAL_CSV_PATH = PROJECT_ROOT / "data" / "terrain_grid.csv"
LOCAL_MODEL_PATH = PROJECT_ROOT / "models" / "safety_lr_pipeline.joblib"


class VisionLandingEngine:
    # ── Zone & Cluster Thresholds ──
    PREFERRED_PROB_THRESHOLD = 0.80
    CONDITIONAL_PROB_THRESHOLD = 0.50
    ABSOLUTE_NO_LAND_FLOOR = 0.30
    MIN_CONFIDENCE_SCORE = 0.50
    MIN_CLUSTER_TILES = 3  # 3 tiles * 25 m^2/tile = 75 m^2 minimum contiguous area

    def __init__(
        self,
        csv_path: Optional[Path] = None,
        model_path: Optional[Path] = None,
        min_cluster_tiles: int = MIN_CLUSTER_TILES,
        min_confidence: float = MIN_CONFIDENCE_SCORE,
    ):
        self.csv_path = Path(csv_path) if csv_path else LOCAL_CSV_PATH
        self.model_path = Path(model_path) if model_path else LOCAL_MODEL_PATH
        self.min_cluster_tiles = min_cluster_tiles
        self.min_confidence = min_confidence
        self.grid_resolution_m = 5.0  # Each grid cell is 5m x 5m

        logger.info("INIT_VISION_ENGINE", status="Loading terrain data & safety model...")
        self.df = self._load_and_anchor_data()
        self.feature_cols = [
            col for col in self.df.columns if col not in ["label", "grid_x", "grid_y"]
        ]
        self.model = self._load_or_train_model()
        self._generate_terrain_grid()

    def _load_and_anchor_data(self) -> pd.DataFrame:
        """Load terrain CSV from local cache (or download once and cache) and anchor (grid_x, grid_y)."""
        if self.csv_path.exists():
            df = pd.read_csv(self.csv_path)
            source = str(self.csv_path)
        else:
            logger.info("DOWNLOADING_TERRAIN_CSV", url=DATA_URL)
            df = pd.read_csv(DATA_URL)
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(self.csv_path, index=False)
            source = "remote_url_cached_locally"

        # Spatial anchoring BEFORE any shuffle/split (matches notebook exactly)
        df["grid_y"] = df.index // GRID_COLS
        df["grid_x"] = df.index % GRID_COLS

        logger.info(
            "DATA_LOADED",
            source=source,
            rows=len(df),
            grid=f"{GRID_COLS}x{len(df) // GRID_COLS}",
        )
        return df

    def _load_or_train_model(self) -> Pipeline:
        """Load cached joblib pipeline or train on 80% stratified split (seed=42) and save."""
        if self.model_path.exists():
            pipeline = joblib.load(self.model_path)
            logger.info("MODEL_LOADED_FROM_CACHE", path=str(self.model_path))
            return pipeline

        X = self.df[self.feature_cols]
        y = self.df["label"]

        # 80/20 stratified split matching safety_landing_model.ipynb
        X_train, _, y_train, _ = train_test_split(
            X, y, test_size=0.20, stratify=y, random_state=42
        )

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,
                    C=0.01,
                ),
            ),
        ])
        pipeline.fit(X_train, y_train)

        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, self.model_path)
        logger.info("MODEL_TRAINED_AND_CACHED", path=str(self.model_path))
        return pipeline

    def _generate_terrain_grid(self):
        """Predict safety probability across all 3,000 grid tiles and build 2D lookup arrays."""
        X = self.df[self.feature_cols]

        self.df_grid = self.df[["grid_x", "grid_y"]].copy()
        self.df_grid["safety_prob"] = self.model.predict_proba(X)[:, 1]
        if "confidence_score" in self.df.columns:
            self.df_grid["confidence_score"] = self.df["confidence_score"].values
        else:
            self.df_grid["confidence_score"] = 1.0

        # 2D matrices shaped (GRID_ROWS=50, GRID_COLS=60) for fast morphological/cluster ops
        num_rows = len(self.df_grid) // GRID_COLS
        self.prob_matrix = np.zeros((num_rows, GRID_COLS), dtype=float)
        self.conf_matrix = np.zeros((num_rows, GRID_COLS), dtype=float)

        for _, row in self.df_grid.iterrows():
            gx, gy = int(row["grid_x"]), int(row["grid_y"])
            self.prob_matrix[gy, gx] = row["safety_prob"]
            self.conf_matrix[gy, gx] = row["confidence_score"]

        preferred = int((self.df_grid["safety_prob"] > self.PREFERRED_PROB_THRESHOLD).sum())
        conditional = int(
            (
                (self.df_grid["safety_prob"] >= self.CONDITIONAL_PROB_THRESHOLD)
                & (self.df_grid["safety_prob"] <= self.PREFERRED_PROB_THRESHOLD)
            ).sum()
        )
        restricted = int((self.df_grid["safety_prob"] < self.CONDITIONAL_PROB_THRESHOLD).sum())

        logger.info(
            "TERRAIN_GRID_READY",
            total_tiles=len(self.df_grid),
            preferred_zones=preferred,
            conditional_zones=conditional,
            restricted_zones=restricted,
        )

    def _extract_clusters(
        self,
        binary_mask: np.ndarray,
        min_tiles: int,
        current_x: int,
        current_y: int,
        tier_name: str,
    ) -> List[Dict]:
        """
        4-connected component labeling via BFS.
        Filters out isolated tiles smaller than `min_tiles` and computes
        probability-weighted cluster centroids.
        """
        rows, cols = binary_mask.shape
        visited = np.zeros_like(binary_mask, dtype=bool)
        clusters: List[Dict] = []
        cluster_id = 0

        for r in range(rows):
            for c in range(cols):
                if binary_mask[r, c] and not visited[r, c]:
                    # Flood-fill (4-connectivity: North, South, East, West)
                    queue = deque([(r, c)])
                    visited[r, c] = True
                    cells: List[Tuple[int, int]] = []  # (x, y) = (c, r)

                    while queue:
                        cy, cx = queue.popleft()
                        cells.append((cx, cy))
                        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                            ny, nx = cy + dy, cx + dx
                            if 0 <= ny < rows and 0 <= nx < cols:
                                if binary_mask[ny, nx] and not visited[ny, nx]:
                                    visited[ny, nx] = True
                                    queue.append((ny, nx))

                    if len(cells) < min_tiles:
                        continue

                    cluster_id += 1
                    probs = np.array([self.prob_matrix[y, x] for x, y in cells])
                    confs = np.array([self.conf_matrix[y, x] for x, y in cells])
                    xs = np.array([x for x, _ in cells], dtype=float)
                    ys = np.array([y for _, y in cells], dtype=float)

                    # Probability-weighted centroid
                    weights = probs / probs.sum()
                    cx_float = float(np.sum(xs * weights))
                    cy_float = float(np.sum(ys * weights))

                    # Snap to the nearest actual tile inside the cluster so we never land outside an L-shape
                    interior_dists = np.hypot(xs - cx_float, ys - cy_float)
                    best_idx = int(np.argmin(interior_dists))
                    target_x, target_y = int(xs[best_idx]), int(ys[best_idx])

                    dist_from_drone = float(
                        np.hypot(target_x - current_x, target_y - current_y)
                    )

                    clusters.append({
                        "cluster_id": cluster_id,
                        "x": target_x,
                        "y": target_y,
                        "centroid_x": round(cx_float, 2),
                        "centroid_y": round(cy_float, 2),
                        "prob": float(self.prob_matrix[target_y, target_x]),
                        "mean_prob": float(np.mean(probs)),
                        "mean_conf": float(np.mean(confs)),
                        "area_tiles": len(cells),
                        "area_m2": len(cells) * (self.grid_resolution_m ** 2),
                        "dist_tiles": round(dist_from_drone, 2),
                        "tier": tier_name,
                    })

        # Sort clusters by Euclidean distance from drone (primary) and mean probability (tie-breaker)
        clusters.sort(key=lambda c: (c["dist_tiles"], -c["mean_prob"]))
        return clusters

    def find_candidate_clusters(
        self,
        current_grid_x: int,
        current_grid_y: int,
        max_distance_tiles: Optional[float] = None,
    ) -> List[Dict]:
        """
        Returns ranked contiguous landing clusters honoring:
          1. Preferred clusters (prob > 0.80, conf >= min_confidence, area >= min_cluster_tiles)
          2. Conditional clusters (prob >= 0.50, conf >= min_confidence, area >= min_cluster_tiles)
          3. Optional reachability radius filter (`max_distance_tiles`)
        """
        conf_ok = self.conf_matrix >= self.min_confidence
        above_floor = self.prob_matrix >= self.ABSOLUTE_NO_LAND_FLOOR

        # 1. Preferred contiguous clusters
        pref_mask = (self.prob_matrix > self.PREFERRED_PROB_THRESHOLD) & conf_ok & above_floor
        preferred_clusters = self._extract_clusters(
            pref_mask, self.min_cluster_tiles, current_grid_x, current_grid_y, "PREFERRED"
        )

        # 2. Conditional contiguous clusters
        cond_mask = (self.prob_matrix >= self.CONDITIONAL_PROB_THRESHOLD) & conf_ok & above_floor
        conditional_clusters = self._extract_clusters(
            cond_mask, self.min_cluster_tiles, current_grid_x, current_grid_y, "CONDITIONAL"
        )

        all_candidates = preferred_clusters + conditional_clusters
        if max_distance_tiles is not None:
            all_candidates = [
                c for c in all_candidates if c["dist_tiles"] <= max_distance_tiles
            ]
        return all_candidates

    def find_safest_zone(
        self,
        current_grid_x: int,
        current_grid_y: int,
        max_distance_tiles: Optional[float] = None,
        exclude_coords: Optional[List[Tuple[int, int]]] = None,
    ) -> Optional[Dict]:
        """
        Selects the nearest contiguous safe cluster (or falls back to secondary cluster).
        Never selects any tile with safety_prob < 0.30.
        """
        candidates = self.find_candidate_clusters(
            current_grid_x, current_grid_y, max_distance_tiles=max_distance_tiles
        )

        if exclude_coords:
            excluded_set = set(exclude_coords)
            candidates = [
                c for c in candidates if (c["x"], c["y"]) not in excluded_set
            ]

        if not candidates:
            logger.error(
                "NO_SAFE_CLUSTER",
                status="No contiguous cluster meets area, confidence, and safety thresholds. Aborting.",
            )
            return None

        best = candidates[0]
        log_fn = logger.info if best["tier"] == "PREFERRED" else logger.warning
        log_fn(
            "SAFE_CLUSTER_SELECTED",
            tier=best["tier"],
            target_grid=f"({best['x']}, {best['y']})",
            prob=round(best["prob"], 3),
            mean_prob=round(best["mean_prob"], 3),
            area_tiles=best["area_tiles"],
            area_m2=best["area_m2"],
            dist_tiles=best["dist_tiles"],
        )
        return best

    def grid_to_local_meters(
        self, target_x: int, target_y: int, current_x: int, current_y: int
    ) -> Tuple[float, float]:
        """Converts grid offset to local NED (North-East) meters for ArduPilot."""
        dx_grid = target_x - current_x
        dy_grid = target_y - current_y

        # Grid Y is North, Grid X is East
        north_m = float(dy_grid * self.grid_resolution_m)
        east_m = float(dx_grid * self.grid_resolution_m)

        return north_m, east_m