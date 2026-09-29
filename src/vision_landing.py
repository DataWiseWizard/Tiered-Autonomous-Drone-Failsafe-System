"""
vision_landing.py
Tier 3: Vision Landing Engine
Loads the safety model, evaluates a simulated terrain grid, 
and returns the safest landing coordinates.
"""
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import structlog

logger = structlog.get_logger()

DATA_URL = "https://docs.google.com/spreadsheets/d/1tCQf9YVzj8zET1bjTlettAV5WfyeNpo4EBEjo5H1Z9Y/export?format=csv"
GRID_COLS = 60  # From your notebook: 60 columns x 50 rows = 3000 tiles


class VisionLandingEngine:
    def __init__(self):
        logger.info("INIT_VISION_ENGINE", status="Fetching terrain data & training model...")
        self.df = self._load_and_anchor_data()
        self.feature_cols = [col for col in self.df.columns if col not in ['label', 'grid_x', 'grid_y']]
        self.model = self._train_model()
        self.grid_resolution_m = 5.0  # Each grid cell represents 5x5 meters
        self._generate_terrain_grid()

    def _load_and_anchor_data(self) -> pd.DataFrame:
        """Fetch CSV and add spatial anchoring (grid_x, grid_y) exactly as in the notebook."""
        df = pd.read_csv(DATA_URL)
        
        # Spatial anchoring from the notebook:
        # grid_y = row // 60, grid_x = row % 60
        df['grid_y'] = df.index // GRID_COLS
        df['grid_x'] = df.index % GRID_COLS
        
        logger.info("DATA_LOADED", rows=len(df), grid=f"{GRID_COLS}x{len(df)//GRID_COLS}")
        return df

    def _train_model(self):
        """Train the Logistic Regression pipeline exactly as in the notebook."""
        X = self.df[self.feature_cols]
        y = self.df['label']
        
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42, C=0.01))
        ])
        pipeline.fit(X, y)
        logger.info("MODEL_TRAINED", status="Logistic Regression ready")
        return pipeline

    def _generate_terrain_grid(self):
        """Predict safety probability across all grid tiles."""
        X = self.df[self.feature_cols]
        
        self.df_grid = self.df[['grid_x', 'grid_y']].copy()
        self.df_grid['safety_prob'] = self.model.predict_proba(X)[:, 1]
        
        # Log summary statistics
        preferred = (self.df_grid['safety_prob'] > 0.8).sum()
        conditional = ((self.df_grid['safety_prob'] >= 0.5) & (self.df_grid['safety_prob'] <= 0.8)).sum()
        restricted = (self.df_grid['safety_prob'] < 0.5).sum()
        
        logger.info("TERRAIN_GRID_READY",
                   total_tiles=len(self.df_grid),
                   preferred_zones=int(preferred),
                   conditional_zones=int(conditional),
                   restricted_zones=int(restricted))

    def find_safest_zone(self, current_grid_x: int, current_grid_y: int) -> dict:
        """
        Implements the Tier 3 Fallback Hierarchy from the notebook:
        1. Preferred zones: probability > 0.8 (closest to drone)
        2. Conditional zones: 0.5-0.8 (fallback)
        3. Absolute floor: never land < 0.3 (returns None → abort)
        """
        # 1. Target Preferred Zones (> 0.8)
        preferred = self.df_grid[self.df_grid['safety_prob'] > 0.8].copy()
        
        if not preferred.empty:
            preferred['dist'] = np.sqrt(
                (preferred['grid_x'] - current_grid_x)**2 + 
                (preferred['grid_y'] - current_grid_y)**2
            )
            target = preferred.loc[preferred['dist'].idxmin()]
            logger.info("TARGET_FOUND", tier="PREFERRED", 
                       prob=f"{target['safety_prob']:.3f}",
                       grid=f"({int(target['grid_x'])}, {int(target['grid_y'])})",
                       dist_tiles=f"{target['dist']:.1f}")
            return {
                "x": int(target['grid_x']), 
                "y": int(target['grid_y']), 
                "prob": target['safety_prob'], 
                "tier": "PREFERRED"
            }
            
        # 2. Fallback: Conditional Zones (0.5 - 0.8)
        conditional = self.df_grid[
            (self.df_grid['safety_prob'] >= 0.5) & 
            (self.df_grid['safety_prob'] <= 0.8)
        ].copy()
        
        if not conditional.empty:
            conditional['dist'] = np.sqrt(
                (conditional['grid_x'] - current_grid_x)**2 + 
                (conditional['grid_y'] - current_grid_y)**2
            )
            target = conditional.loc[conditional['dist'].idxmin()]
            logger.warning("TARGET_FALLBACK", tier="CONDITIONAL",
                          prob=f"{target['safety_prob']:.3f}",
                          grid=f"({int(target['grid_x'])}, {int(target['grid_y'])})",
                          dist_tiles=f"{target['dist']:.1f}")
            return {
                "x": int(target['grid_x']), 
                "y": int(target['grid_y']), 
                "prob": target['safety_prob'], 
                "tier": "CONDITIONAL"
            }
            
        # 3. Absolute Floor: No safe zone found
        logger.error("NO_SAFE_ZONE", status="All zones below safety threshold. Aborting landing.")
        return None

    def grid_to_local_meters(self, target_x: int, target_y: int, 
                              current_x: int, current_y: int) -> tuple:
        """Converts grid offset to local NED (North-East-Down) meters for ArduPilot."""
        dx_grid = target_x - current_x
        dy_grid = target_y - current_y
        
        # Grid Y is North, Grid X is East
        north_m = dy_grid * self.grid_resolution_m
        east_m = dx_grid * self.grid_resolution_m
        
        return north_m, east_m