"""frost-trio — FrostTrioBlend for the poker44 fleet.

Pure gradient trio (no neural, no stacking). DART regularisation + depth diversity keeps the members decorrelated.

Architecture: dart LightGBM(63) + depthwise XGBoost(d6) + deep CatBoost(d8); weighted-rank 40/35/25 on the tree view.
"""
import numpy as np

from fleet_engine import blend_parts


class FrostTrioBlend:
    """dart LightGBM(63) + depthwise XGBoost(d6) + deep CatBoost(d8); weighted-rank 40/35/25 on the tree view"""

    blend_mode = "rank"

    def __init__(self, members, cols, calibrator=None):
        self.members = list(members)          # [{"name","view","weight","est"}]
        self.cols = dict(cols)                # {view: [column names]}
        self.calibrator = calibrator          # optional isotonic post-calibrator

    def needed_views(self):
        return sorted({m["view"] for m in self.members})

    def score(self, view_mats):
        parts, weights = [], []
        for m in self.members:
            parts.append(m["est"].predict_proba(view_mats[m["view"]])[:, 1])
            weights.append(float(m["weight"]))
        p = blend_parts(parts, weights, self.blend_mode)
        if self.calibrator is not None:
            p = self.calibrator.predict(np.clip(np.asarray(p, dtype=float), 0.0, 1.0))
        return np.asarray(p, dtype=float)
