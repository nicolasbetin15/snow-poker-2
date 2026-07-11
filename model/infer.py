"""Self-contained inference for frost-trio (FrostTrioBlend).

Loads the trained artifact, builds only the feature view(s) this model uses
(identical code paths to training), blends member scores, then applies the
rank-preserving threshold remap plus a per-batch human-safety budget. Ranking
(hence AP / recall@FPR) is never altered by post-processing.
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
from typing import Any, Dict, List

import numpy as np

from fleet_engine import batch_safety_budget, build_matrix, remap_to_threshold
from hg_features import VIEWS
from hg_model import FrostTrioBlend  # noqa: F401  (required to unpickle the artifact)

_ART_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
_ARTIFACT = os.environ.get("POKER44_ARTIFACT", "frost_trio_v10.pkl")
_MAX_POS_FRAC = float(os.environ.get("POKER44_MAX_POS_FRAC", "0.18"))


def artifact_path() -> str:
    return os.path.join(_ART_DIR, _ARTIFACT)


def artifact_sha256() -> str:
    digest = hashlib.sha256()
    with open(artifact_path(), "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class ServingModel:
    def __init__(self, art_dir: str = _ART_DIR):
        with open(os.path.join(art_dir, _ARTIFACT), "rb") as fh:
            self.ens = pickle.load(fh)
        with open(os.path.join(art_dir, "meta.json")) as fh:
            self.meta = json.load(fh)
        self.threshold: float = float(self.meta["deploy_threshold"])
        self.artifact_name: str = _ARTIFACT

    def score_chunks(self, chunks: List[List[Dict[str, Any]]]) -> List[float]:
        if not chunks:
            return []
        mats = {v: build_matrix(chunks, VIEWS[v], self.ens.cols[v])[0]
                for v in self.ens.needed_views()}
        p = self.ens.score(mats)
        scores = remap_to_threshold(np.asarray(p, dtype=float), self.threshold)
        scores = batch_safety_budget(scores, _MAX_POS_FRAC)
        return [0.1 if not chunk else round(float(s), 6)
                for chunk, s in zip(chunks, scores)]


_SINGLETON = None


def get_model() -> ServingModel:
    global _SINGLETON
    if _SINGLETON is None:
        _SINGLETON = ServingModel()
    return _SINGLETON
