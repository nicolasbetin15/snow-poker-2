"""Shared numerics for the poker44 detection fleet (serving + training).

Every fleet miner embeds an identical copy of this module. The per-miner
*model* differences live in each repo's ``hg_model.py`` / ``infer.py`` /
``train_hg.py`` (member roster, blend, feature view, FPR target); this file
only holds the boundary-neutral primitives they all share:

  * ``rank01`` / ``blend_parts`` — calibration-free member fusion;
  * ``remap_to_threshold`` — monotone remap moving the deploy threshold to 0.5
    (the validator rounds predictions at 0.5) WITHOUT reordering scores, so AP
    and recall@FPR equal the model's own ranking quality;
  * ``batch_safety_budget`` — cap the fraction of positive calls per batch,
    again strictly rank-preserving, to protect humans on adversarial batches;
  * ``fpr_quantile_threshold`` — deploy threshold from the human score quantile
    (robust to cross-date shift, directly controls FPR);
  * ``mono_vector`` — sign-stable monotone constraints from walk-forward data;
  * ``build_matrix`` — align a feature view into a fixed column order.
"""
from __future__ import annotations

import warnings
from typing import Callable, Dict, List, Sequence

import numpy as np

# Cosmetic warnings from the sklearn<->lightgbm feature-name handoff and from
# spearmanr on constant feature columns (handled explicitly in mono_vector).
warnings.filterwarnings("ignore", message="X does not have valid feature names")
try:  # scipy>=1.9 exposes ConstantInputWarning
    from scipy.stats import ConstantInputWarning
    warnings.filterwarnings("ignore", category=ConstantInputWarning)
except Exception:  # pragma: no cover
    pass


# --------------------------------------------------------------------------- #
# member fusion
# --------------------------------------------------------------------------- #
def rank01(scores) -> np.ndarray:
    """Map scores to their in-batch rank in [0, 1] (ties broken stably)."""
    s = np.asarray(scores, dtype=float)
    if s.size <= 1:
        return np.zeros_like(s)
    return np.argsort(np.argsort(s, kind="stable"), kind="stable").astype(float) / (s.size - 1)


def blend_parts(parts: Sequence[np.ndarray], weights: Sequence[float], mode: str) -> np.ndarray:
    """Fuse member score vectors under one of four blend modes.

    ``rank``   weighted average of per-member in-batch ranks (scale-free);
    ``mean``   weighted average of member probabilities;
    ``logit``  weighted average in logit space then sigmoid (sharper);
    ``single`` pass the first member through unchanged (calibrated members).
    """
    ps = [np.asarray(p, dtype=float) for p in parts]
    w = np.asarray(weights, dtype=float)
    if w.sum() <= 0:
        w = np.ones_like(w)
    w = w / w.sum()
    if mode == "single":
        return ps[0]
    if mode == "rank":
        return sum(wi * rank01(p) for wi, p in zip(w, ps))
    if mode == "mean":
        return sum(wi * p for wi, p in zip(w, ps))
    if mode == "logit":
        eps = 1e-6
        z = 0.0
        for wi, p in zip(w, ps):
            pc = np.clip(p, eps, 1 - eps)
            z = z + wi * np.log(pc / (1 - pc))
        return 1.0 / (1.0 + np.exp(-z))
    raise ValueError(f"unknown blend mode {mode!r}")


# --------------------------------------------------------------------------- #
# rank-preserving post-processing (identical maths in every miner)
# --------------------------------------------------------------------------- #
def remap_to_threshold(p: np.ndarray, t: float) -> np.ndarray:
    """Monotone piecewise-linear remap so decision threshold ``t`` sits at 0.5."""
    t = float(min(max(t, 1e-6), 1 - 1e-6))
    out = np.where(p >= t, 0.5 + 0.5 * (p - t) / (1 - t), 0.5 * p / t)
    return np.clip(out, 0.0, 1.0)


def batch_safety_budget(scores: np.ndarray, max_frac: float) -> np.ndarray:
    """Cap the fraction of >=0.5 calls per batch WITHOUT changing the ranking.

    Only scores already past 0.5 are compressed, into the open interval between
    the highest sub-threshold score and 0.5, so global ordering is preserved and
    at least one confident call always survives (k >= 1)."""
    s = np.asarray(scores, dtype=float)
    n = s.size
    if n == 0 or max_frac >= 1.0:
        return s
    k = max(1, int(np.floor(max_frac * n)))
    positive = np.flatnonzero(s >= 0.5)
    if positive.size <= k:
        return s
    order = positive[np.argsort(-s[positive], kind="stable")]
    squeeze = order[k:]
    below = s[s < 0.5]
    lo = min(float(below.max()) if below.size else 0.45, 0.499)
    span = 0.5 - lo
    out = s.copy()
    m = squeeze.size
    for rank, idx in enumerate(squeeze):
        out[idx] = lo + span * (m - rank) / (m + 1.0)
    return np.clip(out, 0.0, 1.0)


def fpr_quantile_threshold(p_neg: np.ndarray, target_fpr: float) -> float:
    """Threshold = quantile of NEGATIVE (human) scores giving ~target FPR."""
    p_neg = np.asarray(p_neg, dtype=float)
    if p_neg.size == 0:
        return 0.5
    q = 1.0 - float(target_fpr)
    return float(np.quantile(p_neg, min(max(q, 0.0), 1.0)))


# --------------------------------------------------------------------------- #
# training-time helpers
# --------------------------------------------------------------------------- #
def mono_vector(X: np.ndarray, y: np.ndarray, dates: np.ndarray,
                *, min_dates: int = 5, min_abs: float = 0.05,
                min_agree: float = 0.7) -> List[int]:
    """Per-feature monotone sign kept only when sign-stable across dates."""
    from scipy.stats import spearmanr  # local import: training-only dependency
    ud = sorted(set(dates.tolist()))
    out: List[int] = []
    for j in range(X.shape[1]):
        sg: List[float] = []
        for d in ud:
            m = dates == d
            if m.sum() >= 8 and len(set(y[m].tolist())) > 1:
                r = spearmanr(X[m, j], y[m]).correlation
                if r is not None and not np.isnan(r):
                    sg.append(r)
        ok = (len(sg) >= min_dates and abs(float(np.mean(sg))) >= min_abs
              and float((np.sign(sg) == np.sign(np.mean(sg))).mean()) >= min_agree)
        out.append(int(np.sign(np.mean(sg))) if ok else 0)
    return out


def build_matrix(chunks, view_fn: Callable, cols: List[str] | None = None):
    """Featurize chunks through ``view_fn`` into a fixed column order."""
    feats = [view_fn(c) for c in chunks]
    if cols is None:
        keys = set()
        for d in feats:
            keys.update(d.keys())
        cols = sorted(keys)
    X = np.array([[float(d.get(c, 0.0)) for c in cols] for d in feats], dtype=float)
    return X, list(cols)


def atomic_write(payload_bytes: bytes, path: str) -> None:
    import os
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        fh.write(payload_bytes)
    os.replace(tmp, path)
