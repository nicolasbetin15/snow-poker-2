#!/usr/bin/env python3
"""Training pipeline for frost-trio (FrostTrioBlend) — publishes the recipe, not the weights.

Honest walk-forward validation (train strictly-past dates -> test the next unseen
date, REAL rows only). Live-robustness via pooled augmentation up to validator
group size. Deploy threshold from the human-score quantile at the target FPR.
The artifact is written atomically so a serving miner never sees a half file.
"""
from __future__ import annotations

import json
import os
import pickle
import random
import sys
import time

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
for _p in (HERE, REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _load_env(path):
    try:
        with open(path) as fh:
            for raw in fh:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                value = value.split(" #", 1)[0].strip()
                if len(value) >= 2 and value[0] == value[-1] and value[0] in (chr(34), chr(39)):
                    value = value[1:-1]
                os.environ.setdefault(key.strip(), value)
    except FileNotFoundError:
        pass


_load_env(os.path.join(REPO, ".env"))

from poker44.validator.payload_view import prepare_hand_for_miner  # noqa: E402
from dataset import load_examples  # noqa: E402
from reward_fn import reward  # noqa: E402
from hg_features import VIEWS  # noqa: E402
from hg_model import FrostTrioBlend  # noqa: E402
from fleet_engine import (atomic_write, blend_parts, build_matrix,  # noqa: E402
                          fpr_quantile_threshold, mono_vector)
from fleet_members import build_members  # noqa: E402

RECIPE = {"model_name": "frost-trio", "model_class": "FrostTrioBlend",
          "kind": "trio", "version": "1.0",
          "framework": "dart LightGBM(63) + depthwise XGBoost(d6) + deep CatBoost(d8); weighted-rank 40/35/25 on the tree view", "note": "Pure gradient trio (no neural, no stacking). DART regularisation + depth diversity keeps the members decorrelated."}
ART = os.environ.get("POKER44_ART_DIR", "").strip() or os.path.join(HERE, "artifacts")
ARTIFACT = os.environ.get("POKER44_ARTIFACT", "frost_trio_v10.pkl")
TARGET_FPR = float(os.environ.get("POKER44_TARGET_FPR", "0.04"))
NJ = int(os.environ.get("POKER44_TRAIN_JOBS", "4"))
WF = int(os.environ.get("POKER44_WF_POINTS", "3"))
SEED = 2202
POOL_RANGE = (88, 104)
POOL_PER_DATE = 3


def sanitize(hands):
    out = []
    for h in hands:
        try:
            out.append(prepare_hand_for_miner(h))
        except Exception:
            out.append(h)
    return out


def augment(san, y, dates):
    """Pooled (live-size) resamples of PUBLIC benchmark groups only."""
    rng = random.Random(SEED)
    aug_chunks, aug_y, aug_dates = [], [], []
    by_key = {}
    for i, (d, lab) in enumerate(zip(dates, y)):
        by_key.setdefault((d, int(lab)), []).append(i)
    for (d, lab), idxs in sorted(by_key.items()):
        if len(idxs) < 2:
            continue
        for _ in range(POOL_PER_DATE):
            target = rng.randint(*POOL_RANGE)
            pool, used = [], 0
            for i in rng.sample(idxs, len(idxs)):
                pool.extend(san[i]); used += 1
                if len(pool) >= target:
                    break
            if used >= 2 and len(pool) >= POOL_RANGE[0]:
                aug_chunks.append(pool[:target]); aug_y.append(lab); aug_dates.append(d)
    return aug_chunks, np.asarray(aug_y, dtype=int), np.asarray(aug_dates)


def _fit_members(spec, mats, mask, y):
    fitted = []
    for m in spec:
        m["est"].fit(mats[m["view"]][mask], y[mask])
        fitted.append({"name": m["name"], "view": m["view"],
                       "weight": m["weight"], "est": m["est"]})
    return fitted


def _predict_blend(fitted, blend_mode, mats, mask):
    parts = [m["est"].predict_proba(mats[m["view"]][mask])[:, 1] for m in fitted]
    weights = [m["weight"] for m in fitted]
    return blend_parts(parts, weights, blend_mode)


if __name__ == "__main__":
    os.makedirs(ART, exist_ok=True)
    t0 = time.time()
    ex = load_examples()
    if not ex:
        print("no training data - check POKER44_TRAIN_DATA_DIR", file=sys.stderr)
        sys.exit(2)
    san = [sanitize(e.hands) for e in ex]
    y = np.asarray([e.label for e in ex], dtype=int)
    dates = np.asarray([e.source_date for e in ex])
    aug_chunks, aug_y, aug_dates = augment(san, y, dates)
    all_chunks = san + aug_chunks
    ally = np.concatenate([y, aug_y]) if len(aug_y) else y
    alldates = np.concatenate([dates, aug_dates]) if len(aug_dates) else dates
    is_real = np.zeros(len(all_chunks), dtype=bool); is_real[:len(san)] = True

    probe, blend_mode, use_cal = build_members(RECIPE["kind"], [], SEED, NJ)
    needed = sorted({m["view"] for m in probe})
    mats, cols = {}, {}
    for v in needed:
        mats[v], cols[v] = build_matrix(all_chunks, VIEWS[v])
    use_mono = RECIPE["kind"] in ("stack", "mono")
    mono_full = (mono_vector(mats["tree"][is_real], y, dates)
                 if use_mono and "tree" in mats else [])
    print(RECIPE["model_name"] + ": %d real + %d aug | views=%s | %d monotone (%.0fs)"
          % (len(y), len(aug_y), needed, sum(1 for c in mono_full if c), time.time() - t0),
          flush=True)

    ud = sorted(set(dates.tolist()))
    oof = np.full(len(y), np.nan)
    for td in ud[-WF:]:
        tr = alldates < td
        te_real = dates == td
        te = np.zeros(len(all_chunks), dtype=bool); te[:len(san)] = te_real
        if tr.sum() < 60 or len(set(ally[tr].tolist())) < 2 or not te.any():
            continue
        mono_tr = (mono_vector(mats["tree"][tr & is_real], ally[tr & is_real],
                               alldates[tr & is_real])
                   if use_mono and "tree" in mats else [])
        spec, bmode, _ = build_members(RECIPE["kind"], mono_tr, SEED, NJ)
        fitted = _fit_members(spec, mats, tr, ally)
        oof[te_real] = _predict_blend(fitted, bmode, mats, te)
        print("  wf %s (%.0fs)" % (td, time.time() - t0), flush=True)

    m = ~np.isnan(oof)
    if not m.any():
        print("walk-forward produced no scores; refusing to deploy blind", file=sys.stderr)
        sys.exit(3)
    calibrator = None
    oof_use = oof.copy()
    if use_cal:
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(oof[m], y[m])
        oof_use[m] = calibrator.predict(oof[m])
    cv_ap = float(average_precision_score(y[m], oof[m]))
    rew, res = reward(oof_use[m], y[m])
    deploy_t = fpr_quantile_threshold(oof_use[m][y[m] == 0], TARGET_FPR)
    print("WALK-FORWARD[%dd]: cv_ap=%.4f reward=%.4f recall@fpr=%.3f fpr=%.4f (%.0fs)"
          % (WF, cv_ap, rew, res["bot_recall"], res["fpr"], time.time() - t0), flush=True)

    spec, bmode, _ = build_members(RECIPE["kind"], mono_full, SEED, NJ)
    allmask = np.ones(len(all_chunks), dtype=bool)
    fitted = _fit_members(spec, mats, allmask, ally)
    ens = FrostTrioBlend(fitted, cols, calibrator=calibrator)

    meta = {
        "model_name": RECIPE["model_name"], "model_class": RECIPE["model_class"],
        "family": "poker44-fleet", "kind": RECIPE["kind"], "model": RECIPE["framework"],
        "note": RECIPE["note"], "feature_version": "fleet.v1", "views": needed,
        "trained_on": "sanitized (prepare_hand_for_miner; train == serve)",
        "deploy_threshold": float(deploy_t), "target_fpr": TARGET_FPR,
        "blend_mode": bmode, "calibrated": bool(use_cal), "seed": SEED,
        "cv_ap": cv_ap, "cv_reward": float(rew), "cv_recall": float(res["bot_recall"]),
        "cv_fpr": float(res["fpr"]),
        "validation": "walk-forward over the last %d dates (train past -> test next unseen date)" % WF,
        "reward_formula": "0.75*AP + 0.25*recall@fpr<=0.05 (official 2026-06-26)",
        "n_train_real": int(len(y)), "n_train_aug": int(len(aug_y)),
        "augmentation": {"pool_range": list(POOL_RANGE), "pool_per_date": POOL_PER_DATE},
        "n_features": {v: int(mats[v].shape[1]) for v in needed},
        "n_monotone": int(sum(1 for c in mono_full if c)), "n_dates": int(len(ud)),
        "benchmark_releases": sorted(set(dates.tolist())), "artifact": ARTIFACT,
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    atomic_write(pickle.dumps(ens), os.path.join(ART, ARTIFACT))
    atomic_write(json.dumps(meta, indent=2).encode("utf-8"), os.path.join(ART, "meta.json"))
    print("saved %s + meta.json | cv_ap=%.4f cv_reward=%.4f" % (ARTIFACT, cv_ap, rew), flush=True)
