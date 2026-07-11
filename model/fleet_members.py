"""Per-recipe member rosters for the poker44 fleet (training only; gitignored-safe).

``build_members(kind, mono, seed, nj)`` returns, for one of the 12 fleet
architectures, a list of member dicts::

    {"name": str, "view": "tree"|"wide"|"ngram", "weight": float, "est": estimator}

plus the blend mode and whether an isotonic post-calibrator is fitted on the
walk-forward out-of-fold scores. Fresh estimators are built on every call so
each walk-forward fold refits cleanly. ``mono`` is the sign-stable monotone
vector aligned to the ``tree`` view columns (only the monotone kinds use it).
"""
from __future__ import annotations

from typing import List

import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from sklearn.ensemble import (ExtraTreesClassifier, HistGradientBoostingClassifier,
                              RandomForestClassifier, StackingClassifier,
                              VotingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def _lgb(seed, nj, **kw):
    params = dict(n_estimators=500, learning_rate=0.03, num_leaves=63,
                  subsample=0.9, subsample_freq=1, colsample_bytree=0.9,
                  reg_lambda=1.0, n_jobs=nj, random_state=seed, verbose=-1)
    params.update(kw)
    return lgb.LGBMClassifier(**params)


def _xgb(seed, nj, **kw):
    params = dict(n_estimators=500, learning_rate=0.04, tree_method="hist",
                  n_jobs=nj, random_state=seed, eval_metric="logloss")
    params.update(kw)
    return xgb.XGBClassifier(**params)


def _cat(seed, nj, **kw):
    params = dict(iterations=500, learning_rate=0.03, depth=6, verbose=0,
                  thread_count=nj, random_seed=seed, allow_writing_files=False)
    params.update(kw)
    return cb.CatBoostClassifier(**params)


def _mlp(seed, hidden, pca, nj):
    steps = [("s", StandardScaler())]
    if pca:
        steps.append(("p", PCA(pca, random_state=seed)))
    steps.append(("m", MLPClassifier(hidden, alpha=2.0, max_iter=600,
                                     early_stopping=True, validation_fraction=0.15,
                                     n_iter_no_change=15, random_state=seed)))
    return Pipeline(steps)


def build_members(kind: str, mono: List[int], seed: int, nj: int):
    """Return (members, blend_mode, use_calibrator) for a fleet architecture."""
    mono = list(int(c) for c in (mono or []))

    if kind == "stack":  # GlacierStack — leaf-wise stack + monotone LGBM + PCA-MLP
        stack = StackingClassifier(
            [("lgb", _lgb(seed, nj, num_leaves=111, n_estimators=550, learning_rate=0.02)),
             ("xgl", _xgb(seed + 1, nj, grow_policy="lossguide", max_leaves=64,
                          max_depth=0, n_estimators=550, learning_rate=0.03)),
             ("cat", _cat(seed + 2, nj, depth=7, iterations=650, learning_rate=0.025)),
             ("rf", RandomForestClassifier(n_estimators=500, max_depth=18, n_jobs=nj,
                                           random_state=seed + 3,
                                           class_weight="balanced_subsample"))],
            final_estimator=LogisticRegression(C=0.5, max_iter=1000), cv=4, n_jobs=1)
        monov = VotingClassifier([(f"l{i}", _lgb(seed + 10 + i, nj, num_leaves=31,
                                  n_estimators=500, monotone_constraints=mono))
                                  for i in range(3)], voting="soft", n_jobs=1)
        mlp = VotingClassifier([(f"m{i}", _mlp(seed + 20 + i, (80,), 56, nj))
                                for i in range(3)], voting="soft", n_jobs=1)
        return ([{"name": "stack4", "view": "tree", "weight": 0.35, "est": stack},
                 {"name": "mono", "view": "tree", "weight": 0.30, "est": monov},
                 {"name": "pca_mlp", "view": "wide", "weight": 0.35, "est": mlp}],
                "rank", False)

    if kind == "trio":  # FrostTrio — dart LGBM + depthwise XGB + deep CatBoost
        return ([{"name": "lgb_dart", "view": "tree", "weight": 0.40,
                  "est": _lgb(seed, nj, boosting_type="dart", num_leaves=63,
                              n_estimators=400, learning_rate=0.05, drop_rate=0.1)},
                 {"name": "xgb_depth", "view": "tree", "weight": 0.35,
                  "est": _xgb(seed + 1, nj, grow_policy="depthwise", max_depth=6,
                              n_estimators=500, subsample=0.9, colsample_bytree=0.9)},
                 {"name": "cat_deep", "view": "tree", "weight": 0.25,
                  "est": _cat(seed + 2, nj, depth=8, iterations=600, learning_rate=0.03)}],
                "rank", False)

    if kind == "mono":  # BlizzardMono — three monotone LGBMs, human-safety tuned
        return ([{"name": "mono_a", "view": "tree", "weight": 0.5,
                  "est": _lgb(seed, nj, num_leaves=31, n_estimators=500,
                              monotone_constraints=mono)},
                 {"name": "mono_b", "view": "tree", "weight": 0.3,
                  "est": _lgb(seed + 1, nj, num_leaves=15, n_estimators=600,
                              learning_rate=0.025, monotone_constraints=mono)},
                 {"name": "mono_c", "view": "tree", "weight": 0.2,
                  "est": _lgb(seed + 2, nj, num_leaves=63, n_estimators=450,
                              learning_rate=0.035, monotone_constraints=mono)}],
                "mean", False)

    if kind == "forest":  # TundraForest — ExtraTrees + RandomForest + HistGB vote
        return ([{"name": "extratrees", "view": "tree", "weight": 0.34,
                  "est": ExtraTreesClassifier(n_estimators=600, n_jobs=nj,
                                              random_state=seed,
                                              class_weight="balanced_subsample")},
                 {"name": "rf", "view": "tree", "weight": 0.33,
                  "est": RandomForestClassifier(n_estimators=600, max_depth=20, n_jobs=nj,
                                                random_state=seed + 1,
                                                class_weight="balanced_subsample")},
                 {"name": "histgb", "view": "tree", "weight": 0.33,
                  "est": HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05,
                                                        max_iter=400, l2_regularization=1.0,
                                                        random_state=seed + 2)}],
                "mean", False)

    if kind == "neural":  # AuroraNeuro — 5-seed PCA64->MLP(128,64) + LGBM
        mlp = VotingClassifier([(f"m{i}", _mlp(seed + i, (128, 64), 64, nj))
                                for i in range(5)], voting="soft", n_jobs=1)
        return ([{"name": "mlp5", "view": "wide", "weight": 0.6, "est": mlp},
                 {"name": "lgb", "view": "tree", "weight": 0.4,
                  "est": _lgb(seed + 30, nj, num_leaves=63, n_estimators=500)}],
                "rank", False)

    if kind == "calib":  # SolsticeCalib — stacked trio + isotonic calibration
        stack = StackingClassifier(
            [("lgb", _lgb(seed, nj, num_leaves=63, n_estimators=500)),
             ("xgb", _xgb(seed + 1, nj, grow_policy="depthwise", max_depth=6, n_estimators=500)),
             ("cat", _cat(seed + 2, nj, depth=6, iterations=500))],
            final_estimator=LogisticRegression(C=0.7, max_iter=1000), cv=4, n_jobs=1)
        return ([{"name": "stack3", "view": "tree", "weight": 1.0, "est": stack}],
                "single", True)

    if kind == "fusion":  # IcicleFusion — 5-way rank fusion of diverse learners
        return ([{"name": "lgb", "view": "tree", "weight": 0.2,
                  "est": _lgb(seed, nj, num_leaves=63, n_estimators=450)},
                 {"name": "xgb", "view": "tree", "weight": 0.2,
                  "est": _xgb(seed + 1, nj, grow_policy="depthwise", max_depth=6, n_estimators=450)},
                 {"name": "cat", "view": "tree", "weight": 0.2,
                  "est": _cat(seed + 2, nj, depth=6, iterations=450)},
                 {"name": "extratrees", "view": "tree", "weight": 0.2,
                  "est": ExtraTreesClassifier(n_estimators=500, n_jobs=nj, random_state=seed + 3,
                                              class_weight="balanced_subsample")},
                 {"name": "histgb", "view": "tree", "weight": 0.2,
                  "est": HistGradientBoostingClassifier(max_depth=4, learning_rate=0.06,
                                                        max_iter=350, random_state=seed + 4)}],
                "rank", False)

    if kind == "seq":  # DriftSeq — order-aware n-gram GBM trio
        return ([{"name": "lgb_ngram", "view": "ngram", "weight": 0.4,
                  "est": _lgb(seed, nj, num_leaves=63, n_estimators=500)},
                 {"name": "xgb_ngram", "view": "ngram", "weight": 0.3,
                  "est": _xgb(seed + 1, nj, grow_policy="depthwise", max_depth=6, n_estimators=500)},
                 {"name": "cat_ngram", "view": "ngram", "weight": 0.3,
                  "est": _cat(seed + 2, nj, depth=6, iterations=500)}],
                "rank", False)

    if kind == "widedeep":  # SummitWideDeep — wide MLP + wide LGBM
        mlp = VotingClassifier([(f"m{i}", _mlp(seed + i, (100, 50), 0, nj))
                                for i in range(3)], voting="soft", n_jobs=1)
        return ([{"name": "mlp_wide", "view": "wide", "weight": 0.5, "est": mlp},
                 {"name": "lgb_wide", "view": "wide", "weight": 0.5,
                  "est": _lgb(seed + 30, nj, num_leaves=95, n_estimators=500, learning_rate=0.025)}],
                "mean", False)

    if kind == "quantile":  # HarborQuantile — one deep LGBM + isotonic calibration
        return ([{"name": "lgb_deep", "view": "tree", "weight": 1.0,
                  "est": _lgb(seed, nj, num_leaves=255, n_estimators=900, learning_rate=0.02,
                              subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
                              min_child_samples=30)}],
                "single", True)

    if kind == "balanced":  # MeridianBalanced — balanced RF + reweighted XGB
        return ([{"name": "rf_bal", "view": "tree", "weight": 0.5,
                  "est": RandomForestClassifier(n_estimators=600, max_depth=18, n_jobs=nj,
                                                random_state=seed, class_weight="balanced")},
                 {"name": "xgb_rw", "view": "tree", "weight": 0.5,
                  "est": _xgb(seed + 1, nj, grow_policy="depthwise", max_depth=6,
                              n_estimators=500, max_delta_step=1)}],
                "mean", False)

    if kind == "deepcat":  # CascadeDeepCat — deep CatBoost + LGBM
        return ([{"name": "cat_deep", "view": "tree", "weight": 0.6,
                  "est": _cat(seed, nj, depth=10, iterations=700, learning_rate=0.03,
                              l2_leaf_reg=3.0)},
                 {"name": "lgb", "view": "tree", "weight": 0.4,
                  "est": _lgb(seed + 1, nj, num_leaves=63, n_estimators=500)}],
                "mean", False)

    raise ValueError(f"unknown fleet kind {kind!r}")
