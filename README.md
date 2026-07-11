# frost-trio

Poker44 (Bittensor SN126) bot-detection miner — **FrostTrioBlend**, version 1.0.

## Architecture
dart LightGBM(63) + depthwise XGBoost(d6) + deep CatBoost(d8); weighted-rank 40/35/25 on the tree view.

Pure gradient trio (no neural, no stacking). DART regularisation + depth diversity keeps the members decorrelated.

Members are fused by the recipe's blend mode; post-processing is strictly
rank-preserving (a monotone remap moves the deploy threshold to 0.5, and a
per-batch safety budget caps the positive-call fraction without reordering),
so AP and recall@FPR equal the model's own ranking quality.

## Feature surface
Features are computed ONLY from miner-visible behavioural fields (action
sequences, sizings relative to the big blind, structural aggregates). No hole
cards, board cards, hand outcomes, or player identifiers are used. Bet sizes are
quantized to the validator's exact bb-bucket grid and every per-hand scalar is
aggregated to the chunk with order statistics, so scoring is stable from
benchmark-size (30-40 hands) to live-size (80-105 hands) groups. This miner's
target FPR is **0.04** and its per-batch positive budget is **0.18**.

## Training data
Trained exclusively on the PUBLIC Poker44 benchmark releases
(`api.poker44.net/api/v1/benchmark`), plus pooled size-resamples of those same
public groups. No validator-only data is used. Hands are sanitized through
`prepare_hand_for_miner` at train time (train == serve).

## Auto-retrain
`model/autopilot.py` runs daily: it pulls any new benchmark release into the
shared cache, retrains a candidate into a staging dir, accepts it only if
walk-forward reward does not regress and FPR stays under the deploy ceiling,
then hot-swaps the artifact and restarts the miner (previous artifact backed up).

## Verification / transparency
Every response carries a model manifest: `repo_url` + `repo_commit` (this repo at
the serving commit), `implementation_files` + `implementation_sha256` (the exact
serving + training source, content-hashed), `artifact_filename` +
`artifact_sha256` (the trained-weights identity — weights are not distributed;
the hash pins them), and the training-data / private-data attestations.

## License
MIT
