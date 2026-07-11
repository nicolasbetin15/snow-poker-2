"""Feature views for the poker44 fleet (published; train == serve exactly).

Three views are available; each miner's model consumes only the view(s) its
recipe declares, so the served feature surface differs across the fleet:

  ``tree``  per-chunk behavioural aggregates + bucket/entropy fingerprints from
            the base library, plus chunk-size descriptors (raw and log) so the
            model learns the group-size axis (live groups run larger than the
            benchmark groups).
  ``wide``  ``tree`` merged with the v2 order-statistic aggregates into one
            deduplicated dictionary (neural / wide members consume this union).
  ``ngram`` ``tree`` plus order-aware action n-gram regularity descriptors that
            expose bots replaying near-identical sub-sequences (hero/size
            invariant, so they survive the validator's action-window sampling).

All statistics use only miner-visible behavioural fields — no hole cards, board
cards, hand outcomes, or player identities.
"""
import math
from collections import Counter

from features_v2 import extract_features_v2
from poker44_ml.features import chunk_features

_MEANINGFUL = ("check", "call", "bet", "raise", "fold")


def _entropy(counts):
    tot = sum(counts)
    if tot <= 0:
        return 0.0
    ps = [c / tot for c in counts if c > 0]
    if len(ps) <= 1:
        return 0.0
    ent = -sum(p * math.log(p) for p in ps)
    return ent / math.log(len(ps))


def tree_view(chunk):
    hands = chunk or []
    d = chunk_features(hands)
    n = float(len(hands))
    d["hand_count"] = n
    d["hand_count_log"] = math.log1p(n)
    return d


def wide_view(chunk):
    d = dict(extract_features_v2(chunk or []))
    d.update(tree_view(chunk))
    return d


def _hand_action_seq(hand):
    seq = []
    for a in (hand.get("actions") or []):
        at = str((a or {}).get("action_type") or "").strip().lower()
        if at in _MEANINGFUL:
            seq.append(at)
    return seq


def ngram_view(chunk):
    """tree_view + order-aware action n-gram regularity across the chunk."""
    d = tree_view(chunk)
    hands = chunk or []
    bigrams, trigrams = [], []
    for h in hands:
        seq = _hand_action_seq(h)
        bigrams.extend(tuple(seq[i:i + 2]) for i in range(len(seq) - 1))
        trigrams.extend(tuple(seq[i:i + 3]) for i in range(len(seq) - 2))
    nb, nt = len(bigrams), len(trigrams)
    cb, ct = Counter(bigrams), Counter(trigrams)
    d["ngram_bi_entropy"] = _entropy(list(cb.values()))
    d["ngram_tri_entropy"] = _entropy(list(ct.values()))
    d["ngram_bi_top_share"] = (max(cb.values()) / nb) if nb else 0.0
    d["ngram_tri_top_share"] = (max(ct.values()) / nt) if nt else 0.0
    d["ngram_bi_unique_share"] = (len(cb) / nb) if nb else 0.0
    d["ngram_tri_unique_share"] = (len(ct) / nt) if nt else 0.0
    # fraction of hands whose full meaningful-action sequence repeats elsewhere
    seqs = [tuple(_hand_action_seq(h)) for h in hands]
    cs = Counter(seqs)
    d["ngram_repeat_hand_share"] = (
        sum(1 for s in seqs if cs[s] > 1) / len(seqs)) if seqs else 0.0
    return d


VIEWS = {"tree": tree_view, "wide": wide_view, "ngram": ngram_view}
