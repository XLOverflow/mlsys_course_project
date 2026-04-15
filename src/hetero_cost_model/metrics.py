"""Evaluation metrics for the cost model.

- ``mape``           : mean absolute percentage error on predicted latency.
- ``spearman``       : rank correlation (strategy ordering quality).
- ``top_k_accuracy`` : fraction of the true top-k recovered by pred top-k.
- ``ndcg``           : NDCG@k with ``relevance = N - true_rank``.
"""
from __future__ import annotations

import math
from typing import List, Sequence


def mape(pred: Sequence[float], true: Sequence[float]) -> float:
    count = 0
    total = 0.0
    for p, t in zip(pred, true):
        if t == 0:
            continue
        total += abs(p - t) / abs(t)
        count += 1
    return total / count if count else float("nan")


def spearman(pred: Sequence[float], true: Sequence[float]) -> float:
    n = len(pred)
    if n < 2:
        return float("nan")
    rank_pred = _rank(pred)
    rank_true = _rank(true)
    mean_p = sum(rank_pred) / n
    mean_t = sum(rank_true) / n
    num = sum((a - mean_p) * (b - mean_t) for a, b in zip(rank_pred, rank_true))
    den = math.sqrt(
        sum((a - mean_p) ** 2 for a in rank_pred)
        * sum((b - mean_t) ** 2 for b in rank_true)
    )
    return num / den if den else float("nan")


def top_k_accuracy(pred: Sequence[float], true: Sequence[float], k: int = 1) -> float:
    top_true = sorted(range(len(true)), key=lambda i: true[i])[:k]
    top_pred = sorted(range(len(pred)), key=lambda i: pred[i])[:k]
    return len(set(top_true) & set(top_pred)) / k


def ndcg(pred: Sequence[float], true: Sequence[float], k: int = 5) -> float:
    """Rank-based NDCG@k. Lower latency means higher relevance."""
    n = len(true)
    if n == 0:
        return float("nan")

    true_order = sorted(range(n), key=lambda i: true[i])
    relevance = [0.0] * n
    for rank, idx in enumerate(true_order):
        relevance[idx] = float(n - rank)

    pred_order = sorted(range(n), key=lambda i: pred[i])[:k]
    dcg = sum(relevance[idx] / math.log2(rank + 2) for rank, idx in enumerate(pred_order))
    ideal = sorted(relevance, reverse=True)[:k]
    idcg = sum(v / math.log2(rank + 2) for rank, v in enumerate(ideal))
    return dcg / idcg if idcg else float("nan")


def _rank(xs: Sequence[float]) -> List[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg_rank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


__all__ = ["mape", "spearman", "top_k_accuracy", "ndcg"]
