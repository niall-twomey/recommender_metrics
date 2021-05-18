from functools import lru_cache

import numpy as np
from sklearn import metrics as skl_metrics

__all__ = [
    "average_precision",
    "precision",
    "recall",
    "auroc",
    "ndcg",
    "METRIC_FUNCTIONS",
    "DEFAULT_METRICS",
]


@lru_cache(maxsize=128)
def ranks_at(k):
    return np.arange(1, 1 + k)


@lru_cache(maxsize=128)
def discounts_at(k):
    return np.log2(ranks_at(k) + 1)


def average_precision(labels: np.ndarray, scores: np.ndarray, k: int) -> float:
    labels_at_k = labels[:k]
    if not labels_at_k.any():
        return 0.0  # TODO: verify default value
    ranks = ranks_at(min(k, len(labels_at_k)))
    precisions = labels_at_k[labels_at_k].cumsum() / ranks[labels_at_k]
    # precisions = labels_at_k[labels_at_k].cumsum() / (np.flatnonzero(labels_at_k) + 1)
    return precisions.mean()


def precision(labels: np.ndarray, scores: np.ndarray, k: int) -> float:
    return labels[:k].mean()


def recall(labels: np.ndarray, scores: np.ndarray, k: int) -> float:
    denominator = labels.sum()
    if denominator == 0:
        return 1.0  # TODO: verify default value
    return labels[:k].sum() / denominator


def auroc(labels: np.ndarray, scores: np.ndarray, k: int) -> float:
    # Implementation based on: https://www.kaggle.com/c/microsoft-malware-prediction/discussion/76013
    labels_at_k = labels[:k]
    n_pos = sum(labels_at_k)
    n_neg = len(labels_at_k) - n_pos
    if not min(n_pos, n_neg):
        return float(n_neg == 0)  # TODO: check this default return value
    ranks = ranks_at(min(k, len(labels_at_k)))
    return 1 - (np.sum(ranks[labels_at_k == 1]) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    # return skl_metrics.roc_auc_score(y_true=labels[:k], y_score=scores[:k])


def ndcg(labels: np.ndarray, scores: np.ndarray, k: int) -> float:
    # NOTE: this function is slow to compute
    if labels[:k].shape[0] <= 1:
        return 0.0  # TODO: check this default return value
    return 0.0  # skl_metrics.ndcg_score(y_true=labels[None, :k], y_score=scores[None, :k], k=k)


METRIC_FUNCTIONS = dict(mAP=average_precision, precision=precision, recall=recall, auroc=auroc, ndcg=ndcg)

DEFAULT_METRICS = [
    "mAP",
    "precision",
    "recall",
    # "auroc",
    # "ndcg",
]
