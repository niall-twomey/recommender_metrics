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


def average_precision(scores: np.ndarray, labels: np.ndarray, ranks: np.ndarray, k: int) -> float:
    labels_at_k = labels[:k]
    if not labels_at_k.any():
        return 0.0  # TODO: verify default value
    precisions = labels_at_k[labels_at_k].cumsum() / ranks[:k][labels_at_k]
    return precisions.mean()


def precision(scores: np.ndarray, labels: np.ndarray, ranks: np.ndarray, k: int) -> float:
    return labels[:k].sum() / ranks[:k][-1]


def recall(scores: np.ndarray, labels: np.ndarray, ranks: np.ndarray, k: int) -> float:
    denominator = labels.sum()
    if denominator == 0:
        return 1.0  # TODO: verify default value
    return labels[:k].sum() / denominator


def auroc(scores: np.ndarray, labels: np.ndarray, ranks: np.ndarray, k: int) -> float:
    # NOTE: this function is slow to compute
    uniques = np.unique(labels[:k])
    if uniques.shape[0] == 1:
        return float(next(iter(uniques)))  # TODO: check this default return value
    return skl_metrics.roc_auc_score(y_true=labels[:k], y_score=scores[:k])


def ndcg(scores: np.ndarray, labels: np.ndarray, ranks: np.ndarray, k: int) -> float:
    # NOTE: this function is slow to compute
    if labels[:k].shape[0] <= 1:
        return 0  # TODO: check this default return value
    return skl_metrics.ndcg_score(y_true=labels[None, :k], y_score=scores[None, :k], k=k)


METRIC_FUNCTIONS = dict(mAP=average_precision, precision=precision, recall=recall, auroc=auroc, ndcg=ndcg)

DEFAULT_METRICS = [
    "mAP",
    "precision",
    "recall",
    # 'auroc',
    # 'ndcg',
]
