import numpy as np
from sklearn import metrics as skl_metrics

__all__ = [
    'average_precision',
    'precision',
    'recall',
    'auroc',
    'ndcg',
    'METRIC_FUNCTIONS',
    'DEFAULT_METRICS',
]


def average_precision(scores, labels, ranks, k):
    labels_at_k, ranks_at_k = labels[:k], ranks[:k]
    if not labels_at_k.any():
        return 0.0  # TODO: verify default value
    precisions = labels_at_k[labels_at_k].cumsum() / ranks_at_k[labels_at_k]
    return precisions.mean()


def precision(scores, labels, ranks, k):
    labels_at_k, ranks_at_k = labels[:k], ranks[:k]
    return labels_at_k.sum() / ranks_at_k[-1]


def recall(scores, labels, ranks, k):
    denominator = labels.sum()
    if denominator == 0:
        return 1.0  # TODO: verify default value
    labels_at_k, scores_at_k, ranks_at_k = labels[:k], scores[:k], ranks[:k]
    return labels_at_k.sum() / denominator


def auroc(scores, labels, ranks, k):
    uniques = np.unique(labels)
    if uniques.shape[0] == 1:
        return uniques[0]  # TODO: check this default return value
    return skl_metrics.roc_auc_score(
        y_true=labels[:k],
        y_score=scores[:k]
    )


def ndcg(scores, labels, ranks, k):
    if labels.shape[0] <= 1:
        return 0  # TODO: check this default return value
    return skl_metrics.ndcg_score(
        y_true=labels[:k],
        y_score=scores[:k],
        k=k,
    )


METRIC_FUNCTIONS = dict(
    mAP=average_precision,
    precision=precision,
    recall=recall,
    auroc=auroc,
    ndcg=ndcg,
)

DEFAULT_METRICS = [
    'mAP',
    'precision',
    'recall',
]

