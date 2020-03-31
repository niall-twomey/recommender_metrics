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


def average_precision(df, df_at_k, score_col, label_col, ranked_col):
    pos_group = df_at_k.loc[df_at_k[label_col] == 1]
    if len(pos_group) == 0:
        return 0.0  # TODO: check this default return value
    precisions = pos_group[label_col].cumsum() / pos_group[ranked_col]
    return precisions.mean()


def precision(df, df_at_k, score_col, label_col, ranked_col):
    precision_at_k = df_at_k[label_col].cumsum() / df_at_k[ranked_col]
    return precision_at_k.values[-1]


def recall(df, df_at_k, score_col, label_col, ranked_col):
    den = df[label_col].sum()
    if den == 0:
        return 1.0  # TODO: check this default return value
    recalls = df_at_k[label_col].cumsum() / den
    return recalls.values[-1]


def auroc(df, df_at_k, score_col, label_col, ranked_col):
    uniques = df_at_k[label_col].unique()
    if uniques.shape[0] == 1:
        return uniques[0]  # TODO: check this default return value
    return skl_metrics.roc_auc_score(
        y_true=df_at_k[label_col],
        y_score=df_at_k[score_col]
    )


def ndcg(df, df_at_k, score_col, label_col, ranked_col):
    if df_at_k.shape[0] <= 1:
        return 0  # TODO: check this default return value
    return skl_metrics.ndcg_score(
        y_true=df_at_k[label_col].values[None, :],
        y_score=df_at_k[score_col].values[None, :],
        k=df_at_k.shape[0],
    )


METRIC_FUNCTIONS = dict(
    mAP=average_precision,
    precision=precision,
    recall=recall,
    auroc=auroc,
    ndcg=ndcg,
)

DEFAULT_METRICS = [
    'mAP', 'precision', 'recall'
]
