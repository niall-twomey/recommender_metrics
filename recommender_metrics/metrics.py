import pandas as pd
from sklearn import metrics

from recommender_metrics.utils import rank_dataframe, verbose_iterator

__all__ = [
    'calculate_metrics_from_dataframe',
    'calculate_metrics',
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
    return metrics.roc_auc_score(
        y_true=df_at_k[label_col],
        y_score=df_at_k[score_col]
    )


def ndcg(df, df_at_k, score_col, label_col, ranked_col):
    if df_at_k.shape[0] <= 1:
        return 0  # TODO: check this default return value
    return metrics.ndcg_score(
        y_true=df_at_k[label_col].values[None, :],
        y_score=df_at_k[score_col].values[None, :],
        k=df_at_k.shape[0],
    )


METRIC_FUNCTIONS = dict(
    mAP=average_precision,
    precison=precision,
    recall=recall,
    auroc=auroc,
    ndcg=ndcg,
)

DEFAULT_METRICS = [
    'mAP', 'precison', 'recall'
]


def validate_k_list(k_list):
    if k_list is None:
        return [1, 5, 10, 20]
    elif isinstance(k_list, int):
        assert k_list > 0
        return [k_list]
    elif isinstance(k_list, list):
        assert all(map(lambda kk: isinstance(kk, int) and kk > 0, k_list))
        return k_list
    raise ValueError


def validate_metrics(metrics):
    if metrics is None:
        metrics = DEFAULT_METRICS.copy()
    if isinstance(metrics, list) and all(map(lambda mm: isinstance(mm, str), metrics)):
        metrics = {mm: METRIC_FUNCTIONS[mm] for mm in metrics}
    assert isinstance(metrics, dict)
    if not all(map(callable, metrics.values())):
        raise TypeError(f'All metrics passed into this function must be callable')
    return metrics


def calculate_metrics_from_dataframe(
        df,
        k_list=None,
        ascending=False,
        group_col='group_id',
        label_col='label',
        score_col='score',
        metrics=None,
        verbose=True,
):
    """
    This function evaluates recommender metrics on a dataframe. While metric evaluation is fully
    configurable, by default the mean average precision at k (mAP@k), precision at k (precision@k)
    and recall at k (recall@k) are calculated for k in [1, 5, 10 and 20]. Performance is evaluated
    by first grouping by `group_col`, ranking by `score_col`, and evaluating the derived rank order
    for each group against `label_col`. The level at which these parameters are set can be
    controlled by setting the `k_list` parameter, which may be a positive integer, or a list of
    positiive integers.

    If the data in `score_col` is given by a model, it's likely that `ascending` argument should
    be set to `True`. If the `score_col` data comes from a search order position it should be
    set to `False`.

    Parameters
    ----------
    df : pandas dataframe
        This argument specifies the data that is to be scored. Only three columns are required,
        and these are given by the `group_col`, `label_col` and `score_col` arguments.

    k_list : int, list(int); optional (default=None)
        This specifies the level to which the performance metrics are calculated (e.g. mAP@20).
        This argument can specify one value of k (`k_list=20`), a list of parameters
        (`k_list=[1, 2, 4, 8]`), or it can revert to the default (`k_list=None`) wherein the
        values [1, 5, 10, 20] are used.

    ascending : bool; optional (default=False)
        This argument specifies whether the scores are ranked in ascending or descending order
        (default is descending). For models that give higher scores for better user-item affinity,
        ascending should be set to `False`, but if the data is generated from a search engine
        (where lower positions are indicitive of a 'better' position), ascending should be set to
        `True`.

    group_col : str; optional (default='group_id')
        This argument specifies the column of `df` over which groupings should be constructed.

    label_col : str; optional (default='label')
        This argument specifies the column of `df` that holds the ground truth labels.

    score_col : str; optional (default='score')
        This argument specifies the column of `df`

    metrics : dict or None; optional (default=None)
        The items of this dicionary specify the human-readable name of the metrics and a
        callable function to evaluate these metrics. The function signature for each metric
        must follow the following form:

        >>> def my_metric_function(df, df_at_k, score_col, label_col, ranked_col):
        >>>     pass

        Here, `df` is a slice of a dataframe with three columns (`score_col`, `label_col` and
        `ranked_col`) that is associated with a particular `group_id`, and `df_at_k` is a slice
        of `df` for rank values less than `k`. The column name variables are given so that metric
        is not tied to a particular schema. In general, these functions are for @k calculation.
        However, some metrics (e.g. recall) require a larger set of items to define the total
        number of possible. For example, if a search returns 100 items, and out of this 50 are
        relevant, the denominator of recall@20 will be 50.

    verbose : bool; optional (default=True)
        This specifies the verbosity level. If set to `True` the tqdm library will be invoked
        to siaplay a progress bar across the outermost grouping iterator.

    Returns
    -------
    results : pandas Dataframe
        A dataframe containing the performance metrics (as columns) computed across each group (row)

    results_mean : pandas Dataframe
        A dataframe of the metrics averaged across the groups.
    """

    assert group_col in df, f'The column {group_col} must be in the dataframe ({df.columns})'
    assert score_col in df, f'The column {score_col} must be in the dataframe ({df.columns})'
    assert label_col in df, f'The column {label_col} must be in the dataframe ({df.columns})'

    # Do basic validation
    k_list = validate_k_list(k_list)
    metrics = validate_metrics(metrics)

    # Rank the dataframe, and also sort by the group rank
    df_ranked_sorted, ranked_col = rank_dataframe(
        df=df,
        ascending=ascending,
        from_zero=False,
        sort_group_rank=True,
        group_col=group_col,
        score_col=score_col,
    )

    results_list = list()
    k_list = sorted(k_list)

    # Iterate over groups
    for group_id, sorted_ranked_group in verbose_iterator(
            df_ranked_sorted.groupby(group_col), verbose=verbose,
            desc=f'Calculating performance metrics over {group_col}'
    ):
        res = {group_col: group_id}

        # Iterate over the list of k values
        for k in k_list:
            # Slice the dataframe for ranks less or k (equality test since
            # the dataframe is indexed from 1; see `from_zero` field above)
            sorted_ranked_group_at_k = sorted_ranked_group.loc[(
                    df_ranked_sorted[ranked_col] <= k
            )]

            # Evaluate the metrics, and specify the k value in the keys
            for key, func in metrics.items():
                res[f'{key}@{k}'] = func(
                    df_at_k=sorted_ranked_group_at_k,
                    df=sorted_ranked_group,
                    score_col=score_col,
                    ranked_col=ranked_col,
                    label_col=label_col,
                )

        results_list.append(res)

    # Prepare the results dataframe
    results = pd.DataFrame(results_list)
    results.set_index(group_col, drop=True, inplace=True)
    results_mean = results.mean()

    return results, results_mean


def calculate_metrics(
        group_ids,
        scores,
        labels,
        k_list=None,
        ascending=False,
        metrics=None,
        verbose=True,
):
    """
    This function evaluates recommender metrics on a dataframe. While metric evaluation is fully
    configurable, by default the mean average precision at k (mAP@k), precision at k (precision@k)
    and recall at k (recall@k) are calculated for k in [1, 5, 10 and 20]. Performance is evaluated
    by first grouping by `group_col`, ranking by `score_col`, and evaluating the derived rank order
    for each group against `label_col`. The level at which these parameters are set can be
    controlled by setting the `k_list` parameter, which may be a positive integer, or a list of
    positiive integers.

    If the data in `score_col` is given by a model, it's likely that `ascending` argument should
    be set to `True`. If the `score_col` data comes from a search order position it should be
    set to `False`.

    Parameters
    ----------
    group_ids : list, ndarray, pandas Series; length N
        This array specifies the groups IDs over which the metrics are averaged

    scores : list, ndarray, pandas Series; length N
        This array specifies the scores that are used for evaluation

    labels : list, ndarray, pandas Series; length N
        This array specifies the ground truth labels that are used for evaluation.

    k_list : int, list(int); optional (default=None)
        This specifies the level to which the performance metrics are calculated (e.g. mAP@20).
        This argument can specify one value of k (`k_list=20`), a list of parameters
        (`k_list=[1, 2, 4, 8]`), or it can revert to the default (`k_list=None`) wherein the
        values [1, 5, 10, 20] are used.

    ascending : bool; optional (default=False)
        This argument specifies whether the scores are ranked in ascending or descending order
        (default is descending). For models that give higher scores for better user-item affinity,
        ascending should be set to `False`, but if the data is generated from a search engine
        (where lower positions are indicitive of a 'better' position), ascending should be set to
        `True`.

    metrics : dict or None; optional (default=None)
        The items of this dicionary specify the human-readable name of the metrics and a
        callable function to evaluate these metrics. The function signature for each metric
        must follow the following form:

        >>> def my_metric_function(df, df_at_k, score_col, label_col, ranked_col):
        >>>     pass

        Here, `df` is a slice of a dataframe with three columns (`score_col`, `label_col` and
        `ranked_col`) that is associated with a particular `group_id`, and `df_at_k` is a slice
        of `df` for rank values less than `k`. The column name variables are given so that metric
        is not tied to a particular schema. In general, these functions are for @k calculation.
        However, some metrics (e.g. recall) require a larger set of items to define the total
        number of possible. For example, if a search returns 100 items, and out of this 50 are
        relevant, the denominator of recall@20 will be 50.

    verbose : bool; optional (default=True)
        This specifies the verbosity level. If set to `True` the tqdm library will be invoked
        to siaplay a progress bar across the outermost grouping iterator.

    Returns
    -------
    results : pandas Dataframe
        A dataframe containing the performance metrics (as columns) computed across each group (row)

    results_mean : pandas Dataframe
        A dataframe of the metrics averaged across the groups.
    """

    if len(set(map(len, [group_ids, scores, labels]))) != 1:
        raise ValueError(
            'All inputs to this function must be of the same length'
        )

    # Populate a new dataframe and invoke the previous function to calculate metrics
    df = pd.DataFrame(dict(
        group_id=group_ids,
        score=scores,
        label=labels,
    ))

    return calculate_metrics_from_dataframe(
        df=df,
        k_list=k_list,
        ascending=ascending,
        group_col='group_id',
        score_col='score',
        label_col='label',
        metrics=metrics,
        verbose=verbose,
    )


if __name__ == '__main__':
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    from recommender_metrics import random_data

    full_results, mean_results = calculate_metrics_from_dataframe(random_data.predefined_data())
    print(mean_results)

    full_results, mean_results = calculate_metrics_from_dataframe(random_data.generate_random_data())
    print(mean_results)
