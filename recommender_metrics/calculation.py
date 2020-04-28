from collections import Counter
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import numpy as np

from recommender_metrics.metrics import DEFAULT_METRICS
from recommender_metrics.metrics import METRIC_FUNCTIONS
from recommender_metrics.utils import group_score_and_labelled_data
from recommender_metrics.utils import verbose_iterator

__all__ = [
    "calculate_metrics_from_grouped_data",
    "calculate_metrics_from_dataframe",
    "calculate_metrics",
    "validate_k_list",
    "validate_metrics",
]


def validate_k_list(k_list: Optional[Union[int, List[int]]]) -> List[int]:
    if k_list is None:
        return [1, 5, 10, 20]
    elif isinstance(k_list, int):
        assert k_list > 0
        return [k_list]
    elif isinstance(k_list, list):
        assert all(map(lambda kk: isinstance(kk, int) and kk > 0, k_list))
        return k_list
    raise ValueError


def validate_metrics(
    metrics: Optional[Union[List[str], Dict[str, Callable]]], metric_dict: Dict[str, Callable] = None
) -> Dict[str, Callable]:
    metric_dict = {**METRIC_FUNCTIONS, **(metric_dict or dict())}
    if metrics is None:
        metrics = DEFAULT_METRICS
    if isinstance(metrics, str):
        metrics = [metrics]
    if isinstance(metrics, list) and all(map(lambda mm: isinstance(mm, str), metrics)):
        metrics = {mm: metric_dict[mm] for mm in metrics}
    assert isinstance(metrics, dict)
    if not all(map(callable, metrics.values())):
        raise TypeError(f"All metrics passed into this function must be callable")
    return metrics


def _reduce_results(results_list: List[Dict[str, float]]) -> Dict[str, float]:
    counter = Counter()
    for result in results_list:
        counter.update(result)
    count = float(len(results_list))
    return {kk: float(vv) / count for kk, vv in counter.items() if "@" in kk}


def _evaluate_performance_single_thread(
    group_dict: Dict[Any, Dict[str, np.ndarray]], k_list: List[int], metrics: Dict[str, Callable], verbose: bool
) -> List[Dict[str, float]]:
    results_list = [dict() for _ in range(len(group_dict))]

    # Iterate over groups
    for res, (group_id, group) in verbose_iterator(
        iterator=zip(results_list, group_dict.items()),
        verbose=verbose,
        total=len(group_dict),
        desc=f"Evaluating performance",
    ):
        for k in k_list:
            for key, func in metrics.items():
                res[f"{key}@{k}"] = func(k=k, **group)

    return results_list


def _evaluate_performance_multiple_threads(
    grouped_data: Dict[Any, Dict[str, np.ndarray]], k_list: List[int], metrics: Dict[str, Callable], n_threads: int
) -> List[Dict[str, float]]:
    raise NotImplementedError


def calculate_metrics_from_grouped_data(
    grouped_data: Dict[Any, Dict[str, np.ndarray]],
    k_list: Optional[Union[int, List[int]]] = None,
    metrics: Optional[Union[List[str], Dict[str, Callable]]] = None,
    verbose: bool = True,
    reduce: bool = True,
    n_threads: int = 1,
) -> Union[Dict[str, float], List[Dict[str, float]]]:
    assert isinstance(n_threads, int) and n_threads > 0

    # Do basic validation
    k_list = validate_k_list(k_list)
    metrics = validate_metrics(metrics)

    #  Calculate the results
    if n_threads > 1:
        results_list = _evaluate_performance_multiple_threads(
            grouped_data=grouped_data, k_list=k_list, metrics=metrics, n_threads=n_threads,
        )

    else:
        results_list = _evaluate_performance_single_thread(
            group_dict=grouped_data, k_list=k_list, metrics=metrics, verbose=verbose,
        )

    if reduce:
        return _reduce_results(results_list=results_list)

    return results_list


def calculate_metrics_from_dataframe(
    df,
    group_col: str = "group_id",
    label_col: str = "label",
    score_col: str = "score",
    k_list: Optional[Union[int, List[int]]] = None,
    metrics: Optional[Union[List[str], Dict[str, Callable]]] = None,
    ascending: bool = False,
    verbose: bool = True,
    reduce: bool = True,
    remove_empty: bool = False,
    n_threads: int = 1,
) -> Union[Dict[str, float], List[Dict[str, float]]]:
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

        >>> def my_metric_function(scores, labels, ranks, k):
        >>>     pass

        Here, `scores`, `labels`, `ranks` are ndarrays of length N (N>=k). Since some metrics
        (e.g. recall) require the full set of data to produce outputs the full list of arguments
        are passed in. Functions calculating @k need to slice these. For an exmaple of this see
        the `recommender_metrics/metrics.py` file.

    verbose : bool; optional (default=True)
        This specifies the verbosity level. If set to `True` the tqdm library will be invoked
        to siaplay a progress bar across the outermost grouping iterator.

    reduce : bool, optional (default=True)
        This argument determines whether to return the (arrhythmic) mean of the scores across
        the groups.

    remove_empty : bool, optional (default=False)
        This argument specifies whether groups of data with no positive labels should be used
        in evaluation. If `True` this effectively scales results by the CTR (if labels relate
        to clicks).

    n_threads : int; optional (default=1)
        This argument specifies the number of threads that this computation is done.

    Returns
    -------
    results : dict or pandas.DataFrame
        If `reduce=True` a dictionary of metric name / metric result measures is returned. These
            are the results of averaging across `group_id`.
        If `reduce=False` a list of dictionaries is returned. Each element corresponds to the metrics
            estimated on a particular group.
    """

    return calculate_metrics(
        group_ids=df[group_col].values,
        scores=df[score_col].values,
        labels=df[label_col].values,
        k_list=k_list,
        ascending=ascending,
        metrics=metrics,
        verbose=verbose,
        reduce=reduce,
        remove_empty=remove_empty,
        n_threads=n_threads,
    )


def calculate_metrics(
    group_ids: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    k_list: Optional[Union[int, List[int]]] = None,
    metrics: Optional[Union[List[str], Dict[str, Callable]]] = None,
    ascending: bool = False,
    verbose: bool = True,
    reduce: bool = True,
    remove_empty: bool = False,
    n_threads: int = 1,
) -> Union[Dict[str, float], List[Dict[str, float]]]:
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

        >>> def my_metric_function(scores, labels, ranks, k):
        >>>     pass

        Here, `scores`, `labels`, `ranks` are ndarrays of length N (N>=k). Since some metrics
        (e.g. recall) require the full set of data to produce outputs the full list of arguments
        are passed in. Functions calculating @k need to slice these. For an exmaple of this see
        the `recommender_metrics/metrics.py` file.

    verbose : bool; optional (default=True)
        This specifies the verbosity level. If set to `True` the tqdm library will be invoked
        to siaplay a progress bar across the outermost grouping iterator.

    reduce : bool, optional (default=True)
        This argument determines whether to return the (arrhythmic) mean of the scores across
        the groups.

    remove_empty : bool, optional (default=False)
        This argument specifies whether groups of data with no positive labels should be used
        in evaluation. If `True` this effectively scales results by the CTR (if labels relate
        to clicks).

    n_threads : int; optional (default=1)
        This argument specifies the number of threads that this computation is done.

    Returns
    -------
    results : dict or pandas.DataFrame
        If `reduce=True` a dictionary of metric name / metric result measures is returned. These
            are the results of averaging across `group_id`.
        If `reduce=False` a list of dictionaries is returned. Each element corresponds to the metrics
            estimated on a particular group.
    """

    grouped_data = group_score_and_labelled_data(
        group_ids=group_ids,
        scores=scores,
        labels=labels,
        ascending=ascending,
        verbose=verbose,
        remove_empty=remove_empty,
    )

    return calculate_metrics_from_grouped_data(
        grouped_data=grouped_data, metrics=metrics, verbose=verbose, reduce=reduce, n_threads=n_threads, k_list=k_list
    )
