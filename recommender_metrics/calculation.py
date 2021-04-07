from functools import partial
from typing import Any
from typing import Callable
from typing import Dict
from typing import Iterator
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
from sklearn.base import BaseEstimator

from recommender_metrics.metrics import DEFAULT_METRICS
from recommender_metrics.metrics import METRIC_FUNCTIONS
from recommender_metrics.utils import group_score_and_labelled_data
from recommender_metrics.utils import validate_array_type
from recommender_metrics.utils import verbose_iterator

__all__ = [
    "calculate_metrics_from_dataframe",
    "calculate_metrics",
    "IncrementalMetrics",
]


def _validate_k_list(k_list: Optional[Union[int, List[int]]]) -> List[int]:
    if k_list is None:
        return [1, 5, 10, 20]
    elif isinstance(k_list, int):
        assert k_list > 0
        return [k_list]
    elif isinstance(k_list, list):
        assert all(map(lambda kk: isinstance(kk, int) and kk > 0, k_list))
        return k_list
    raise ValueError


def _validate_metrics(
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


def _metric_iterator(k_list: List[int], metrics: Dict[str, Callable]) -> Iterator[Tuple[int, str, Callable]]:
    for k in k_list:
        for func_name, func in metrics.items():
            yield k, func_name, func


def _evaluate_performance_single_thread(
    group_dict: Dict[Any, Dict[str, Union[float, np.ndarray]]],
    k_list: List[int],
    metrics: Dict[str, Callable],
    verbose: bool,
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    num_metrics = len(k_list) * len(metrics)
    results = np.empty(shape=(len(group_dict), num_metrics))
    weights = np.empty(shape=len(group_dict))

    # Iterate over groups
    for gi, group in verbose_iterator(
        iterator=enumerate(group_dict.values()),
        verbose=verbose,
        total=len(group_dict),
        desc=f"Evaluating performance",
    ):
        weights[gi] = weight = group.pop("weight", 1.0)
        for fi, (k, _, metric) in enumerate(_metric_iterator(k_list, metrics)):
            results[gi, fi] = metric(k=k, **group) * weight

    keys = [f"{metric_name}@{k}" for k, metric_name, _ in _metric_iterator(k_list, metrics)]

    return keys, results, weights


def _evaluate_performance_multiple_threads(
    grouped_data: Dict[Any, Dict[str, np.ndarray]],
    k_list: List[int],
    metrics: Dict[str, Callable],
    n_threads: int,
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    raise NotImplementedError


def _calculate_metrics_from_grouped_data(
    grouped_data: Dict[Any, Dict[str, np.ndarray]],
    k_list: Optional[Union[int, List[int]]] = None,
    metrics: Optional[Union[List[str], Dict[str, Callable]]] = None,
    verbose: bool = True,
    reduce: bool = True,
    n_threads: int = 1,
) -> Union[Dict[str, float], Dict[str, np.ndarray]]:
    assert isinstance(n_threads, int) and (n_threads > 0 or n_threads == -1)

    # Do basic validation
    k_list = _validate_k_list(k_list)
    metrics = _validate_metrics(metrics)

    #  Calculate the results
    if n_threads > 1:
        keys, results, weights = _evaluate_performance_multiple_threads(
            grouped_data=grouped_data,
            k_list=k_list,
            metrics=metrics,
            n_threads=n_threads,
        )

    else:
        keys, results, weights = _evaluate_performance_single_thread(
            group_dict=grouped_data,
            k_list=k_list,
            metrics=metrics,
            verbose=verbose,
        )

    if reduce:
        return dict(zip(keys, results.sum(0) / weights.sum()))

    return dict(zip(keys, results.T))


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
    eps: float = 1e-9,
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

    eps : float (default=1e-9)
        This argument specifies the the range of noice that is added to the scores to break ties.
        Scores will be modified to `scores + np.random.uniform(-eps, eps, shape)`. Set to 0 to
        not do this.

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
        eps=eps,
    )


def calculate_metrics(
    group_ids: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    weights: Optional[np.ndarray] = None,
    k_list: Optional[Union[int, List[int]]] = None,
    metrics: Optional[Union[List[str], Dict[str, Callable]]] = None,
    ascending: bool = False,
    verbose: bool = True,
    reduce: bool = True,
    remove_empty: bool = False,
    n_threads: int = 1,
    eps: float = 1e-9,
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

    weights : list, ndarray, pandas Series; length N
        This array specifies the weights associated with particular instances

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

    eps : float (default=1e-9)
        This argument specifies the the range of noice that is added to the scores to break ties.
        Scores will be modified to `scores + np.random.uniform(-eps, eps, shape)`. Set to 0 to
        not do this.

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
        weights=weights,
        ascending=ascending,
        verbose=verbose,
        remove_empty=remove_empty,
        eps=eps,
    )

    return _calculate_metrics_from_grouped_data(
        grouped_data=grouped_data,
        metrics=metrics,
        verbose=verbose,
        reduce=reduce,
        n_threads=n_threads,
        k_list=k_list,
    )


class IncrementalMetrics(BaseEstimator):
    """
    This class wraps the main metric calculation functionality. For large datasets it can be slow and memory
    intensive to generate the requisite data to calculate metrics. This class wraps things so that you can do
    each group individually.
    """

    def __init__(
        self,
        k_list=None,
        metrics=None,
        ascending=False,
        verbose=True,
        remove_empty=False,
        n_threads=1,
        eps: float = 1e-9,
    ):
        """
        Parameters
        ----------
        k_list : int, list(int); optional (default=None)
            This specifies the level to which the performance metrics are calculated (e.g. mAP@20).
            This argument can specify one value of k (`k_list=20`), a list of parameters
            (`k_list=[1, 2, 4, 8]`), or it can revert to the default (`k_list=None`) wherein the
            values [1, 5, 10, 20] are used.

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

        ascending : bool; optional (default=False)
            This argument specifies whether the scores are ranked in ascending or descending order
            (default is descending). For models that give higher scores for better user-item affinity,
            ascending should be set to `False`, but if the data is generated from a search engine
            (where lower positions are indicitive of a 'better' position), ascending should be set to
            `True`.

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

        eps : float (default=1e-9)
            This argument specifies the the range of noice that is added to the scores to break ties.
            Scores will be modified to `scores + np.random.uniform(-eps, eps, shape)`. Set to 0 to
            not do this.
        """
        self.ascending = ascending
        self.verbose = verbose
        self.remove_empty = remove_empty
        self.n_threads = n_threads
        self.eps = eps

        self.k_list = _validate_k_list(k_list)
        self.metrics = _validate_metrics(metrics)

        self.metric_names = [f"{func_name}@{k}" for k, func_name, _ in _metric_iterator(self.k_list, self.metrics)]
        self.metric_funcs = [partial(func, k=k) for k, _, func in _metric_iterator(self.k_list, self.metrics)]

        self.weight_sum = None
        self.aggregates = None

        self.reset()

    def reset(self):
        self.weight_sum = 0.0
        self.aggregates = {kk: 0.0 for kk in self.metric_names}

    def append_group(self, scores, labels, weight=1.0, assume_sorted: bool = False):
        """
        Parameters
        ----------
        scores : list, ndarray, pandas Series; length N
            This array specifies the scores that are used for evaluation

        labels : list, ndarray, pandas Series; length N
            This array specifies the ground truth labels that are used for evaluation.

        weight : float (default=1.0)
            A weight assiciated with the group.

        assume_sorted : bool (default=False)
            This argument specifies whether the function should sort by scores or not
        """

        scores = validate_array_type(scores, dtype=float)
        labels = validate_array_type(labels, dtype=bool)

        # Ignore unnecessary computations
        if self.remove_empty and not labels.any():
            return

        # Break ties
        if self.eps:
            scores = scores + np.random.uniform(-self.eps, self.eps, scores.shape)

        # Performance enhancement, if known that scores are correctly sorted already
        if not assume_sorted:
            rank_order = np.argsort(scores) if self.ascending else np.argsort(-scores)
            scores = scores[rank_order]
            labels = labels[rank_order]

        # Perform the aggregation over this group
        for name, func in zip(self.metric_names, self.metric_funcs):
            self.aggregates[name] += func(labels=labels, scores=scores) * weight

        self.weight_sum += weight

    def resolve(self):
        """Return a dictionary containing the metrics"""
        return {kk: vv / self.weight_sum for kk, vv in self.aggregates.items()}

    def keys(self):
        """An iterator over metric keys"""
        return self.aggregates.keys()

    def values(self):
        """An iterator over metric values"""
        return self.aggregates.values()

    def items(self):
        """An iterator over metric items"""
        return self.aggregates.items()

    def __getitem__(self, item):
        """A getter for a particular key"""
        return self.aggregates[item]
