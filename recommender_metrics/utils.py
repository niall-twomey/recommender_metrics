from typing import Any
from typing import Dict
from typing import Iterator
from typing import Optional
from typing import Tuple

import numpy as np
from tqdm import tqdm

__all__ = [
    "sort_recommender_data",
    "partition_by_group_from_sorted",
    "group_score_and_labelled_data",
    "verbose_iterator",
]


def _validate_data_type(arr: np.ndarray) -> np.ndarray:
    """
    Verify that an array type is a numpy array and is one dimensional.

    Parameters
    ----------
    arr : np.ndarray, dtype=any

    Returns
    -------
        np.ndarray : validated datatype
    """

    if not isinstance(arr, np.ndarray):
        arr = np.asarray(arr)
    assert arr.ndim == 1
    return arr


def _encode_group_ids(group_ids: np.ndarray) -> np.ndarray:
    """
    A helper function that (might) be helpful in speeding up the sorting process by converting
    the unique values of the groups to integers. Note, this function isn't reversible since it
    doesn't return the unique values themselves.

    Parameters
    ----------
    group_ids : np.ndarray with any dtype

    Returns
    -------
        np.ndarray : The encoded group IDs
    """

    unique, inverse = np.unique(group_ids, return_inverse=True)
    return inverse


def _get_sorted_indices(group_ids: np.ndarray, scores: np.ndarray, ascending: bool = False) -> np.ndarray:
    """
    This function returns the lexicographically order of two arrays. Order is first based on the
    `group_ids` argument and secondly by the `scores`. Groups/scores can be sorted as follows:

    >>> inds = _get_sorted_indices(group_ids, scores)
    >>> sorted_group_ids, sorted_scores = group_ids[inds], scores[inds]

    Parameters
    ----------
    group_ids : np.ndarray, any dtype
        The IDs over which to group the scores.
    scores : np.ndarray, numeric dtype
        Scores associated with the group
    ascending : bool, optional (default=False)
        Specifies whether the sorting is done in ascending order (default is descending)

    Returns
    -------
        np.ndarray : The ranked order of the input data.

    """

    # group_ids = _encode_group_ids(group_ids)  # Doesn't add much improvement
    # nump.lexsort sorts from right to left in its first argument
    if ascending:
        return np.lexsort((scores, group_ids))
    return np.lexsort((-scores, group_ids))


def _get_changepoints(group_ids: np.ndarray) -> np.ndarray:
    """
    Given a sorted array of `group_ids`, this function returns the points in the array where
    the value of the `group_ids` change.

    Parameters
    ----------
    group_ids : np.ndarray
        The (sorted) array of group IDs

    Returns
    -------
        np.ndarray : the positions in the array where `group_ids` change
    """

    # The mask is sorted, so find the changepoints mask
    changepoint_mask = group_ids[1:] != group_ids[:-1]
    changepoint_mask = np.append(True, changepoint_mask)
    changepoint_mask = np.append(changepoint_mask, True)

    # Get the changepoint inds
    split_inds = np.argwhere(changepoint_mask).ravel()

    return split_inds


def sort_recommender_data(
    group_ids: np.ndarray, scores: np.ndarray, labels: np.ndarray, ascending: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    This function takes the group specification, scores and labels arrays. The function then sorts
    these according to the groups and scores, and returns the sorted arrays.

    Parameters
    ----------
    group_ids : np.ndarray
        Array containing the unique groups
    scores : np.ndarray
        Array containing the scores
    labels : np.ndarray
        Array containing the labels
    ascending : bool, optional (default=False)
        Whether to sort in ascending or descending (default) order

    Returns
    -------
        groups_sorted : np.ndarray
            The group IDs sorted by (group_id, score)
        scores_sorted : np.ndarray
            The scores sorted by (group_id, score)
        labels_sorted : np.ndarray
            The labels sorted by (group_id, score)
    """

    inds = _get_sorted_indices(group_ids=group_ids, scores=scores, ascending=ascending)

    return group_ids[inds], scores[inds], labels[inds]


def partition_by_group_from_sorted(
    group_ids: np.ndarray, scores: np.ndarray, labels: np.ndarray, verbose: bool = False
) -> Dict[Any, Dict[str, np.ndarray]]:
    """
    This function takes in (sorted) group spec, scores and labels and slices these into a dictionary
    structure in which every `key` corresponds a single group, and the `values` are another dictionary
    containing the `scores`, `labels` and `ranks` of that group.

    Parameters
    ----------
    group_ids : np.ndarray
        Array containing the unique groups
    scores : np.ndarray
        Array containing the scores
    labels : np.ndarray
        Array containing the labels
    verbose : bool, optional (default=False)

    Returns
    -------
        group_dict : dict
            Each `key` is a group ID, and its corresponding `value` is another dictionary containing
            `scores`, `labels` and `ranks` of that group (each an np.ndarray)
    """

    split_inds = _get_changepoints(group_ids)

    return {
        group_ids[start]: dict(
            scores=scores[start:end].astype(float),
            labels=labels[start:end].astype(bool),
            ranks=np.arange(1, end - start + 1).astype(int),
        )
        for start, end in verbose_iterator(
            zip(split_inds[:-1], split_inds[1:]),
            total=len(split_inds) - 1,
            desc=f"Grouping data before evaluation",
            verbose=int(verbose) > 1,
        )
    }


def group_score_and_labelled_data(
    group_ids: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    remove_empty: bool = False,
    ascending: bool = False,
    verbose: bool = True,
) -> Dict[Any, Dict[str, np.ndarray]]:
    """
    This dict has `np.unique(group_ids).shape[0]` elements. The keys correspond to the unique values
    of `group_ids`, and the `value`s are dictionary explicitly providing the `scores`, `labels` and
    `ranks` of the input data according to the optional configurations

    Parameters
    ----------
    group_ids : np.ndarray
        Array containing the unique groups
    scores : np.ndarray
        Array containing the scores
    labels : np.ndarray
        Array containing the labels
    remove_empty : bool, optional (default=False)
        Specifies whether to keep or remove groups with no positive labels
    ascending : bool, optional (default=False)
        Specifies whether to sort based on ascending or descending scores.
    verbose : bool, optional (default=True)
        The default verbosity level

    Returns
    -------
        group_dict : dict
            The group specification, see `partition_by_group_from_sorted` for details.
    """

    group_ids = _validate_data_type(group_ids)
    scores = _validate_data_type(scores)
    labels = _validate_data_type(labels)

    group_ids, scores, labels = sort_recommender_data(
        group_ids=group_ids, scores=scores, labels=labels, ascending=ascending,
    )

    grouped_data = partition_by_group_from_sorted(group_ids=group_ids, scores=scores, labels=labels, verbose=verbose)

    if remove_empty:
        grouped_data = {kk: vv for kk, vv in grouped_data.items() if vv["labels"].any()}

    return grouped_data


def verbose_iterator(
    iterator: Iterator[Any], total: Optional[int] = None, desc: Optional[str] = None, verbose: bool = True
) -> Iterator[Any]:
    """
    This function yields from the iterator as normal, but if `verbose=True` also uses the TQDM package
    to display progress

    Parameters
    ----------
    iterator : iterator
        An iterator over sequence of objects.
    total : int, optional (default=None)
        The number of items in the iterator
    desc : str, optional (default=None)
        A description of the iterator that is used when `verbose=True`
    verbose : bool, optional (default=True)

    Yields
    ------
        Yields from the `iterator`
    """

    if verbose:
        iterator = tqdm(iterator, total=total, desc=desc)
    yield from iterator
