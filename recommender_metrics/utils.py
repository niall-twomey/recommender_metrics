import numpy as np
from tqdm import tqdm

__all__ = [
    'sort_recommender_data',
    'partition_by_group_from_sorted',
    'group_score_and_labelled_data',
    'verbose_iterator',
]


def _validate_data_type(arr):
    if isinstance(arr, np.ndarray):
        return arr
    return np.asarray(arr)


def _get_sorted_indices(
        group_ids,
        scores,
        ascending=False
):
    # nump.lexsort sorts from right to left in its first argument
    if ascending:
        return np.lexsort((scores, group_ids))
    return np.lexsort((-scores, group_ids))


def _get_changepoints(group_ids):
    # The mask is sorted, so find the changepoints mask
    changepoint_mask = (group_ids[1:] != group_ids[:-1])
    changepoint_mask = np.append(True, changepoint_mask)
    changepoint_mask = np.append(changepoint_mask, True)

    # Get the changepoint inds
    split_inds = np.argwhere(changepoint_mask).ravel()

    # Slice positions:
    #   {group_ids[start]: (start, end) for start, end in zip(split_inds[:-1], split_inds[1:])}

    return split_inds


def sort_recommender_data(
        group_ids,
        scores,
        labels,
        ascending=False
):
    inds = _get_sorted_indices(
        group_ids=group_ids,
        scores=scores,
        ascending=ascending,
    )

    return group_ids[inds], scores[inds], labels[inds]


def partition_by_group_from_sorted(
        group_ids,
        scores,
        labels,
        verbose,
):
    split_inds = _get_changepoints(group_ids)

    return {
        group_ids[start]: dict(
            scores=scores[start:end].astype(float),
            labels=labels[start:end].astype(bool),
            ranks=np.arange(1, end - start + 1).astype(int),
        ) for start, end in verbose_iterator(
            zip(
                split_inds[:-1],
                split_inds[1:],
            ),
            total=len(split_inds) - 1,
            desc=f'Grouping data before evaluation',
            verbose=verbose,
        )
    }


def group_score_and_labelled_data(
        group_ids,
        scores,
        labels,
        ascending=False,
        verbose=True
):
    group_ids = _validate_data_type(group_ids)
    scores = _validate_data_type(scores)
    labels = _validate_data_type(labels)

    group_ids, scores, labels = sort_recommender_data(
        group_ids=group_ids,
        scores=scores,
        labels=labels,
        ascending=ascending,
    )

    sorted_data = dict(
        group_ids=group_ids,
        scores=scores,
        labels=labels,
    )

    grouped_data = partition_by_group_from_sorted(
        group_ids=group_ids,
        scores=scores,
        labels=labels,
        verbose=verbose
    )

    return sorted_data, grouped_data


def verbose_iterator(iterator, total=None, desc=None, verbose=True):
    if verbose:
        return tqdm(iterator, total=total, desc=desc)
    return iterator
