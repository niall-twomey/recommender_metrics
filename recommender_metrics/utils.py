import numpy as np
from tqdm import tqdm

__all__ = [
    'group_score_and_labelled_data',
    'verbose_iterator',
]


def validate_data_type(arr):
    if isinstance(arr, np.ndarray):
        return arr
    return np.asarray(arr)


def get_sort_order(
        group_ids,
        scores,
        ascending=False
):
    # nump.lexsort sorts from right to left in its first argument
    if ascending:
        return np.lexsort((scores, group_ids))
    return np.lexsort((-scores, group_ids))


def sort_data(
        group_ids,
        scores,
        labels,
        ascending=False
):
    inds = get_sort_order(
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
    # The mask is sorted, so find the changepoints mask
    changepoint_mask = (group_ids[1:] != group_ids[:-1])
    changepoint_mask = np.append(True, changepoint_mask)
    changepoint_mask = np.append(changepoint_mask, True)

    # Get the changepoint inds
    split_inds = np.argwhere(changepoint_mask).ravel()

    # Slice positions:
    #   {group_ids[start]: (start, end) for start, end in zip(split_inds[:-1], split_inds[1:])}

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
    group_ids = validate_data_type(group_ids)
    scores = validate_data_type(scores)
    labels = validate_data_type(labels)

    group_ids, scores, labels = sort_data(
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


if __name__ == '__main__':
    def main():
        rng = np.random.RandomState(1234)

        N = 30

        group_ids = rng.randint(0, 10, N)
        scores = rng.normal(0, 1, N)
        labels = rng.rand(N) > 0.8

        sorted_data, grouped_data = group_score_and_labelled_data(
            group_ids=group_ids,
            scores=scores,
            labels=labels,
        )

        print(np.c_[group_ids, scores.round(1), labels])

        for group, kwargs in grouped_data.items():
            print(group)
            for kk, vv in kwargs.items():
                print(kk, vv)
            print()
            kwargs['scores'] += 10


    main()
