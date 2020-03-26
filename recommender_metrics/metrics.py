import pandas as pd

from recommender_metrics import utils
from tqdm import tqdm

__all__ = [
    'average_precision',
    'precision',
    'recall',
    'calculate_metrics_from_dataframe',
    'calculate_metrics'
]


def average_precision(df_at_k, df, ranked_col, label_col):
    pos_group = df_at_k.loc[df_at_k[label_col] == 1]
    if len(pos_group) == 0:
        return 0.0  # TODO: check this ret val
    precisions = pos_group[label_col].cumsum() / pos_group[ranked_col]
    return precisions.mean()


def precision(df_at_k, df, ranked_col, label_col):
    precision_at_k = df_at_k[label_col].cumsum() / df_at_k[ranked_col]
    return precision_at_k.values[-1]


def recall(df_at_k, df, ranked_col, label_col):
    den = df[label_col].sum()
    if den == 0:
        return 1.0  # TODO: check this ret val
    recalls = df_at_k[label_col].cumsum() / den
    return recalls.values[-1]


def calculate_metrics_from_dataframe(
        df,
        k_list=None,
        ascending=False,
        group_col='group_id',
        label_col='label',
        score_col='score',
        metrics=None
):
    # Do basic validation on the list of k values
    if isinstance(k_list, int):
        k_list = [k_list]
    elif k_list is None or len(k_list) == 0:
        k_list = [1, 5, 10, 20]

    # Do basic validation on the dictionary of metrics
    if metrics is None or len(metrics) == 0:
        metrics = dict(
            mAP=average_precision,
            precison=precision,
            reccall=recall
        )
    assert isinstance(metrics, dict)
    if not all(map(callable, metrics.values())):
        raise TypeError(
            f'All metrics passed into this function must be callable'
        )

    # Rank the dataframe, and also sort by the group rank
    df_ranked_sorted, ranked_col = utils.rank_dataframe(
        df=df,
        ascending=ascending,
        from_zero=False,
        sort_group_rank=True,
        group_col=group_col,
        score_col=score_col,
    )

    results_list = list()
    k_list = sorted(k_list)

    keys = [group_col]

    # Iterate over groups
    for group_id, sorted_ranked_group in tqdm(
            df_ranked_sorted.groupby(group_col),
            desc='Metric calculation'
    ):
        res = {group_col: group_id}

        # Iterate over the list of k values
        for k in k_list:
            # Slice the dataframe for ranks less or k (equality test since
            # the dataframe is indexed from 1; see `from_zero` field above)
            sorted_ranked_group_at_k = sorted_ranked_group.loc[
                df_ranked_sorted[ranked_col] <= k
                ]

            # Evaluate the metrics, and specify the k value in the keys
            for key, func in metrics.items():
                res[f'{key}_at_{k}'] = func(
                    df_at_k=sorted_ranked_group_at_k,
                    df=sorted_ranked_group,
                    ranked_col=ranked_col,
                    label_col=label_col,
                )

        results_list.append(res)

    # Prepare the results dataframe
    results = pd.DataFrame(results_list)
    results.set_index(keys, drop=True, inplace=True)
    results_mean = results.mean()

    return results, results_mean


def calculate_metrics(
        group_ids,
        scores,
        labels,
        k_list=None,
        ascending=False,
        metrics=None
):
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
    )


def main():
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    from recommender_metrics.random_data import generate_random_data
    df = generate_random_data()
    print(df.head())

    full_results, mean_results = calculate_metrics(
        group_ids=df.group_id,
        scores=df.score,
        labels=df.label,
    )

    print(full_results)
    print(mean_results)


if __name__ == '__main__':
    main()
