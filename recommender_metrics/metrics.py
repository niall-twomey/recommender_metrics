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


def average_precision(df_at_k, df):
    pos_group = df_at_k.loc[df_at_k.label == 1]
    if len(pos_group) == 0:
        return 0.0  # TODO: check this ret val
    precisions = pos_group.label.cumsum() / pos_group.score_rank
    return precisions.mean()


def precision(df_at_k, df):
    precision_at_k = df_at_k['label'].cumsum() / df_at_k['score_rank']
    return precision_at_k.values[-1]


def recall(df_at_k, df):
    den = df['label'].sum()
    if den == 0:
        return 1.0  # TODO: check this ret val
    recalls = df_at_k['label'].cumsum() / den
    return recalls.values[-1]


def calculate_metrics_from_dataframe(df, k_list=None, **metrics):
    # Do basic validation on the list of k values
    if isinstance(k_list, int):
        k_list = [k_list]

    if k_list is None or len(k_list) == 0:
        k_list = [1, 5, 10, 20]

    # Do basic validation on the dictionary of metrics
    if len(metrics) == 0:
        metrics = dict(
            mAP=average_precision,
            precison=precision,
            reccall=recall
        )

    if not all(map(callable, metrics.values())):
        raise TypeError(
            f'All metrics passed into this function must be callable'
        )

    # Rank the dataframe, and also sort by the group rank
    df_ranked_sorted = utils.rank_dataframe(
        df=df,
        from_zero=False,
        sort_group_rank=True,
    )

    results_list = list()
    k_list = sorted(k_list)

    # Iterate over groups
    for group_id, sorted_ranked_group in tqdm(
            df_ranked_sorted.groupby('group_id'),
            desc='Metric calculation'
    ):
        res = dict(
            group_id=group_id,
            user_id=sorted_ranked_group['user_id'].iloc[0]
        )

        # Iterate over the list of k values
        for k in k_list:
            # Slice the dataframe for ranks less or k (equality test since
            # the dataframe is indexed from 1; see `from_zero` field above)
            sorted_ranked_group_at_k = sorted_ranked_group.loc[
                df_ranked_sorted['score_rank'] <= k
                ]

            # Evaluate the metrics, and specify the k value in the keys
            for key, func in metrics.items():
                res[f'{key}_at_{k}'] = func(
                    df_at_k=sorted_ranked_group_at_k,
                    df=sorted_ranked_group
                )

        results_list.append(res)

    # Prepare the results dataframe
    results = pd.DataFrame(results_list)
    results.set_index(['group_id', 'user_id'], drop=True, inplace=True)
    results_mean = results.mean()

    return results, results_mean


def calculate_metrics(group_ids, user_ids, item_ids, scores, labels, k_list=None, **metrics):
    if len(set(map(len, [group_ids, user_ids, item_ids, scores, labels]))) != 1:
        raise ValueError(
            'All inputs to this function must be of the same length'
        )

    # Populate a new dataframe and invoke the previous function to calculate metrics
    df = pd.DataFrame(dict(
        group_id=group_ids,
        user_id=user_ids,
        item_id=item_ids,
        score=scores,
        label=labels,
    ))

    return calculate_metrics_from_dataframe(
        df=df, k_list=k_list, **metrics
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
        user_ids=df.user_id,
        item_ids=df.item_id,
        scores=df.score,
        labels=df.label
    )

    print(mean_results)


if __name__ == '__main__':
    main()
