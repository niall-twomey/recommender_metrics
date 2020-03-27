import numpy as np
import pandas as pd

__all__ = [
    'generate_random_data',
    'predefined_data',
]


def _test_gh_data():
    from recommender_metrics.metrics import calculate_metrics_from_dataframe, average_precision

    groupby_keys = ['session_no', 'query']
    info_keys = ['action', 'position']

    scores = pd.read_csv('../data_private/map_sample_dataset.csv')
    scores = scores[groupby_keys + info_keys][scores.action == 'click']

    rows = []
    for group_id, group in scores.groupby(['session_no', 'query']):
        session_no, query = group_id
        click_set = set(group.position.values - 1)
        assert min(click_set) >= 0
        # This data assumes that at least 20 items are presented to the user
        for ii in range(max(20, int(max(click_set)))):
            rows.append(dict(
                group_id=(session_no, query),
                label=int(ii in click_set),
                score=-ii,
            ))
        # Debug messages
        # if session_no == 3513781251:
        #     sel = scores[scores.session_no == session_no]
        #     print(sel.shape)
        #     print(sel.to_string(index=False))
        #     inds = sel.sort_values('position')['position'].unique()
        #     inds = inds[inds < 20]
        #     print(inds)
        #     print(np.ones_like(inds).cumsum() / inds)
        #     print((np.ones_like(inds).cumsum() / inds).mean())
        #     print()

    # Calculate metrics over this data
    score_df = pd.DataFrame(rows)
    mets, met_mean = calculate_metrics_from_dataframe(score_df)

    # Load GH's results
    scores_sorted = pd.read_csv('../data_private/map_sample_dataset_scored.csv')
    ghd = scores_sorted.groupby(
        ['session_no', 'query']
    )['mean_average_precision'].max().to_dict()

    # Print the instances GH's and my metrics disagree
    for session_no, met in mets.iterrows():
        me, gh = met['mAP@20'], ghd[session_no]
        if not np.isclose(me, gh):
            print(f'session={session_no} nt={me:.5f} gh={gh:.5f}')

    # Known (excusable) discrepancies:
    #   3513781251

    print(mets)
    print(met_mean)


def predefined_data():
    # This data is from the following excellent discussion on metric calculation for rec sys:
    #  https://ils.unc.edu/courses/2013_spring/inls509_001/lectures/10-EvaluationMetrics.pdf

    data = []
    for ii, ll in enumerate(
            [1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1]
    ):
        data.append(dict(
            group_id=-1,
            user_id=-1,
            recipe_id=-1,
            score=-ii,
            label=ll,
        ))

    return pd.DataFrame(data)


def generate_random_data(n_users=20, n_items=100, n_interactions_per_user=20, random_state=1234):
    rng = np.random.RandomState(random_state)

    data = []
    for user_id in range(n_users):
        n_user_interactions = rng.randint(1, n_interactions_per_user)
        for recipe_id in rng.choice(n_items, n_user_interactions):
            data.append(dict(
                group_id=user_id,
                user_id=user_id,
                item_id=recipe_id,
                score=rng.normal(),
                label=int(rng.rand() > 0.75),
            ))

    return pd.DataFrame(
        data
    )


if __name__ == '__main__':
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    _test_gh_data()
