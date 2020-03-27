import pandas as pd
import random

__all__ = [
    'generate_random_data',
    'predefined_data',
]


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


def generate_random_data(n_users=20, n_items=100, n_interactions_per_user=20, random_seed=1234):
    random.seed(random_seed)

    data = []
    for user_id in range(n_users):
        n_user_interactions = random.randint(1, n_interactions_per_user)
        for ii in range(n_user_interactions):
            item_id = random.randint(0, n_items)
            score = random.random()
            data.append(dict(
                group_id=user_id,
                user_id=user_id,
                item_id=item_id,
                score=score,
                label=int(random.random() > 0.75),
            ))

    return pd.DataFrame(
        data
    )
