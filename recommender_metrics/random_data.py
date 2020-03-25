import numpy as np
import pandas as pd

__all__ = [
    'generate_random_data',
    'predefined_data',
]


def predefined_data():
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
