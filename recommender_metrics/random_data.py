import numpy as np
import pandas as pd

__all__ = [
    'generate_random_data'
]


def generate_random_data(n_users=20, n_items=100, n_interactions_per_user=10, random_state=1234):
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
    ).sort_values(
        by=['group_id', 'score'],
        ascending=[True, False]
    )
