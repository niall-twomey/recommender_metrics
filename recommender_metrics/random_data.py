import random

__all__ = [
    "generate_random_data",
    "search_data",
]


def search_data():
    # This data is from the following excellent discussion on metric calculation for rec sys:
    #  https://ils.unc.edu/courses/2013_spring/inls509_001/lectures/10-EvaluationMetrics.pdf

    groups = []
    positions = []
    labels = []

    for ii, ll in enumerate([1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1]):
        labels.append(ll)
        positions.append(ii)
        groups.append(1)

    return groups, positions, labels


def generate_random_data(n_users=20, n_items=100, n_interactions_per_user=20, random_seed=1234):
    import uuid

    random.seed(random_seed)

    groups = []
    scores = []
    labels = []

    for user_id in range(n_users):
        user_id = str(uuid.uuid1())
        n_user_interactions = random.randint(1, n_interactions_per_user)
        for ii in range(n_user_interactions):
            score = random.random()
            groups.append(user_id)
            scores.append(score)
            labels.append(int(random.random() > 0.75))

    return groups, scores, labels
