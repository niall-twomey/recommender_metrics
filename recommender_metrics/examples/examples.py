try:
    import pandas as pd

    pd.set_option("display.max_rows", 500)
    pd.set_option("display.max_columns", 500)
    pd.set_option("display.width", 1000)
except ImportError:
    pass


def basic_usage_example_1():
    from recommender_metrics import calculate_metrics
    import numpy as np
    import json

    print("Running example1")

    rng = np.random.RandomState(1234)
    metrics = calculate_metrics(
        group_ids=rng.randint(0, 10, 100),
        scores=rng.normal(0, 1, 100),
        labels=rng.rand(100) > 0.8,
    )
    print(json.dumps(metrics, indent=2))
    print("\n\n\n")


def basic_usage_example_2():
    from recommender_metrics import calculate_metrics
    from recommender_metrics import generate_random_data
    import json

    groups, scores, labels = generate_random_data()
    print("Data:")
    print("  #groups:", len(groups))
    print("  #scores:", len(scores))
    print("  #labels:", len(labels))
    print()

    metrics = calculate_metrics(group_ids=groups, scores=scores, labels=labels)
    print("Metrics:")
    print(json.dumps(metrics, indent=2))
    print("\n\n\n")


def basic_usage_ascending_scores():
    from recommender_metrics import calculate_metrics
    from recommender_metrics import search_data
    import json

    print("Running example2")

    groups, positions, labels = search_data()
    print("Data:")
    print("     groups:", groups)
    print("  positions:", positions)
    print("     labels:", labels)
    print()

    metrics = calculate_metrics(group_ids=groups, scores=positions, labels=labels, ascending=True)
    print("Metrics:")
    print(json.dumps(metrics, indent=2))
    print("\n\n\n")


def basic_usage_group_filttering():
    from recommender_metrics import calculate_metrics
    from recommender_metrics import generate_random_data
    import json

    print("Running example3")

    groups, scores, labels = generate_random_data()
    print("Data:")
    print("  #groups:", len(groups))
    print("  #scores:", len(scores))
    print("  #labels:", len(labels))
    print()

    metrics = calculate_metrics(group_ids=groups, scores=scores, labels=labels, remove_empty=True)
    print("Metrics:")
    print(json.dumps(metrics, indent=2))
    print("\n\n\n")


def basic_usage_large_dataset():
    from recommender_metrics import calculate_metrics
    from recommender_metrics import generate_random_data
    import json

    groups, scores, labels = generate_random_data(n_users=50000)
    print("Larger data:")
    print("  #groups:", len(groups))
    print("  #scores:", len(scores))
    print("  #labels:", len(labels))
    print()

    metrics = calculate_metrics(group_ids=groups, scores=scores, labels=labels)
    print("Metrics:")
    print(json.dumps(metrics, indent=2))
    print("\n\n\n")


def basic_usage_custom_metrics_k():
    from recommender_metrics import calculate_metrics
    from recommender_metrics import generate_random_data
    import json

    print("Running example4")

    groups, scores, labels = generate_random_data()
    print("Data:")
    print("  #groups:", len(groups))
    print("  #scores:", len(scores))
    print("  #labels:", len(labels))
    print()

    metrics = calculate_metrics(
        group_ids=groups,
        scores=scores,
        labels=labels,
        k_list=[1, 2, 4, 8, 16],
        metrics=["mAP", "precision", "recall", "ndcg", "auroc"],
    )
    print("Metrics:")
    print(json.dumps(metrics, indent=2))
    print("\n\n\n")


if __name__ == "__main__":
    basic_usage_example_1()
    basic_usage_example_2()
    basic_usage_ascending_scores()
    basic_usage_group_filttering()
    basic_usage_large_dataset()
    basic_usage_custom_metrics_k()
