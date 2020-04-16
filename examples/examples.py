try:
    import pandas as pd

    pd.set_option("display.max_rows", 500)
    pd.set_option("display.max_columns", 500)
    pd.set_option("display.width", 1000)
except ImportError:
    pass


def example1():
    from recommender_metrics import calculate_metrics
    import numpy as np
    import json

    rng = np.random.RandomState(1234)
    metrics = calculate_metrics(
        group_ids=rng.randint(0, 10, 100), scores=rng.normal(0, 1, 100), labels=rng.rand(100) > 0.8,
    )
    print(json.dumps(metrics, indent=2))
    print("\n\n\n")


def example2():
    from recommender_metrics import calculate_metrics
    from recommender_metrics import search_data
    import json

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


def example3():
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


if __name__ == "__main__":
    example1()
    example2()
    example3()
