import sys
from time import time
from unittest import TestCase

import numpy as np

import recommender_metrics


class Timer(object):
    def __enter__(self):
        self.start = time()
        return self

    def __exit__(self, *args):
        self.end = time()
        self.interval = self.end - self.start


def time_func(func, extra=None, *args, **kwargs):
    try:
        with Timer() as t:
            output = func(*args, **kwargs)
    finally:
        calling_func = sys._getframe(1).f_code.co_name
        print(f"Function {calling_func} {extra or ''} took {t.interval}s to execute")
    return output


class BasicTests(TestCase):
    def dict_vals_all_close(self, d1, d2):
        for kk, vv in d1.items():
            self.assertAlmostEqual(vv, d2[kk], msg=f"With key: {kk}\nsingle: {d1}\n multi: {d2}")

    def test_numpy_input_default_k_list(self):
        rng = np.random.RandomState(1234)

        metrics = time_func(
            func=recommender_metrics.calculate_metrics,
            group_ids=rng.randint(0, 10, 100),
            scores=rng.normal(0, 1, 100),
            labels=rng.rand(100) > 0.8,
        )

        self.dict_vals_all_close(
            {
                "mAP@1": 0.3,
                "precision@1": 0.3,
                "recall@1": 0.21666666666666665,
                "mAP@5": 0.41500000000000004,
                "precision@5": 0.18,
                "recall@5": 0.5166666666666667,
                "mAP@10": 0.35478174603174606,
                "precision@10": 0.2088888888888889,
                "recall@10": 0.9666666666666666,
                "mAP@20": 0.35613756613756614,
                "precision@20": 0.20297979797979798,
                "recall@20": 1.0,
            },
            metrics,
        )

    def test_numpy_input_k_int(self):
        rng = np.random.RandomState(1234)

        metrics = time_func(
            func=recommender_metrics.calculate_metrics,
            group_ids=rng.randint(0, 10, 100),
            scores=rng.normal(0, 1, 100),
            labels=rng.rand(100) > 0.8,
            k_list=1,
        )

        self.dict_vals_all_close(
            {"mAP@1": 0.3, "precision@1": 0.3, "recall@1": 0.21666666666666665}, metrics,
        )

    def test_multi_threads(self):
        pass
