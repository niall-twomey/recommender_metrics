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


rng = np.random.RandomState(1234)

TEST_CASE_LIST = [
    dict(
        kwargs=dict(group_ids=rng.randint(0, 10, 100), scores=rng.normal(0, 1, 100), labels=rng.rand(100) > 0.8),
        name="Random numpy",
        targets={
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
    ),
    dict(
        kwargs=dict(zip(("group_ids", "scores", "labels", "ascending"), recommender_metrics.search_data() + (True,))),
        name="Search data ascending",
        targets={
            "mAP@1": 1.0,
            "precision@1": 1.0,
            "recall@1": 0.1,
            "mAP@5": 0.8041666666666667,
            "precision@5": 0.8,
            "recall@5": 0.4,
            "mAP@10": 0.8121315192743763,
            "precision@10": 0.7,
            "recall@10": 0.7,
            "mAP@20": 0.7555050505050506,
            "precision@20": 0.5,
            "recall@20": 1.0,
        },
    ),
    dict(
        kwargs=dict(zip(("group_ids", "scores", "labels"), recommender_metrics.generate_random_data())),
        name="Bigger random sample",
        targets={
            "mAP@1": 0.25,
            "precision@1": 0.25,
            "recall@1": 0.12250000000000001,
            "mAP@5": 0.47944444444444434,
            "precision@5": 0.2983333333333334,
            "recall@5": 0.4845634920634921,
            "mAP@10": 0.4311398809523809,
            "precision@10": 0.2794444444444445,
            "recall@10": 0.69015873015873,
            "mAP@20": 0.4035164507995021,
            "precision@20": 0.3168808696033928,
            "recall@20": 1.0,
        },
    ),
]


class BasicTests(TestCase):
    def dict_vals_all_close(self, target, pred, desc):
        for kk, vv in target.items():
            self.assertAlmostEqual(vv, pred[kk], msg=f"Error with key ({desc}): {kk}\ntarget: {target}\n  pred: {pred}")

    def test_numpy_input_defaults(self):
        for case in TEST_CASE_LIST:
            metrics = time_func(recommender_metrics.calculate_metrics, extra=case.get("name"), **case.get("kwargs"))
            self.dict_vals_all_close(case.get("targets"), metrics, desc=case.get("name"))

    def test_numpy_input_k_int(self):
        for k in recommender_metrics.validate_k_list(None):
            for case in TEST_CASE_LIST:
                metrics = time_func(
                    recommender_metrics.calculate_metrics, extra=case.get("name"), k_list=k, **case.get("kwargs")
                )
                self.dict_vals_all_close(
                    {kk: vv for kk, vv in case.get("targets").items() if kk.endswith(f"@{k}")},
                    metrics,
                    desc=case.get("name"),
                )

    def test_metrics(self):
        for metric in recommender_metrics.DEFAULT_METRICS:
            for case in TEST_CASE_LIST:
                metrics = time_func(
                    recommender_metrics.calculate_metrics, extra=case.get("name"), metrics=metric, **case.get("kwargs")
                )
                self.dict_vals_all_close(
                    {kk: vv for kk, vv in case.get("targets").items() if kk.startswith(metric)},
                    metrics,
                    desc=case.get("name"),
                )

    def test_multi_threads(self):
        pass
