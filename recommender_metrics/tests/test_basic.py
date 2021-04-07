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
    dict(
        kwargs=dict(zip(("group_ids", "scores", "labels"), recommender_metrics.generate_random_data(n_users=50000))),
        name="Large sample",
        targets={
            "mAP@1": 0.25338,
            "precision@1": 0.25338,
            "recall@1": 0.25912677994227995,
            "mAP@5": 0.40336191666666665,
            "precision@5": 0.25206833333333345,
            "recall@5": 0.591021176046176,
            "mAP@10": 0.3937965079805996,
            "precision@10": 0.251275753968254,
            "recall@10": 0.8339520981240983,
            "mAP@20": 0.3793676925719221,
            "precision@20": 0.2516421896227267,
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

    def test_numpy_input_with_weights(self):
        for case in TEST_CASE_LIST:
            metrics = time_func(
                recommender_metrics.calculate_metrics,
                extra=case.get("name") + "weighted",
                weights=np.ones(len(case["kwargs"]["group_ids"])) / 10,
                **case.get("kwargs"),
            )
            self.dict_vals_all_close(case.get("targets"), metrics, desc=case.get("name"))

    def test_numpy_input_k_int(self):
        for k in [1, 5, 10, 20]:
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

    def test_empty(self):
        target = {
            "mAP@1": 0.2631578947368421,
            "precision@1": 0.2631578947368421,
            "recall@1": 0.07631578947368421,
            "mAP@5": 0.5046783625730993,
            "precision@5": 0.3140350877192983,
            "recall@5": 0.45743525480367586,
            "mAP@10": 0.4538314536340852,
            "precision@10": 0.2941520467836258,
            "recall@10": 0.6738512949039264,
            "mAP@20": 0.424754158736318,
            "precision@20": 0.33355881010883454,
            "recall@20": 1.0,
        }

        groups, scores, labels = recommender_metrics.generate_random_data()
        metrics = recommender_metrics.calculate_metrics(
            group_ids=groups, scores=scores, labels=labels, remove_empty=True
        )
        self.dict_vals_all_close(target=target, pred=metrics, desc=f"Removal of empty group labels")

    def test_gist_case(self):
        # https://gist.github.com/bwhite/3726239#gistcomment-2852580

        res = recommender_metrics.calculate_metrics(
            k_list=[1, 2, 3],
            group_ids=np.asarray([1, 1, 1, 2, 2, 2, 3, 3, 3]),
            scores=np.asarray([1, 2, 3, 1, 2, 3, 1, 2, 3]),
            labels=np.asarray([1, 1, 1, 0, 1, 1, 0, 0, 1]),
            reduce=False,
        )

        target = {
            "mAP@1": np.asarray([1.0, 1.0, 1.0]),
            "precision@1": np.asarray([1.0, 1.0, 1.0]),
            "recall@1": np.asarray([0.33333333, 0.5, 1.0]),
            "mAP@2": np.asarray([1.0, 1.0, 1.0]),
            "precision@2": np.asarray([1.0, 1.0, 0.5]),
            "recall@2": np.asarray([0.66666667, 1.0, 1.0]),
            "mAP@3": np.asarray([1.0, 1.0, 1.0]),
            "precision@3": np.asarray([1.0, 0.66666667, 0.33333333]),
            "recall@3": np.asarray([1.0, 1.0, 1.0]),
        }

        for key, val in target.items():
            assert np.allclose(val, res[key])

    def test_incremental(self):
        from tqdm import tqdm

        for data in TEST_CASE_LIST:
            kwargs = data["kwargs"]

            ascending = kwargs.get("ascending", False)
            remove_empty = kwargs.get("remove_empty", False)

            mets = recommender_metrics.IncrementalMetrics(ascending=ascending, verbose=False, remove_empty=remove_empty)
            groups = recommender_metrics.group_score_and_labelled_data(
                group_ids=kwargs["group_ids"],
                scores=kwargs["scores"],
                labels=kwargs["labels"],
                ascending=ascending,
                verbose=False,
                remove_empty=remove_empty,
            )
            for group_id, group in tqdm(groups.items(), total=len(groups), desc=f"incremental {data['name']}"):
                mets.append_group(scores=group["scores"], labels=group["labels"], weight=1)

            # This method below is correct and produces the same results as the above. however, it is *very* slow in
            # slicing the data. If all data are available and need to resort to slicing, use the above approach
            # instead. This is not how this scenario is intended to operate.
            # mets = recommender_metrics.IncrementalMetrics(ascending=ascending,verbose=False,remove_empty=remove_empty)
            # for group_id in tqdm(np.unique(kwargs["group_ids"])):
            #     inds = np.where(np.asarray(kwargs["group_ids"]) == group_id)[0]
            #     scores = np.asarray(kwargs["scores"])[inds]
            #     labels = np.asarray(kwargs["labels"])[inds]
            #     mets.append_group(scores=scores, labels=labels, weight=1.0)
            # self.dict_vals_all_close(data["targets"], mets.resolve(), data["name"])

    def test_auroc_ndcg(self):
        # groups, scores, labels = recommender_metrics.generate_random_data()
        # metrics = recommender_metrics.calculate_metrics(
        #     group_ids=groups, scores=scores, labels=labels, remove_empty=True, metrics=["ndcg", "auroc"], k_list=[20]
        # )
        # target = None
        print("test_auroc_ndcg not implemented...")
        # self.dict_vals_all_close(target=target, pred=metrics, desc=f"Removal of empty group labels")

    def test_multi_threads(self):
        pass
