# Recommender Metrics

## Introduction

This is a very basic package that implements recommendation metrics. The library was written with the intent of being 
simple and straightforward to understand/maintain/expand instead of optimising for efficiency.  

The main functions of the library are: 

 - `recommender_metrics.metrics.calculate_metrics` 
 - `recommender_metrics.metrics.calculate_metrics_from_dataframe` 

and their docstrings outline their use in detail. 

## Installation 

### Pipenv

Add the following to the `Pipfile`: 

```
recommender_metrics = {git = "https://github.com/niall-twomey/recommender_metrics.git",ref = "master"}
```

### Pip

Execute the following command 

```shell script
pip install git+https://github.com/niall-twomey/recommender_metrics.git@master
```



## Basic examples 

The examples below demonstrate how it can be used: 

```python
from recommender_metrics import calculate_metrics
import numpy as np
import json

rng = np.random.RandomState(1234)
metrics = calculate_metrics(
    group_ids=rng.randint(0, 10, 100),
    scores=rng.normal(0, 1, 100),
    labels=rng.rand(100) > 0.8
)

print(json.dumps(metrics, indent=2))
print('\n\n\n')
```

Which gives the following output: 

```
Grouping data before evaluation: 100%|███████████████████████████| 10/10 [00:00<00:00, 54400.83it/s]
Evaluating performance: 100%|█████████████████████████████████████| 10/10 [00:00<00:00, 9688.85it/s]
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
  "recall@20": 1.0
}
```

This works with `ascending`-like scores (that you might get with search data). The following example 
is duplicated from (here)[https://ils.unc.edu/courses/2013_spring/inls509_001/lectures/10-EvaluationMetrics.pdf]. 


```python
from recommender_metrics import calculate_metrics
from recommender_metrics import search_data
import json

groups, positions, labels = search_data()
print('Data:')
print('     groups:', groups)
print('  positions:', positions)
print('     labels:', labels)
print()

metrics = calculate_metrics(
    group_ids=groups,
    scores=positions,
    labels=labels,
    ascending=True
)
print('Metrics:')
print(json.dumps(metrics, indent=2))
print('\n\n\n')
```

And it gives the following output

```
Data:
     groups: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  positions: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
     labels: [1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1]

Grouping data before evaluation: 100%|██████████████████████████████| 1/1 [00:00<00:00, 5050.40it/s]
Evaluating performance: 100%|███████████████████████████████████████| 1/1 [00:00<00:00, 5125.41it/s]
Metrics:
{
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
  "recall@20": 1.0
}
```

Other pre-defined data can be evaluated as follows 

```python
from recommender_metrics import calculate_metrics 
from recommender_metrics import generate_random_data 
import json 

data = generate_random_data()
print('Data')
print(data.head())
print()

metrics = calculate_metrics(
    group_ids=data['group_id'], 
    scores=data['score'], 
    labels=data['label']
)
print('Metrics:')
print(json.dumps(metrics, indent=2))
```

This should print outputs like these below following:

```
Data:
  #groups: 255
  #scores: 255
  #labels: 255

Grouping data before evaluation: 100%|███████████████████████████| 20/20 [00:00<00:00, 63119.70it/s]
Evaluating performance: 100%|████████████████████████████████████| 20/20 [00:00<00:00, 10215.06it/s]
Metrics:
{
  "mAP@1": 0.35,
  "precision@1": 0.35,
  "recall@1": 0.12380952380952381,
  "mAP@5": 0.5009722222222222,
  "precision@5": 0.2800000000000001,
  "recall@5": 0.43892857142857145,
  "mAP@10": 0.45625,
  "precision@10": 0.2656547619047619,
  "recall@10": 0.7352380952380952,
  "mAP@20": 0.43771204652167156,
  "precision@20": 0.2812859012762263,
  "recall@20": 1.0
}
```

Note, that in this case the `group_id` columns is an integer, but in reality it can be any hashable type (e.g. tuple). 

The metrics can be calculated directly from a dataframe with the `calculate_metrics_from_dataframe`. By default this
function requires that the input dataframe has columns `group_id`, `label` and `score`. However, if these are called
something different, their names can be specified with the optional `group_col`, `label_col` and `score_col` arguments. 
