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

metrics, metrics_averaged = calculate_metrics(
    group_ids=np.random.randint(0, 10, 100),
    scores=np.random.normal(0, 1, 100),
    labels=np.random.rand(100) > 0.8
)

print(json.dumps(metrics_averaged, indent=2))
```

Which gives the following output: 

```
Calculating performance metrics over group_id: 100%|██████████| 10/10 [00:00<00:00, 89.70it/s]
{
  "mAP@1": 0.2,
  "precison@1": 0.2,
  "recall@1": 0.18333333333333332,
  "mAP@5": 0.44833333333333336,
  "precison@5": 0.2,
  "recall@5": 0.7166666666666666,
  "mAP@10": 0.41158730158730156,
  "precison@10": 0.1577777777777778,
  "recall@10": 0.8583333333333332,
  "mAP@20": 0.38844530469530475,
  "precison@20": 0.17391414141414144,
  "recall@20": 1.0
}
```

This works with `ascending`-like scores (that you might get with search data). The following example 
is duplicated from (here)[https://ils.unc.edu/courses/2013_spring/inls509_001/lectures/10-EvaluationMetrics.pdf]. 


```python
from recommender_metrics import calculate_metrics
from recommender_metrics import search_data
import json

data = search_data()
print('Data')
print(data.head())
print()

metrics = calculate_metrics(
    group_ids=data['group_id'].values,
    scores=data['score'].values,
    labels=data['label'].values,
    ascending=True
)
print('Metrics:')
print(json.dumps(metrics, indent=2))
```

And it gives the following output

```
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
Data
   group_id  user_id  item_id     score  label
0         0        0       14  0.007491      1
1         0        0       74  0.034926      0
2         0        0       12  0.766481      0
3         0        0        3  0.986688      0
4         0        0       82  0.623281      0
Calculating performance metrics over group_id: 100%|██████████| 20/20 [00:00<00:00, 97.07it/s]
Metrics:
{
  "mAP@1": 0.35,
  "precison@1": 0.35,
  "recall@1": 0.12380952380952381,
  "mAP@5": 0.5009722222222222,
  "precison@5": 0.2800000000000001,
  "recall@5": 0.43892857142857145,
  "mAP@10": 0.45625,
  "precison@10": 0.2656547619047619,
  "recall@10": 0.7352380952380952,
  "mAP@20": 0.43771204652167156,
  "precison@20": 0.2812859012762263,
  "recall@20": 1.0
}
```

Note, that in this case the `group_id` columns is an integer, but in reality it can be any hashable type (e.g. tuple). 

The metrics can be calculated directly from a dataframe as follows: 

```python
from recommender_metrics import calculate_metrics_from_dataframe 
from recommender_metrics import generate_random_data 
import json 
data = generate_random_data()
print(json.dumps(calculate_metrics_from_dataframe(data), indent=2))
```

which will output the same metrics: 

```
{
  "mAP@1": 0.35,
  "precison@1": 0.35,
  "recall@1": 0.12380952380952381,
  "mAP@5": 0.5009722222222222,
  "precison@5": 0.2800000000000001,
  "recall@5": 0.43892857142857145,
  "mAP@10": 0.45625,
  "precison@10": 0.2656547619047619,
  "recall@10": 0.7352380952380952,
  "mAP@20": 0.43771204652167156,
  "precison@20": 0.2812859012762263,
  "recall@20": 1.0
}
```

By default the `calculate_metrics_from_dataframe` function requires that the input dataframe has columns `group_id`, 
`label` and `score`. However, these can be specified with the optional `group_col`, `label_col` and `score_col` 
arguments. 
