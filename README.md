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

Some pre-defined data can be evaluated as follows 

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

## Multiprocessing 

A super basic multiprocessing wrapper around this code has been done. This can simply be acquired by setting the 
`n_threads` argument in the main evaluation functions to a value larger than 1. The following code snipped illustrates 
a comparison between the two approaches 

```python
from recommender_metrics import random_data, calculate_metrics_from_dataframe
from datetime import datetime


def single_multi_thread_test(df, n_threads=5):
    print(f'Dataframe shape={df.shape} with {df.group_id.nunique()} unique groups')

    start = datetime.now()
    single = calculate_metrics_from_dataframe(df)
    print(f'TIME FOR SINGLE THREAD ({datetime.now() - start})')

    start = datetime.now()
    multi = calculate_metrics_from_dataframe(df, n_threads=n_threads)
    print(f'TIME FOR {n_threads} THREADS ({datetime.now() - start})')

    print(f'All the same: {(single.items() == multi.items())}')
    print()


single_multi_thread_test(random_data.predefined_data())
single_multi_thread_test(random_data.generate_random_data())
single_multi_thread_test(random_data.generate_random_data(
    n_users=1000,
    n_items=10000,
    n_interactions_per_user=100,
))
```

And on my laptop I get the following outputs

```
Calculating performance metrics over group_id: 100%|██████████| 1/1 [00:00<00:00, 110.72it/s]
Constructing arguments: 100%|██████████| 4/4 [00:00<00:00, 1027.51it/s]
TIME FOR SINGLE THREAD (0:00:00.020452)
Computing metrics: 100%|██████████| 4/4 [00:00<00:00, 286.79it/s]
TIME FOR 5 THREADS (0:00:00.130910)
All the same: True

Dataframe shape=(255, 5) with 20 unique groups
Calculating performance metrics over group_id: 100%|██████████| 20/20 [00:00<00:00, 116.21it/s]
TIME FOR SINGLE THREAD (0:00:00.192167)
Constructing arguments: 100%|██████████| 4/4 [00:00<00:00, 573.62it/s]
Computing metrics: 100%|██████████| 80/80 [00:00<00:00, 468.03it/s]
TIME FOR 5 THREADS (0:00:00.246602)
All the same: True

Dataframe shape=(49615, 5) with 1000 unique groups
Calculating performance metrics over group_id: 100%|██████████| 1000/1000 [00:08<00:00, 119.20it/s]
TIME FOR SINGLE THREAD (0:00:08.947291)
Constructing arguments: 100%|██████████| 4/4 [00:00<00:00, 28.59it/s]
Computing metrics: 100%|██████████| 4000/4000 [00:02<00:00, 1827.97it/s]
TIME FOR 5 THREADS (0:00:02.983843)
All the same: True
```

The takeaway message from this is that when the data is small (the first two examples) the single threaded version 
is actually more efficient (although the loss in efficienc won't be noticable). I presume this is due to the setup 
that the multiprocessing library needs to do before joining the pools. In contrast, when the size of the groups grows
the gains of multiprocessing are much clearer; concretely evaluation takes 2.75 seconds rather than 8.6 seconds.  

