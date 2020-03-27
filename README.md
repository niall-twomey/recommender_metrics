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
from recommender_metrics import generate_random_data 

data = generate_random_data()
print('Data')
print(data.head())
print()

metrics, metrics_averaged = calculate_metrics(
    group_ids=data['group_id'], 
    scores=data['score'], 
    labels=data['label']
)
print('Metrics:')
print(metrics_averaged)
print()
```

This should print outputs like these below following:

```
Data
   group_id  user_id  item_id     score  label
0         0        0       83  1.056970      0
1         0        0       38 -0.590180      0
2         0        0       53 -0.387864      0
3         0        0       76 -0.046539      0
4         0        0       24  0.515387      1

Calculating performance metrics over group_id: 100%|██████████████| 20/20 [00:00<00:00, 100.95it]

Metrics:
mAP@1          0.050000
precison@1     0.050000
recall@1       0.062500
mAP@5          0.297361
precison@5     0.225000
recall@5       0.463333
mAP@10         0.327321
precison@10    0.210992
recall@10      0.788333
mAP@20         0.309007
precison@20    0.226823
recall@20      1.000000
dtype: float64
```

Note, that in this case the `group_id` columns is an integer, but in reality it can be any hashable type (e.g. tuple). 

The metrics can be calculated directly from a dataframe as follows: 

```python
from recommender_metrics import calculate_metrics_from_dataframe 
from recommender_metrics import generate_random_data 
data = generate_random_data()
print(calculate_metrics_from_dataframe(data))
```

which will output the same metrics: 

```
mAP@1          0.050000
precison@1     0.050000
recall@1       0.062500
mAP@5          0.297361
precison@5     0.225000
recall@5       0.463333
mAP@10         0.327321
precison@10    0.210992
recall@10      0.788333
mAP@20         0.309007
precison@20    0.226823
recall@20      1.000000
dtype: float64
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
    print(f'Dataframe shape={df.shape} with {df.group_id.nunique()} unique groups\n')

    start = datetime.now()
    _, single = calculate_metrics_from_dataframe(df)
    print(f'Single-threaded timing ({datetime.now() - start})')

    start = datetime.now()
    _, multi = calculate_metrics_from_dataframe(df, n_threads=n_threads)
    print(f'Multi-threaded timing ({datetime.now() - start})')

    print(f'All the same: {(single == multi).all()}')
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
Dataframe shape=(20, 5) with 1 unique groups

Calculating performance metrics over group_id: 100%|██████████| 1/1 [00:00<00:00, 112.02it/s]
Single-threaded timing (0:00:00.020167)
Constructing arguments: 100%|██████████| 4/4 [00:00<00:00, 953.68it/s]
Computing metrics: 100%|██████████| 4/4 [00:00<00:00, 291.27it/s]
Multi-threaded timing (0:00:00.133939)
All the same: True

Dataframe shape=(255, 5) with 20 unique groups
Calculating performance metrics over group_id: 100%|██████████| 20/20 [00:00<00:00, 111.66it/s]
Constructing arguments:   0%|          | 0/4 [00:00<?, ?it/s]
Single-threaded timing (0:00:00.201712)
Constructing arguments: 100%|██████████| 4/4 [00:00<00:00, 434.98it/s]
Computing metrics: 100%|██████████| 80/80 [00:00<00:00, 358.48it/s]
Multi-threaded timing (0:00:00.364281)
All the same: True

Dataframe shape=(49615, 5) with 1000 unique groups
Calculating performance metrics over group_id: 100%|██████████| 1000/1000 [00:08<00:00, 124.09it/s]
Single-threaded timing (0:00:08.639233)
Constructing arguments: 100%|██████████| 4/4 [00:00<00:00, 29.59it/s]
Computing metrics: 100%|██████████| 4000/4000 [00:02<00:00, 1961.20it/s]
Multi-threaded timing (0:00:02.747941)
All the same: True
```

The takeaway message from this is that when the data is small (the first two examples) the single threaded version 
is actually more efficient (although the loss in efficienc won't be noticable). I presume this is due to the setup 
that the multiprocessing library needs to do before joining the pools. In contrast, when the size of the groups grows
the gains of multiprocessing are much clearer; concretely evaluation takes 2.75 seconds rather than 8.6 seconds.  

