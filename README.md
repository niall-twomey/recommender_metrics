# Recommender Metrics

## Introduction

This is a very basic package that implements recommendation metrics. The library was written with the intent of being 
simple and straightforward to understand/maintain/expand instead of optimising for efficiency.  

The main functions of the library are: 

 - `recommender_metrics.metrics.calculate_metrics` 
 - `recommender_metrics.metrics.calculate_metrics_from_dataframe` 

and their docstrings outline their use in detail. 

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

