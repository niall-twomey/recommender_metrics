# Recommender Metrics



## Introduction

This is a very basic package that implements recommendation metrics. The library was written with the intent of being 
simple and straightforward to understand/maintain/expand instead of optimising for efficiency.  

The main functions of the library are: 

 - `recommender_metrics.calculate_metrics` 
 - `recommender_metrics.calculate_metrics_from_dataframe` 

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
```

Which gives the following output: 

```
Evaluating performance: 100%|█████████████████████████████████████| 10/10 [00:00<00:00, 8331.95it/s]
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
is duplicated from [here](https://ils.unc.edu/courses/2013_spring/inls509_001/lectures/10-EvaluationMetrics.pdf). 


```python
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
```

And it gives the following output

```
Data:
     groups: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  positions: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
     labels: [1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1]

Evaluating performance: 100%|███████████████████████████████████████| 1/1 [00:00<00:00, 3659.95it/s]
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

groups, scores, labels = generate_random_data()
print("Data:")
print("  #groups:", len(groups))
print("  #scores:", len(scores))
print("  #labels:", len(labels))
print()

metrics = calculate_metrics(group_ids=groups, scores=scores, labels=labels,)
print("Metrics:")
print(json.dumps(metrics, indent=2))
print("\n\n\n")
```

This should print outputs like these below following:

```
Data:
  #groups: 250
  #scores: 250
  #labels: 250

Evaluating performance: 100%|█████████████████████████████████████| 20/20 [00:00<00:00, 9519.53it/s]
Metrics:
{
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
  "recall@20": 1.0
}
```

Note, that in this case the `group_id` columns is an integer, but in reality it can be any hashable type (e.g. tuple). 

The metrics can be calculated directly from a dataframe with the `calculate_metrics_from_dataframe`. By default this
function requires that the input dataframe has columns `group_id`, `label` and `score`. However, if these are called
something different, their names can be specified with the optional `group_col`, `label_col` and `score_col` arguments. 



## Development 

Styling is achieved with [black](https://github.com/psf/black) and [flake8](https://gitlab.com/pycqa/flake8) with minor 
modifications in the `~/.flake8` and `~/.pyproject.toml` files. These styles are enforced with 
[pre-commit](https://pre-commit.com/) tool. Before development ensure that [PyEnv](https://github.com/pyenv/pyenv) and 
[Pipenv](https://github.com/pypa/pipenv) are installed. 

```shell script
pyenv install 3.6.10
pipenv install --dev --python 3.6
pipenv run pre-commit install 
```

This will install pre-commit tools so that black, flake8, and reorder python imports will be executed before committing. 
Only when these all succeed will the commit actually go ahead. 

To run tests/build/install: 

```shell script
python setup.py test 
python setup.py build 
python setup.py install 
```

If not in the pip environment, instead use 

```shell script
pipenv run python setup.py test
pipenv run python setup.py build
pipenv run python setup.py install
```
