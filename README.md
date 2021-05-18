# Recommender Metrics



## Introduction

This is a very basic package with implementions of several popular recommendation metrics. The library was written with the intent of being simple and straightforward to understand/maintain/expand instead of optimising for efficiency.  

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


## Basic example 

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

The TQDM progress bar can be suppressed by adding the flag `verbose=False` into the `calculate_metrics` function. 

Other examples can be found in `recommender_metrics/examples/examples.py` that demonstrate the broader usage and interfaces to the library. 

## Development 

Styling is achieved with [black](https://github.com/psf/black) and [flake8](https://gitlab.com/pycqa/flake8) with minor modifications in the `~/.flake8` and `~/.pyproject.toml` files. These styles are enforced with [pre-commit](https://pre-commit.com/) tool. Before development ensure that [PyEnv](https://github.com/pyenv/pyenv) and [Pipenv](https://github.com/pypa/pipenv) are installed. 

```shell script
pyenv install 3.6.10
pipenv install --dev --python 3.6
pipenv run pre-commit install 
```

This will install pre-commit tools so that black, flake8, and reorder python imports will be executed before committing. Only when these all succeed will the commit actually go ahead. 

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
