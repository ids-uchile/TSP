# TSP
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


Data-driven Mutual Information Estimator based on regularized tree-structured partitions.

## Installation

This package now counts with [`uv`](https://docs.astral.sh/uv/), this makes installation of the dependencies very easy. If you have `uv` already installed, you just need to run:

```
uv sync
```

This command will automatically generate a virtual environment with all that is needed for correctly executing the `TSP` module

## Usage
- Python package `TSP.py` example of use.
```Python
import numpy as np

# Import the package
from TSP import TSP

# Initialize NumPy arrays of shape (n_samples, dim) with random 
# variables samples in Fortran-contiguous memory order
X = np.array([...], order='F').reshape(-1, 1)
Y = np.array([...], order='F').reshape(-1, 1)

# Initialize the TSP with its parameters
l_bn = 0.167
w_bn = 5e-2
reg_factor = 2.5e-5
tsp = TSP(l_bn, w_bn, reg_factor)

# Set observations to the estimator
tsp.grow(X, Y)

# Get TSP size and estimated mutual information
size = tsp.size()
emi = tsp.emi()

# Regularize the TSP mutual information estimator
tsp.regularize()

# Get the regularized TSP size and estimated mutual information
reg_size = tsp.reg_size()
reg_emi = tsp.reg_emi()

# Visualize the space partitions in a 2D scenario
tsp.visualize(X, Y)
```

## Reference
[[1](https://sail.usc.edu/publications/files/silva_tit_2012.pdf)] Silva, J. F., & Narayanan, S. (2012). **Complexity-Regularized Tree-Structured Partition for Mutual Information Estimation**. *IEEE Transactions on Information Theory, 58*(3), 1940-1952.

