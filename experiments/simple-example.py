import numpy as np
import sys
sys.path.append('../src')

# Import the package
from TSP.core import TSP

# Initialize NumPy arrays of shape (n_samples, dim) with random 
# variables samples in Fortran-contiguous memory order
X = np.array([-0.2, 1.5, 4.0, 2.0, -2.4, 2.0, -0.4, -4.0, 2.2, 3.0], order='F').reshape(-1, 1)
Y = np.array([1.2, -4.5, 0.0, 4.0, 3.7, -1.0, 2.4, -1.2, 6.2, 6.0], order='F').reshape(-1, 1)

samples_list = [3000000]#range(1,10000)

for n_samples in samples_list:

    # Generate a bivariate Gaussian distribution
    #n_samples = 100000  # Number of unique samples
    mean = [0, 0]  # Mean of the distribution
    cov = [[1, 0.2], [0.2, 1]]  # Covariance matrix (correlated variables)

    # Generate unique samples from the distribution
    data = np.random.multivariate_normal(mean, cov, size=n_samples)

    # Extract X and Y
    X = data[:, 0].reshape(-1, 1)
    Y = data[:, 1].reshape(-1, 1)


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

    tsp.visualize(X, Y)

# Visualize the space partitions in a 2D scenario

