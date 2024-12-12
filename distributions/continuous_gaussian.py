"""
Continuous joint distribution that corresponds to a bi-variate gaussian, see class docstring for details

Camilo Ramírez C. - FCFM - Universidad de Chile
"""
from .abstract_distribution import AbstractDistribution
from typing import Optional

import numpy as np


class ContinuousGaussian(AbstractDistribution):
    """
    A continuous-continuous distribution expressed by a joint bi-variate gaussian where (X,Y)~N((0, 0), Sigma), where
    Sigma corresponds to the covariance matrix, whose values are [[1, rho], [rho, 1]], being rho (the parameter) the
    correlation (and covariance as both marginal variances are 1) between X and Y.
    """
    def __init__(self, seed: int):
        super().__init__(seed=seed)

    def get_ami(self, parameter: float, n_symbols: Optional[int] = None) -> float:
        if n_symbols is not None:
            raise ValueError("This distribution does not require symbols, set 'n_symbols' to None")
        if not 0 <= parameter < 1:
            raise ValueError("The parameter (rho) must be in [0,1)")
        return -np.log2(1-parameter**2)/2

    def get_parameter(self, ami: float, n_symbols: Optional[int] = None) -> float:
        if n_symbols is not None:
            raise ValueError("This distribution does not require symbols, set 'n_symbols' to None")
        if ami < 0:
            raise ValueError("The AMI must be a non-negative number")
        return np.sqrt(1-4**(-ami))

    def set_parameter(
        self, parameter: float, n_symbols: Optional[int] = None, continuous_noise: Optional[float] = None,
        discrete_shuffling: Optional[str] = None
    ) -> None:
        if not all(value is None for value in [n_symbols, continuous_noise, discrete_shuffling]):
            raise ValueError(
                "The parameters associated with discrete or mixed distributions (n_symbols, continuous_noise and "
                "discrete_shuffling) must be None for this continuous-continuous distribution"
            )
        if not 0 <= parameter < 1:
            raise ValueError("The parameter (rho) must be in [0,1)")
        self.parameter = parameter

    def gen_data(self, sample_size: int) -> np.ndarray:
        return self.random_generator.multivariate_normal(
            mean=(0, 0), cov=((1, self.parameter), (self.parameter, 1)), size=sample_size
        )
