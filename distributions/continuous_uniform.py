"""
Continuous joint distribution that corresponds to a marginal and conditioned uniform, see class docstring for details

Camilo Ramírez C. - FCFM - Universidad de Chile
"""
from .abstract_distribution import AbstractDistribution
from scipy.special import lambertw
from typing import Optional

import numpy as np


class ContinuousUniform(AbstractDistribution):
    """
    A continuous-continuous distribution expressed by a continuous marginal X~Uniform([0, 1]) and a continuous
    conditional random variable Y|X~Uniform([kX, kX+1]), being k (the parameter) in [0, infinity)
    """
    def __init__(self, seed: int):
        super().__init__(seed=seed)

    def get_ami(self, parameter: float, n_symbols: Optional[int] = None) -> float:
        if n_symbols is not None:
            raise ValueError("This distribution does not require symbols, set 'n_symbols' to None")
        if parameter < 0:
            raise ValueError("The parameter (k) must be a non-negative number")
        return parameter*np.log2(np.e)/2 if parameter <= 1 else np.log2(parameter) + np.log2(np.e)/(2*parameter)

    def get_parameter(self, ami: float, n_symbols: Optional[int] = None) -> float:
        if n_symbols is not None:
            raise ValueError("This distribution does not require symbols, set 'n_symbols' to None")
        if ami < 0:
            raise ValueError("The AMI must be a non-negative number")
        return 2*ami/np.log2(np.e) if ami <= np.log2(np.e)/2 else -1/(2*np.real(lambertw(z=-2**(-ami-1))))

    def set_parameter(
        self, parameter: float, n_symbols: Optional[int] = None, continuous_noise: Optional[float] = None,
        discrete_shuffling: Optional[str] = None
    ) -> None:
        if not all(value is None for value in [n_symbols, continuous_noise, discrete_shuffling]):
            raise ValueError(
                "The parameters associated with discrete or mixed distributions (n_symbols, continuous_noise and "
                "discrete_shuffling) must be None for this continuous-continuous distribution"
            )
        if parameter < 0:
            raise ValueError("The parameter (k) must be a non-negative number")
        self.parameter = parameter

    def gen_data(self, sample_size: int) -> np.ndarray:
        x: np.ndarray = self.random_generator.uniform(low=0.0, high=1.0, size=sample_size)
        y: np.ndarray = self.parameter*x + self.random_generator.uniform(low=0.0, high=1.0, size=sample_size)
        return np.stack(arrays=(x, y), axis=1)
