"""
Mixed joint distribution, that corresponds to a discrete marginal, and a conditioned uniform, see class docstring for
details

Camilo Ramírez C. - FCFM - Universidad de Chile
"""
from .abstract_distribution import AbstractDistribution
from typing import Optional

import numpy as np


class Mixed(AbstractDistribution):
    """
    A continuous-discrete distribution expressed by the discrete marginal Y~Uniform({1, 2, ..., m}), where m is the
    amount of symbols, and a continuous conditional random variable X|Y~Uniform([1-k] union [1+k(Y-2), 1+k(Y-1)]),
    being k (the parameter) in [0,1]
    """
    def __init__(self, seed: int):
        super().__init__(seed=seed)

    def get_ami(self, parameter: float, n_symbols: Optional[int] = None) -> float:
        if n_symbols is None:
            raise ValueError(
                "This distribution requires to set a number of symbols, set 'n_symbols' different to None"
            )
        if n_symbols <= 0:
            raise ValueError("The number of symbols must be a positive value")
        if not 0 <= parameter <= 1:
            raise ValueError("The parameter (k) must be in [0,1]")
        return parameter*np.log2(n_symbols)

    def get_parameter(self, ami: float, n_symbols: Optional[int] = None) -> float:
        self._check_get_parameter_arguments_on_discrete(ami=ami, n_symbols=n_symbols)
        return ami/np.log2(n_symbols)

    def set_parameter(
        self, parameter: float, n_symbols: Optional[int] = None, continuous_noise: Optional[float] = None,
        discrete_shuffling: Optional[str] = None
    ) -> None:
        self._set_parameter_arguments_on_discrete(
            parameter=parameter, n_symbols=n_symbols, continuous_noise=continuous_noise,
            discrete_shuffling=discrete_shuffling
        )

    def gen_data(self, sample_size: int) -> np.ndarray:
        y: np.ndarray = self.random_generator.integers(low=1, high=self.n_symbols+1, size=sample_size)
        x: np.ndarray = self.random_generator.uniform(low=0.0, high=1.0, size=sample_size)
        x[x > 1-self.parameter] += self.parameter*(y[x > 1-self.parameter]-1)
        if self.discrete_shuffling == "fixed":
            y = self.fixed_shuffling_array[y-1] + 1
        elif self.discrete_shuffling == "stochastic":
            shuffling_array: np.ndarray = np.arange(self.n_symbols)
            self.random_generator.shuffle(shuffling_array)
            y = shuffling_array[y-1] + 1
        y = y.astype(float) + self.random_generator.uniform(low=0.0, high=self.continuous_noise, size=sample_size)
        return np.stack(arrays=(x, y), axis=1)

