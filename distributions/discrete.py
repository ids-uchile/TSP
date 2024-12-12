"""
Discrete joint distribution, that corresponds to both discrete marginals, see class docstring for details

Camilo Ramírez C. - FCFM - Universidad de Chile
"""
from .abstract_distribution import AbstractDistribution
from scipy.optimize import newton as newton_raphson_method
from typing import Optional

import numpy as np


class Discrete(AbstractDistribution):
    """
    A discrete-discrete distribution expressed by a joint pmf defined for all (i, j) in {1, 2, ..., m}^2 as
    (1+k(m-1))/m^2 if i=j, and (1-k)/m^2 if i is not equal to j. Being k (the parameter) in [0, 1] and m the amount of
    symbols
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
        if 0 < parameter < 1:
            return (
                (1+parameter*(n_symbols-1))*np.log2(1+parameter*(n_symbols-1))/n_symbols
                + (n_symbols-1)*(1-parameter)*np.log2(1-parameter)/n_symbols
            )
        else:
            return parameter*np.log2(n_symbols)

    def get_parameter(self, ami: float, n_symbols: Optional[int] = None) -> float:
        self._check_get_parameter_arguments_on_discrete(ami=ami, n_symbols=n_symbols)
        if ami == 0.0:
            return 0.0
        elif ami == np.log2(n_symbols):
            return 1.0
        else:
            return newton_raphson_method(
                func=lambda x: (
                    (1+x*(n_symbols-1))*np.log2(1+x*(n_symbols-1))/n_symbols
                    + (n_symbols-1)*(1-x)*np.log2(1-x)/n_symbols - ami
                ), x0=0.5,
                fprime=lambda x: (n_symbols-1)*(np.log2(1+x*(n_symbols-1))-np.log2(1-x))/n_symbols,
                fprime2=lambda x: np.log2(np.e)*(n_symbols-1)/((1+x*(n_symbols-1))*(1-x))
            )

    def set_parameter(
        self, parameter: float, n_symbols: Optional[int] = None, continuous_noise: Optional[float] = None,
        discrete_shuffling: Optional[str] = None
    ) -> None:
        self._set_parameter_arguments_on_discrete(
            parameter=parameter, n_symbols=n_symbols, continuous_noise=continuous_noise,
            discrete_shuffling=discrete_shuffling
        )

    def gen_data(self, sample_size: int) -> np.ndarray:
        p_array: np.ndarray = (1+self.parameter*(self.n_symbols-1))/self.n_symbols**2 * np.eye(N=self.n_symbols)
        p_array[p_array == 0] = (1-self.parameter)/self.n_symbols**2
        data: np.ndarray = self.random_generator.choice(a=self.n_symbols**2, size=sample_size, p=p_array.flatten())
        x: np.ndarray = data // self.n_symbols
        y: np.ndarray = data % self.n_symbols
        if self.discrete_shuffling == "fixed":
            x = self.fixed_shuffling_array[x]
        elif self.discrete_shuffling == "stochastic":
            shuffling_array: np.ndarray = np.arange(self.n_symbols)
            self.random_generator.shuffle(shuffling_array)
            x = shuffling_array[x]
        x = x.astype(float) + self.random_generator.uniform(low=-self.continuous_noise, high=self.continuous_noise, size=sample_size) + 1
        y = y.astype(float) + self.random_generator.uniform(low=-self.continuous_noise, high=self.continuous_noise, size=sample_size) + 1
        return np.stack(arrays=(x, y), axis=1)
