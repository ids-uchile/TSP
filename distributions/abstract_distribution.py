"""
Abstract Distribution class, that defines a joint-distribution for a random vector (X,Y), see each class for further
details on the properties of the distribution

Camilo Ramírez C. - FCFM - Universidad de Chile
"""
from numpy.random import Generator
from typing import Optional

import numpy as np
import abc


class AbstractDistribution(abc.ABC):
    """
    Abstract distribution class, all the distribution used for empirical EMI analysis should be inherited from this
    class

    Attributes
    ----------
    random_generator
        Random number generator for the distribution
    parameter: Optional[float]
        Distribution parameter, see class docstring for further description. None if not required or set
    n_symbols: Optional[int]
        Number of symbols, if required by the distribution description, None if else or not set
    continuous_noise: Optional[float]
        The bound for the uniform interval used in the continuization of the discrete components in the random
        variable, this is an additive noise independent of everything else distributed as Uniform([0, w]), where w
        correspond to this parameter, this value must be in (0, 1]. This value must be None if not required. Can be
        None if nor set yet
    discrete_shuffling: Optional[bool]
        If True, performs a shuffling on the discrete components of the random variable, i.e., reasignes the
        realization of these variables, not modifying the continuous components, if False it is not done. This
        value must be None if not required; it can be None although required if not set yet
    fixed_shuffling_array: Optional[np.ndarray]
        An array that defines a fixed shuffling, i.e., fixed_shuffling_array[idx] determines the numerical value of the
        class idx
    """
    def __init__(self, seed: int):
        """
        Class constructor

        Parameters
        ----------
        seed: int
            Value for the seed of the random generator
        """
        self.random_generator: Generator = np.random.default_rng(seed=seed)
        self.parameter: Optional[float] = None
        self.n_symbols: Optional[int] = None
        self.continuous_noise: Optional[float] = None
        self.discrete_shuffling: Optional[str] = None
        self.fixed_shuffling_array: Optional[np.ndarray] = None

    @abc.abstractmethod
    def get_ami(self, parameter: float, n_symbols: Optional[int] = None) -> float:
        """
        Gets the analytical mutual information (AMI) of the distribution, i.e., I(X;Y), given the parameter value, and
        if required by the distribution, the number of symbols that conforms it

        Parameters
        ----------
        parameter: float
            Distribution parameter, see class docstring for further description
        n_symbols: Optional[int]
            Number of symbols if required by the distribution description, None if else

        Raises
        ------
        ValueError
            If the number of symbols is specified (not None) and the distribution does not require it, and vice versa;
            if this number is non-positive, and if the parameter value outbounds the range where it is defined

        Returns
        -------
        float
            The analytical mutual information for the parameters, and the number of symbols (if required) specified
        """
        pass

    @abc.abstractmethod
    def get_parameter(self, ami: float, n_symbols: Optional[int] = None) -> float:
        """
        Gets the parameter value for which the analytical mutual information (AMI) of the distribution, i.e., I(X;Y),
        is the required ami value; to compute this value, the number of symbols is required if the distribution
        has any discrete marginal random value

        Parameters
        ----------
        ami: float
            The desired analytical mutual information (AMI) value for the distribution
        n_symbols: Optional[int]
            Number of symbols if required by the distribution description, None if else

        Raises
        ------
        ValueError
            If the number of symbols is specified (not None) and the distribution does nor required it, and vice versa;
            if the AMI is a negative value; if the number of symbols is not enough for the distribution to reach the
            desired AMI, and if the number of symbols is not a natural number

        Returns
        -------
        float
            The value of the parameter of the distribution which gives the desired analytical mutual information
        """
        pass

    @abc.abstractmethod
    def set_parameter(
            self, parameter: float, n_symbols: Optional[int] = None, continuous_noise: Optional[float] = None,
            discrete_shuffling: Optional[str] = None
    ) -> None:
        """
        Sets the parameter (and the number of symbols if required) of the distribution for further data generation

        Parameters
        ----------
        parameter: float
            Distribution parameter, see class docstring for further description
        n_symbols: Optional[int]
            Number of symbols, if required by the distribution description, None if else
        continuous_noise: Optional[float]
            The bound for the uniform interval used in the continuization of the discrete components in the random
            variable, this is an additive noise independent of everything else distributed as Uniform([0, w]), where w
            correspond to this parameter, this value must be in (0, 1]. This value must be None if not required.
        discrete_shuffling: Optional[str]
            Sets the shuffling mode for the random discrete components of the random variable, i.e., reasignes the
            realization of these variables, not modifying the continuous components. If "none" it does not shuffle, if
            "fixed" a shuffling arrangement will be fixed every time a new "n_symbols" value is given, if "stochastic"
            every data generation procedure will involve a different shuffling. Finally, this value must be None if not
            required due to the continuity of the random marginals involved

        Raises
        ------
        ValueError
            If the number of symbols, the continuous noise, or the discrete shuffling is specified (not None) and the
            distribution does not require them, and vice versa; if the parameter value outbounds the range where it is
            defined, if the number of symbols is not a natural number, if the continuous noise parameter is not in
            (0, 1], and if the discrete_shuffling mode is invalid
        """
        pass

    @abc.abstractmethod
    def gen_data(self, sample_size: int) -> np.ndarray:
        """
        Samples an array of size (N, 2) from the distribution, where N corresponds to the sample size

        Parameters
        ----------
        sample_size: int
            The amount of samples to be retrieved

        Raises
        ------
        RuntimeError
            If the parameters were not set yet

        Returns
        -------
        np.ndarray
            The samples array
        """
        pass

    def _set_parameter_arguments_on_discrete(
        self, parameter: float, n_symbols: Optional[int] = None, continuous_noise: Optional[float] = None,
        discrete_shuffling: Optional[str] = None
    ):
        """
        Auxiliary method that performs the parameter setting when used in a distribution with discrete components; see
        the method set_parameter for further details on the parameters and the exceptions raised by this method

        See Also
        --------
        set_parameter
            The method for which this method is designed
        """
        if any(value is None for value in [n_symbols, continuous_noise, discrete_shuffling]):
            raise ValueError(
                "The parameters associated with mixed distributions (n_symbols, continuous_noise and"
                "discrete_shuffling) must be specified (not None)"
            )
        if n_symbols <= 0:
            raise ValueError("The number of symbols must be a positive value")
        if not 0 < continuous_noise <= 1:
            raise ValueError("The continuous noise parameter must be in (0, 1]")
        if discrete_shuffling not in ["none", "fixed", "stochastic"]:
            raise ValueError("Invalid discrete_shuffling value. Valid values are 'none', 'fixed' and 'stochastic'")
        if not 0 <= parameter <= 1:
            raise ValueError("The parameter (k) value must be in [0, 1]")
        self.parameter = parameter
        self.continuous_noise = continuous_noise
        self.discrete_shuffling = discrete_shuffling
        if self.discrete_shuffling == "fixed" and self.n_symbols != n_symbols:
            self.fixed_shuffling_array = np.arange(n_symbols)
            self.random_generator.shuffle(x=self.fixed_shuffling_array)
        self.n_symbols = n_symbols

    @staticmethod
    def _check_get_parameter_arguments_on_discrete(ami: float, n_symbols: Optional[int] = None) -> None:
        """
        Auxiliary method that checks the consistency of the arguments passed to the method get_parameter when used in a
        distribution with discrete components; see that method for further details on the parameters and the exceptions
        raised by this method

        See Also
        --------
        get_parameter
            The method whose arguments are checked with this method
        """
        if n_symbols is None:
            raise ValueError(
                "This distribution requires to set a number of symbols, set 'n_symbols' different to None"
            )
        if n_symbols <= 0:
            raise ValueError("The number of symbols must be a positive value")
        if ami < 0:
            raise ValueError("The AMI must be a non-negative number")
        if n_symbols < 2**ami:
            raise ValueError(
                f"The number of symbols is not enough to reach the desired AMI, the minimum required number of "
                f"symbols is {np.ceil(2 ** ami)}"
            )
