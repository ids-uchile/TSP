"""
Utilities for this repository
This package is meant to contain functions and classes that are useful for the implementations and experiments

Camilo Ramírez C. - FCFM - Universidad de Chile
"""
from TSP.utils.util_functions import add_tsp_path, reshape_array_as_2d, data_plot, print_emi_output, emi_evolution_plot, \
                                     plot_signals, plot_delta_matrix, animate_emi_evolution_matrix, \
                                     gen_drift_delta_evolution
from TSP.utils.tsp_functions import estimate_mi, compute_emi_evolution
from TSP.utils.tsp import TSP


from typing import Union, List

import numpy as np

Number = Union[int, float]
NumberArray = Union[List[Number], np.ndarray]
