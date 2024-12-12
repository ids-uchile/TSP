"""
Basic example of the estimation of mutual information on random variables using the utis methods

Camilo Ramírez C. - FCFM - Universidad de Chile
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import data_plot, estimate_mi, print_emi_output, compute_emi_evolution, emi_evolution_plot
from distributions import ContinuousGaussian, ContinuousUniform, Mixed, Discrete

import matplotlib.pyplot as plt
import numpy as np

# As an example, use a bi-variate gaussian
sample_size = 2000
parameter = 0.5
n_symbols = 4
continuous_noise = 0.1
discrete_shuffling = 'fixed'

# Change the method according to the desired distribution
distribution = Discrete(seed=42)
distribution.set_parameter(
      parameter=parameter,
      n_symbols=n_symbols,
      continuous_noise=continuous_noise,
      discrete_shuffling=discrete_shuffling
  )

# Generate sampĺes
data = distribution.gen_data(sample_size)

# Split samples in the random variables X and Y
x = data[:, 0]
y = data[:, 1]

# Estimation of the mutual information
post_emi, post_size, prev_emi, prev_size = estimate_mi(x=x, y=y)

# Printing of the emi results in console
print_emi_output(post_emi, post_size, prev_emi, prev_size)

# Build and show the scatter-plot of X and Y
splot: plt.Figure = data_plot(x=x, y=y, mode="scatter", alpha=0.3)
splot.show()

# Compute the emi and tree size evolution as the sample size increases
emi_size_evolution = compute_emi_evolution(x=x, y=y, stride=1, show_tqdm=True)

# Plots the evolution of the EMI
evol_plot: plt.Figure = emi_evolution_plot(evolution_array=emi_size_evolution, plot_post_emi=True, plot_post_size=True,
                                           plot_prev_emi=True, plot_prev_size=True)
evol_plot.show()