"""
Basic example of the estimation of mutual information on random variables using the utis methods

Camilo Ramírez C. - FCFM - Universidad de Chile
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import data_plot, estimate_mi, print_emi_output, compute_emi_evolution, emi_evolution_plot

import matplotlib.pyplot as plt
import numpy as np


# In this example the data is sampled from a bivariate normal 
# Set the parameters for the distribution
mean_x = 0           
mean_y = 0           
std_x = 1            
std_y = 1            
rho = 0.5           

# Create the covariance matrix for the bivariate normal distribution
covariance_matrix = np.array([[std_x**2, rho * std_x * std_y],
                              [rho * std_x * std_y, std_y**2]])
mean = [mean_x, mean_y]

# Generate random samples from the bivariate normal distribution
size = 2*10**3 
x, y = np.random.multivariate_normal(mean, covariance_matrix, size).T

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