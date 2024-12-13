"""
Utilities functions, this file is meant to contain functions that are useful in the context of the TSP

Camilo Ramírez C. - FCFM - Universidad de Chile
"""
from matplotlib import colors, cm, image
from typing import Union, Optional, List, Tuple

import matplotlib.collections as clt
import matplotlib.animation as anim
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

Number = Union[int, float]
NumberArray = Union[List[Number], np.ndarray]


def add_tsp_path(tsp_path: str) -> None:
    """
    This function adds the TSP shared object path to the system path; this function also checks if the given path
    actually contains the TSP file

    Parameters
    ----------
    tsp_path: str
        The string path that contains the TSP shared object

    Raises
    ------
    ImportError
        If the TSP path doesn't exist, or it doesn't contain the TSP shared object
    """
    if not os.path.exists(tsp_path):
        raise ImportError(f"The specified TSP path doesn't exist\ntsp_path: {tsp_path}")
    if not os.path.exists(f"{tsp_path}/TSP.py"):
        raise ImportError("The specified TSP path doesn't contain the TSP shared object")
    sys.path.append(tsp_path)


def reshape_array_as_2d(array: np.ndarray) -> np.ndarray:
    """
    This function verifies that a numpy array has two or one axis, in the later case it reshapes it into a 2-axis
    "column vector"

    Parameters
    ----------
    array: np.ndarray
        Array to reshape and verify

    Raises
    ------
    ValueError
        If the array of the argument has more than 2 axis or if it has no axis

    Returns
    -------
        The 2-axis version of the array
    """
    array_shape_len: int = len(array.shape)
    if array_shape_len > 2:
        raise ValueError("The array of the shape must not contain more than 2 axes")
    if array_shape_len == 0:
        raise ValueError("The array must have at least 1 axes")
    return array[:, np.newaxis] if array_shape_len == 1 else array


def data_plot(x: np.ndarray, y: np.ndarray, fig_size: Tuple[int, int] = (5, 4), mode: str = "scatter",
              color: str = 'b', alpha: float = 0.1, marker: Union[str, List[str]] = '.', grid_size: int = 100,
              c_map: str = "turbo", x_name: str = r"$x$", y_name: str = r"$y$",
              marker_size: Optional[int] = None) -> plt.Figure:
    """
    Function that creates a matplotlib Figure that contains a scatter-plot or a hexbin plot of the samples of x and y
    used as parameters, some aesthetics as the figure size, the color, the alpha of the points, and others can be set

    Parameters
    ----------
    x: np.ndarray
        Point values in the x-axis
    y: np.ndarray
        Point values in the y-axis
    fig_size: Tuple[int, int]
        Size of the figure
    mode: str
        Mode of the data plot, "scatter" (the default value) corresponds to a scatter-plot, and "hexbin" corresponds to
        a hexbin plot
    color: str
        Ignored in hexbin mode: String that represents a color in matplotlib
    alpha: float
        Ignored in hexbin mode: Alpha value of the points in the scatter-plot
    marker: Union[str, List[str]]
        Ignored in hexbin mode: Scatter-plot marker for the points, if it is a list, its size must be the same as the
        data
    grid_size: int
        Ignored in scatter mode: Hexbin-plot grid size
    c_map: str
        Ignored in scatter mode: Hexbin-plot color map string
    x_name: str
        Description of the x coordinate in Latex-style as a raw string
    y_name: str
        Description of the y coordinate in Latex-style as a raw string
    marker_size: Optional[str]
        Optional marker size, which changes the default marker size

    Raises
    ------
    TypeError
        If the marker is not a string or a list
    ValueError
        If the entries have not the same size, if they are not 1-dimensional; if the stated mode is an unknown one; or
        if the marker is a list whose length is not the same as the data length

    Returns
    -------
    The figure that contains the scatter-plot of the x and y samples
    """
    if x.shape != y.shape:
        raise ValueError("The arrays x and y must have the same size")
    if len(x.shape) > 1:
        raise ValueError("The arrays must be 1-dimensional")
    fig: plt.Figure
    ax: plt.Axes
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=fig_size)
    plot_name: str
    if mode == "scatter":
        plot_name = r"$Scatter$-$plot$"
        if type(marker) == str:
            ax.scatter(x=x, y=y, color=color, alpha=alpha, marker=marker, s=marker_size)
        elif type(marker) == list:
            if len(marker) != x.shape[0]:
                raise ValueError("The size of the marker list must match the data points size")
            for point_idx, marker in enumerate(marker):
                ax.scatter(x=x[point_idx], y=y[point_idx], color=color, alpha=alpha, marker=marker, s=marker_size)
        else:
            raise TypeError("The marker must be a string or a list of strings")
        ax.grid()
    elif mode == "hexbin":
        plot_name = r"$Hexbin$-$plot$"
        hb: clt.PolyCollection = ax.hexbin(x=x, y=y, gridsize=grid_size, cmap=c_map)
        ax.axis([np.min(a=x), np.max(a=x), np.min(a=y), np.max(a=y)])
        fig.colorbar(hb, ax=ax, label="Counting Values in a Hexagonal Cell")
    else:
        raise ValueError(f"Mode {mode} unknown for the data_plot function")
    ax.set_xlabel(xlabel=x_name)
    ax.set_ylabel(ylabel=y_name)
    ax.set_title(label=plot_name + " " + x_name + " vs " + y_name)
    fig.tight_layout()
    return fig


def emi_evolution_plot(evolution_array: np.ndarray, plot_post_emi: bool = True, plot_post_size: bool = False,
                       plot_prev_emi: bool = False, plot_prev_size: bool = False, x_name: str = r"$x$",
                       y_name: str = r"$y$") -> plt.Figure:
    """
    Generates a matplotlib figure that contains all the required evolution plots of the emi and the sizes of the
    respective tree

    Parameters
    ----------
    evolution_array: np.ndarray
        Contains the evolution of all the emi and tree sizes (previous and after regularization), this evolution array
        is built as stated in the function tsp_functions.compute_emi_evolution
    plot_post_emi: bool
        Indicates if the emi after regularization will be plotted or not
    plot_post_size: bool
        Indicates if the tree size after regularization will be plotted or not
    plot_prev_emi: bool
        Indicates if the emi previous to regularization will be plotted or not
    plot_prev_size: bool
        Indicates if the tree size previous to regularization will be plotted or not
    x_name: str
        Description of the x coordinate in Latex-style as a raw string
    y_name: str
        Description of the y coordinate in Latex-style as a raw string

    Raises
    ------
    ValueError
        If no plots are stated to be plotted

    Returns
    -------
    The figure that contains the plots stated in the parameters
    """
    plot_emi: bool = plot_post_emi or plot_prev_emi
    plot_size: bool = plot_post_size or plot_prev_size
    if not (plot_emi or plot_size):
        raise ValueError("At least one plot flag must be True, this means that at least one plot must be plotted")
    fig: plt.Figure
    ax: Union[list, plt.Axes]
    emi_ax: Optional[plt.Axes]
    size_ax: Optional[plt.Axes]
    if plot_emi and plot_size:
        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(8, 6), sharex="col")
        emi_ax = ax[0]
        size_ax = ax[1]
        size_ax.set_xlabel("Number of Samples")
    else:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 4))
        emi_ax = ax if plot_emi else None
        size_ax = ax if plot_size else None
    evol_idx_to_plot: np.ndarray = np.array([plot_post_emi, plot_post_size, plot_prev_emi, plot_prev_size], dtype=bool)
    evol_idx: int
    for evol_idx in range(4):
        if evol_idx_to_plot[evol_idx]:
            current_ax: plt.Axes = emi_ax if evol_idx == 0 or evol_idx == 2 else size_ax
            plot_color: str
            reg_label: str
            if evol_idx == 0 or evol_idx == 1:
                plot_color = 'b'
                reg_label = "Post Regularization"
            else:
                plot_color = 'r'
                reg_label = "Pre Regularization"
            current_ax.plot(evolution_array[0], evolution_array[evol_idx+1], color=plot_color, label=reg_label)
    if emi_ax is not None:
        emi_ax.set_ylabel(r"EMI $\hat{I}($" + x_name + "," + y_name + ")")
        emi_ax.set_title(r"EMI Evolution $\hat{I}($" + x_name + "," + y_name + ") with respect to the number of samples")
        emi_ax.grid()
        emi_ax.legend()
    if size_ax is not None:
        size_ax.set_ylabel("Tree Size")
        size_ax.set_title("TSP Tree Size Evolution with respect to the number of samples")
        size_ax.grid()
        size_ax.legend()
    fig.tight_layout()
    return fig


def plot_signals(signals: np.ndarray, signal_names: List[str], signal_colors: List[str],
                 fig_size: Tuple[int, int] = (8, 4)) -> plt.Figure:
    """
    Plots an array of signals

    Parameters
    ----------
    signals: np.ndarray
        The signals array, this considers a 1D matrix of 1 signal or a 2D matrix consisting on multiple signals, one
        signal by column to be plotted
    signal_names: List[str]
        A list of raw strings that determines the signal names, this list must be the same length as the amount of
        signals
    signal_colors: List[str]
        A list of raw signal colors, this list must be the same length as the amount of signals
    fig_size: Tuple[int, int]
        The desired figure size

    Raises
    ------
    ValueError
        If the signal names or the signal colors array are not the same length as te number of signals

    Returns
    -------
        A matplotlib figure that contains the plot
    """
    signal_array: np.ndarray = reshape_array_as_2d(signals)
    n_signals: int = signal_array.shape[1]
    fig: plt.Figure
    ax: plt.Axes
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=fig_size)
    signal_idx: int
    for signal_idx in range(n_signals):
        ax.plot(signal_array[:, signal_idx], color=signal_colors[signal_idx],
                label="Signal " + signal_names[signal_idx])
    ax.grid()
    ax.legend()
    ax.set_xlabel(r" $k$")
    ax.set_ylabel(r"Signal value $k$")
    signal_desc: str = "the signals" if n_signals > 1 else "the signal"
    ax.set_title(signal_desc + " time evolution")
    return fig


def plot_delta_matrix(
        matrix: np.ndarray, min_delta1: Number, max_delta1: Number, min_delta2: Number, max_delta2: Number,
        plot_title: str, color_bar_map_label: str, fig_size: Tuple[int, int] = (5, 5), color_map: str = "jet",
        interpolate: bool = True, v_min: Optional[Number] = None, v_max: Optional[Number] = None,
        log_scale: bool = False
) -> plt.Figure:
    """
    Plots a delta matrix, this matrix contains values associated with delta-mis-modelled systems

    Parameters
    ----------
    matrix: np.ndarray
        Matrix to plot
    min_delta1: Number
        Minimum delta1 value
    max_delta1: Number
        Maximum delta1 value
    min_delta2: Number
        Minimum delta2 value
    max_delta2: Number
        Maximum delta2 value
    plot_title: str
        Title of the plot
    color_bar_map_label: str
        Label of the color-bar
    fig_size: Tuple[int, int]
        Size of the figure
    color_map: str
        String of the color-bar of the plot
    interpolate: bool
        Boolean flag that defines if the plots will consider interpolation or not
    v_min: Optional[Number]
        Minimum value of the imshow plot, if None, the default behaviour will be applied
    v_max: Optional[Number]
        Maximum value of the imshow plot, if None, the default behaviour will be applied
    log_scale: bool
        Flag that defines if the imshow will be on a logarithmic scale (True) or not (False)

    Returns
    -------
    fig: plt.Figure
        A matpotlib figure that contains the plot
    """
    fig: plt.Figure
    ax: plt.Axes
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=fig_size)
    ax.set_ylabel(r"Values of $\delta_1$")
    ax.set_xlabel(r"Values of $\delta_2$")
    ax.tick_params(axis='x', labelrotation=90)
    ax.set_title(label=plot_title)
    img: image.AxesImage = ax.imshow(
        matrix, cmap=color_map, norm=colors.LogNorm(vmin=v_min, vmax=v_max) if log_scale else None,
        extent=[min_delta2, max_delta2, min_delta1, max_delta1], interpolation=None if interpolate else "none",
        vmin=None if log_scale else v_min, vmax=None if log_scale else None
    )
    fig.colorbar(img, ax=ax, label=color_bar_map_label, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def animate_emi_evolution_matrix(
        evolution_array: np.ndarray, sample_sizes: NumberArray, fig_title: str, imshow_extent: List[Number],
        max_value: Number, color_map_str: str = "jet", fig_size: Tuple[int, int] = (5, 5), fps: int = 5,
        interpolate: bool = True
) -> Tuple[anim.FuncAnimation, anim.PillowWriter]:
    """
    Function that animates (returns animator and animator writer objects) for an emi evolution array, this animation
    corresponds to a matplotlib object that enables to visualize the temporal evolution of the elements of the
    evolution matrix as it generates an "imshow" plot of every element of this matrix

    Parameters
    ----------
    evolution_array: np.ndarray
        The evolution array to be animated, this corresponds to a numpy array of shape (NxD1xD2), in where N is the
        total number of elements to animate in this matrix (i.e., number of frames); D1 and D2 corresponds to the
        dimension of these matrices, that are determined from the total amount of delta1 and delta2 values explored
        respectively
    sample_sizes: NumberArray
        An array of shape (N, ) that contains the number of samples that were used to generate the element in the
        evolution array of the same index respectively
    fig_title: str
        Title of the animation, the current number of samples will be added at the end of this title
    imshow_extent: List[Number]
        List of numbers that defines the limits of delta1 and delta2 explored in the emi matrix computation, this list
        is on the form: [min_delta2, max_delta2, min_delta1, max_delta1]
    max_value: Number
        Maximum value of the color map of the animation
    color_map_str: str
        String of the colormap to be used in the animation
    fig_size: Tuple[int, int]
        Size of the figure
    fps: int
        Frames per second of the animation
    interpolate: bool
        Boolean flag that defines if the the image will be interpolated or not

    Returns
    -------
    Tuple[anim.FuncAnimation, anim.PillowWriter]
        Tuple that contains the animation object and the writer object
    """
    # Figure initialization
    fig: plt.Figure
    ax: plt.Axes
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=fig_size)
    # Colormap building
    norm: colors.Normalize = colors.Normalize(vmin=0.0, vmax=max_value, clip=False)
    c_map: colors.LinearSegmentedColormap = cm.get_cmap(name=color_map_str, lut=8192)
    mappable: cm.ScalarMappable = cm.ScalarMappable(norm=norm, cmap=c_map)
    fig.colorbar(mappable=mappable, ax=ax, fraction=0.046, pad=0.04)
    # Axes image initialization
    img: image.AxesImage = ax.imshow(X=[[]], extent=imshow_extent, interpolation=None if interpolate else "none",
                                     vmin=0.0, vmax=max_value, cmap=c_map)
    ax.set_xlabel(xlabel=r"Values of $\delta_2$")
    ax.set_ylabel(ylabel=r"Values of $\delta_1$")
    ax.tick_params(axis='x', labelrotation=90)
    fig.tight_layout()

    def _fig_update(frame_idx: int) -> List[image.AxesImage]:
        """
        Auxiliary function for animation that updates the axes image with the new matrix

        Parameters
        ----------
        frame_idx: int
            Index of the frame to update into the axes image

        Returns
        -------
        List[image.AxesImage]
            A singleton list with the ax image that has been updated
        """
        img.set_array(evolution_array[frame_idx, 0])
        ax.set_title(f"{fig_title} - {sample_sizes[frame_idx]} samples")
        return [img]

    # Builds the animation and writer objects
    animation: anim.FuncAnimation = anim.FuncAnimation(fig=fig, func=_fig_update, frames=len(sample_sizes), blit=True)
    writer: anim.PillowWriter = anim.PillowWriter(fps=fps)
    return animation, writer


def print_emi_output(post_emi: float, post_size: int, prev_emi: float, prev_size: int) -> None:
    """
    Prints the output of the function estimate_mi in a nice way

    Parameters
    ----------
    post_emi: float
        The estimated mutual information after the regularization
    post_size: int
        The size of the tree after the regularization
    prev_emi: float
        The estimated mutual information previous to the regularization
    prev_size: int
        The size of the tree previous to the regularization
    """
    print("Results of the estimation of mutual information (EMI)\n")
    print("Results previous of the tree regularization")
    print(f"Size: {prev_size}")
    print(f"EMI value: {round(prev_emi, 3)}\n")
    print("Results after the tree regularization")
    print(f"Size: {post_size}")
    print(f"EMI value: {round(post_emi, 3)}\n")


def gen_drift_delta_evolution(
    final_delta: Tuple[float, ...], n_samples: int, failure_start: int, failure_stop: int
) -> np.ndarray:
    """
    Given a delta, generates a drift failure delta-evolution array, a drift failure starts in delta=(0, 0) in the
    failure_start point, and finished on delta=final_delta in the failure_stop point, performing a linear interpolation
    for the middle points

    Parameters
    ----------
    final_delta: Tuple[float, ...]
        The delta value to get after the drift
    n_samples: int
        Total number of samples for the run
    failure_start: int
        Point where the delta failure will start
    failure_stop: int
        Point where the delta failure will stop

    Returns
    -------
    An array of th delta-evolution with the drift failure
    """
    delta_array: np.ndarray = np.zeros(shape=(n_samples, 2))
    delta_0_step: float = final_delta[0]/(failure_stop-failure_start+1)
    delta_1_step: float = final_delta[1]/(failure_stop-failure_start+1)
    if delta_0_step == 0.0:
        delta_0_range = np.zeros(shape=failure_stop - failure_start + 2)
    else:
        delta_0_range: np.ndarray = np.arange(start=0.0, stop=final_delta[0] + delta_0_step, step=delta_0_step)
    if delta_1_step == 0.0:
        delta_1_range = np.zeros(shape=failure_stop - failure_start + 2)
    else:
        delta_1_range: np.ndarray = np.arange(start=0.0, stop=final_delta[1] + delta_1_step, step=delta_1_step)
    delta_array[failure_start-1:failure_stop+1, 0] = delta_0_range
    delta_array[failure_start-1:failure_stop+1, 1] = delta_1_range
    delta_array[failure_stop+1:, 0] = final_delta[0]
    delta_array[failure_stop+1:, 1] = final_delta[1]
    return delta_array
