"""
Tree Structured Partition class, represents a tree (set of nodes) 
and the values associated with the MI estimation problem,
such as empirical MI, regularized MI and tree size.

Maximiliano Rojas - Information and Decision System Group - Universidad de Chile
"""

import numpy as np
import matplotlib.pyplot as plt
from TSPNode import TSPNode

class TSP:
    def __init__(self, l_bn, w_bn, _lambda):
        """
        Initialize the TSP instance with the necessary hyperparameters 
        for tree construction and regularization.

        Parameters
        ----------
        l_bn : float
            Exponent controlling the critical mass calculation, must lie in (0, 1/3).
        w_bn : float
            Coefficient for the critical mass calculation.
        _lambda : float
            Regularization parameter.
        
        Raises
        ------
        ValueError
            If l_bn is not in (0, 1/3).
        """
        if l_bn >= 1/3.0 or l_bn <= 0:
            raise ValueError("Parameter `l_bn` must belong to the interval (0, 1/3).")
        
        self.l_bn = l_bn
        self.w_bn = w_bn
        self.kn = 0
        self._lambda = _lambda
        self.root = None
        self.dim = 0
        self.n_samples = 0
        self.tsp_size = 0
        self.tsp_reg_size = 0
        self.tsp_emi = 0.0
        self.tsp_reg_emi = 0.0
        self.tsp_partitions = []

    def grow(self, x, y):
        """
        Builds (grows) the TSP tree from data X and Y. The data arrays are combined,
        and the initial node indexes are generated to recursively partition the space.

        Parameters
        ----------
        x : array-like
            Data samples for the first variable (X). Must be 2D: shape (n_samples, x_dim).
        y : array-like
            Data samples for the second variable (Y). Must be 2D: shape (n_samples, y_dim).
        
        Notes
        -----
        This method updates:
            - self.root with a fully grown TSPNode tree.
            - self.tsp_emi, self.tsp_size, and self.tsp_partitions based on the grown tree.
        """
        # Convert inputs to NumPy arrays if they aren't already
        x = np.asarray(x)
        y = np.asarray(y)

        # Concatenate X and Y into a single data array
        data = np.concatenate((x, y), axis=1)

        # Number of samples and dimensions
        self.n_samples = data.shape[0]
        self.dim = data.shape[1]
        
        # Create a root node
        self.root = TSPNode()

        # Determine critical mass kn = ceil(w_bn * n^(1 - l_bn))
        self.kn = np.ceil(self.w_bn * (self.n_samples ** (1 - self.l_bn)))

        # Define the global bounds for all dimensions
        lowerBounds = np.min(data, axis=0) - 0.001
        upperBounds = np.max(data, axis=0) + 0.001

        # Create a Boolean list indicating which dims belong to X (True) and which to Y (False)
        Xdim_indicator = [True] * x.shape[1] + [False] * y.shape[1]

        # Indices of every sample in the whole space
        nodeIdx = np.arange(self.n_samples, dtype=int)

        # Grow the tree from the root
        self.root = self.root.grow(
            None,                   
            nodeIdx,               
            data,
            lowerBounds,
            upperBounds,
            Xdim_indicator,
            self.dim,
            self.kn,
            projDim=0
        )

        # After full growth, collect results
        self.tsp_emi = self.root.getEMI()
        self.tsp_size = self.root.getSize()
        self.tsp_partitions = self.root.getPartitions()

    def regularize(self):
        """
        Regularizes (prunes) the grown TSP tree to balance the trade-off 
        between the empirical mutual information and model complexity.

        Raises
        ------
        ValueError
            If no tree (self.root) exists (i.e., grow() has not been called).

        Notes
        -----
        - Uses self.minimum_cost_trees to obtain the mutual information (EMI) 
          for every minimal-cost tree.
        - Employs a cost function with a regularization term determined by self._lambda.
        - Updates self.tsp_reg_emi and self.tsp_reg_size with the best result.
        """
        # A grown tree is required
        if self.root is None:
            raise ValueError("Observations not provided.")

        # The size (number of leaves) of the full-grown tree
        full_tree_size = self.tsp_size

        # Regularizer terms
        bn = self.w_bn * np.power(self.n_samples, -self.l_bn)
        inv_deltan = np.exp(self.n_samples ** (1/3.0))

        # Build array of EMI values for each minimal-cost tree size
        treesEMI = np.zeros(full_tree_size)
        self.minimum_cost_trees(treesEMI, full_tree_size)

        # We will look for the optimal cost and size
        optimal_cost = -treesEMI[0]
        optimal_size = 1

        # Precompute repeated logs and constants
        constant_term = 8.0 / self.n_samples
        dim_log = (self.dim + 1) * np.log(2) + self.dim * np.log(self.n_samples)
        log_inv = np.log(8 * inv_deltan)

        # Evaluate cost for every minimum-cost tree size from 2..full_tree_size
        for k in range(2, full_tree_size + 1):
            # Epsilon depends on k
            cost_arg = log_inv + k * dim_log
            epsilon = (12.0 / bn) * np.sqrt(constant_term * cost_arg)

            cost = -treesEMI[k - 1] + self._lambda * epsilon

            # Update optimal cost/size if improved
            if cost < optimal_cost:
                optimal_cost = cost
                optimal_size = k

        # Once pruned, store the results
        self.tsp_reg_emi = -optimal_cost
        self.tsp_reg_size = optimal_size

    def minimum_cost_trees(self, treesEMI, full_tree_size):
        """
        This method computes EMI values for all possible "minimum cost" trees 
        by iteratively splitting the leaf with the maximum absoluteCMIgain 
        and storing partial EMI sums in `treesEMI`.

        Parameters
        ----------
        treesEMI : np.ndarray
            A 1D array (length full_tree_size) where each index k will store
            the EMI value for the minimum-cost tree of size k+1.
        full_tree_size : int
            Total number of leaves (size of the fully grown tree).

        Notes
        -----
        - The first element of treesEMI (for tree size=1) remains 0.
        - The method repeatedly expands the leaf with the highest absoluteCMIgain
          and updates the EMI accordingly.
        - Calls subadditive_insert(...) to maintain leaves in ascending 
          order of absoluteCMIgain.
        """
        # Array of leaves. Each index in `leaves` will eventually point to a leaf in the tree.
        leaves = [None] * full_tree_size
        leaves[0] = self.root

        # By definition, the tree of size=1 has EMI = 0
        # So treesEMI[0] is already 0 by default

        for k in range(full_tree_size - 1):
            # Leaf to expand = the leaf with highest absoluteCMIgain among existing leaves
            leaf_maxCMI = leaves[k]

            # EMI for tree of size k+1 = EMI of tree of size k + absoluteCMIgain of chosen leaf
            treesEMI[k + 1] = treesEMI[k] + leaf_maxCMI.absoluteCMIgain

            # Insert left & right children into `leaves`, maintaining ascending absoluteCMIgain
            self.subadditive_insert(leaf_maxCMI.left, leaves, k)
            self.subadditive_insert(leaf_maxCMI.right, leaves, k + 1)

    def subadditive_insert(self, new_leaf, leaves, j):
        """
        Replaces bubble-sort with a single-pass insertion from right to left.
        This yields the same final ordering as the old bubble-sort approach 
        (ascending absoluteCMIgain) but more efficiently.

        Parameters
        ----------
        new_leaf : TSPNode
            The newly expanded child leaf to insert into the list.
        leaves : list of TSPNode
            The collection of leaves being maintained in ascending order of absoluteCMIgain.
        j : int
            The initial index at which `new_leaf` is placed before insertion sort correction.
        """
        leaves[j] = new_leaf
        new_gain = new_leaf.absoluteCMIgain

        # We'll move left from j until we find the correct insertion point
        i = j
        while i > 0 and leaves[i - 1].absoluteCMIgain > new_gain:
            # Swap adjacent leaves
            leaves[i], leaves[i - 1] = leaves[i - 1], leaves[i]
            i -= 1

    def visualize(self, x, y):
        """
        Visualizes the 2D partitioning (TSP tree boundaries) over the given data.

        Parameters
        ----------
        x : array-like
            Samples for X (2D array).
        y : array-like
            Samples for Y (2D array).

        Raises
        ------
        ValueError
            If the tree has not been grown (self.root is None) 
            or the data dimension is not 2.
        """
        if self.root is None:
            raise ValueError("Observations not provided.")
        if self.dim != 2:
            raise ValueError("The tree can only be visualized for a two dimensional problem.")

        x = np.asarray(x)
        y = np.asarray(y)

        # Partitions
        partitions = self.partitions()

        # Separate partitions by type
        horizontal_partition = [p[1:] for p in partitions if p[0] == 0]
        vertical_partition   = [p[1:] for p in partitions if p[0] == 1]

        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(16, 5))

        # Left subplot: samples only
        axes[0].scatter(x, y, marker='o', edgecolor='k', s=25, alpha=0.5)
        axes[0].set_xlabel('x')
        axes[0].set_ylabel('y')
        axes[0].set_title('Samples')
        axes[0].grid(alpha=0.3)

        # Middle subplot: TSP lines over samples
        axes[1].scatter(x, y, marker='o', edgecolor='k', s=25, alpha=0.5)

        # Plot horizontal lines
        for bound in vertical_partition:
            axes[1].hlines(
                y=bound[0][1],
                xmin=bound[0][0],
                xmax=bound[1][0],
                color='black'
            )

        # Plot vertical lines
        for bound in horizontal_partition:
            axes[1].vlines(
                x=bound[0][0],
                ymin=bound[0][1],
                ymax=bound[1][1],
                color='black'
            )

        axes[1].set_xlabel('x')
        axes[1].set_ylabel('y')
        axes[1].set_title(
            f"TSP: samples={self.n_samples}, cell_samples={int(self.kn)}, grow_size={self.size()}"
        )
        axes[1].grid(alpha=0.3)

        # Right subplot: TSP lines, samples with alpha=0.1
        axes[2].scatter(x, y, marker='o', edgecolor='k', s=25, alpha=0.1)

        for bound in vertical_partition:
            axes[2].hlines(
                y=bound[0][1],
                xmin=bound[0][0],
                xmax=bound[1][0],
                color='black'
            )
        for bound in horizontal_partition:
            axes[2].vlines(
                x=bound[0][0],
                ymin=bound[0][1],
                ymax=bound[1][1],
                color='black'
            )

        axes[2].set_xlabel('x')
        axes[2].set_ylabel('y')
        axes[2].set_title("TSP Partitions")
        axes[2].grid(alpha=0.3)

        plt.tight_layout()
        plt.show()

    def emi(self):
        """
        Returns the empirical mutual information (EMI) for the fully grown TSP tree.

        Returns
        -------
        float
            The EMI value stored in `self.tsp_emi`.

        Raises
        ------
        ValueError
            If the tree has not been grown (self.root is None).
        """
        if self.root is None:
            raise ValueError("Observations not provided.")
        return self.tsp_emi

    def reg_emi(self):
        """
        Returns the regularized empirical mutual information for the pruned TSP tree.

        Returns
        -------
        float
            The regularized EMI value stored in `self.tsp_reg_emi`.

        Raises
        ------
        ValueError
            If the tree has not been grown (self.root is None).
        """
        if self.root is None:
            raise ValueError("Observations not provided.")
        return self.tsp_reg_emi

    def size(self):
        """
        Returns the size (number of leaves) of the fully grown TSP tree.

        Returns
        -------
        int
            The tree size stored in `self.tsp_size`.

        Raises
        ------
        ValueError
            If the tree has not been grown (self.root is None).
        """
        if self.root is None:
            raise ValueError("Observations not provided.")
        return self.tsp_size

    def reg_size(self):
        """
        Returns the size (number of leaves) of the pruned TSP tree 
        after the regularization process.

        Returns
        -------
        int
            The pruned tree size stored in `self.tsp_reg_size`.

        Raises
        ------
        ValueError
            If the tree has not been grown (self.root is None).
        """
        if self.root is None:
            raise ValueError("Observations not provided.")
        return self.tsp_reg_size

    def partitions(self):
        """
        Returns a list of partitions in the form 
        [(dim, lower_bounds, upper_bounds), ...] for each split.

        Returns
        -------
        list
            The list of partitions stored in `self.tsp_partitions`.

        Raises
        ------
        ValueError
            If the tree has not been grown (self.root is None).
        """
        if self.root is None:
            raise ValueError("Observations not provided.")
        return self.tsp_partitions
