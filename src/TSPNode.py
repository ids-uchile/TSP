"""
Tree Structured Partition Node class, represents a node (space partition) in the tree. 

Maximiliano Rojas - Information and Decision System Group - Universidad de Chile
"""

import numpy as np
import math
import numba

@numba.njit
def filter_in_bounds(data, idx_array, lower_bounds, upper_bounds, Xdim_indicator, is_x_partition):
    """
    Filter the samples in 'idx_array' by checking if they lie within the 
    [lower_bounds, upper_bounds] for the relevant dimensions (X or Y).
    
    Parameters
    ----------
    data : np.ndarray
        The entire dataset of shape (N, dim).
    idx_array : np.ndarray
        1D array of sample indices to filter.
    lower_bounds : np.ndarray
        1D array of length 'dim' giving the lower bound for each dimension.
    upper_bounds : np.ndarray
        1D array of length 'dim' giving the upper bound for each dimension.
    Xdim_indicator : np.ndarray (bool)
        1D boolean array of length 'dim'. True for X dimensions, False for Y.
    is_x_partition : bool
        True if we should check only dimensions indicated by Xdim_indicator == True.
        False if we should check only those indicated by Xdim_indicator == False.

    Returns
    -------
    np.ndarray
        The subset of indices that passed all bound checks on relevant dimensions.
    """
    out_idx = np.empty(len(idx_array), dtype=idx_array.dtype)
    count = 0

    for i in range(len(idx_array)):
        sample_idx = idx_array[i]
        # We'll check each dimension that belongs to X (if is_x_partition=True)
        # or belongs to Y (if is_x_partition=False).
        pass_all = True
        for d in range(len(Xdim_indicator)):
            dim_is_x = Xdim_indicator[d]
            # Decide if dimension 'd' is relevant
            if (is_x_partition and dim_is_x) or ((not is_x_partition) and (not dim_is_x)):
                val = data[sample_idx, d]
                # If out of bounds, short-circuit for this sample
                if val < lower_bounds[d] or val > upper_bounds[d]:
                    pass_all = False
                    break
        
        if pass_all:
            out_idx[count] = sample_idx
            count += 1

    return out_idx[:count]


class TSPNode:
    """
    TSPNode represents a node (space partition) in the tree structure used for 
    mutual information estimation. Each node maintains properties such as:
      - Bounds of the partition.
      - Marginal and joint distribution measures.
      - Indices of samples belonging to X or Y partitions.
      - References to left and right child nodes.
    """

    def __init__(self, parent=None):
        """
        Initialize a TSPNode with references to its parent and default 
        properties for distribution measures and partition indices.

        Parameters
        ----------
        parent : TSPNode or None
            The parent node in the tree. If None, this node may become the root.
        """
        self.left = None
        self.right = None
        self.parent = parent
        self.condJointDist = 0.0
        self.condMargProd = 0.0
        self.relativeCMIgain = float("-inf")
        self.absoluteCMIgain = float("-inf")
        self.n_samples = 0
        self.n_marginal_samples_X = 0
        self.n_marginal_samples_Y = 0
        self.idx_marginal_samples_X = []
        self.idx_marginal_samples_Y = []
        self.lowerBounds = np.array([])
        self.upperBounds = np.array([])
        self.partitions = []

    def grow(self, parent, nodeIdx, data, lowerBounds, upperBounds, 
             Xdim_indicator, dim, kn, projDim):
        """
        Grows the partitioning tree by recursively splitting the dataset. 
        Fixes the 'TypeError' by ensuring 'nodeIdx' is always a NumPy array.

        Parameters
        ----------
        parent : TSPNode or None
            The parent node in the tree. If None, this node may be the root.
        nodeIdx : array-like of int
            Indices of samples to be partitioned at the current node.
        data : np.ndarray
            The full dataset of shape (N, dim).
        lowerBounds : np.ndarray
            1D array of length 'dim', the lower bounds of the partition.
        upperBounds : np.ndarray
            1D array of length 'dim', the upper bounds of the partition.
        Xdim_indicator : list of bool
            A boolean list indicating which dimensions belong to X (True) or Y (False).
        dim : int
            Total number of dimensions in the data.
        kn : float or int
            Critical mass threshold for deciding whether to split further.
        projDim : int
            The dimension index used for the current split operation.

        Returns
        -------
        TSPNode
            The newly created node (this node) after potential recursive splitting.
        """

        # Ensure nodeIdx is a NumPy array (important for advanced indexing)
        if not isinstance(nodeIdx, np.ndarray):
            nodeIdx = np.asarray(nodeIdx, dtype=int)

        # Create a new TSPNode with the given parent
        node = TSPNode(parent)

        # If we're at the root level (projDim == 0), set the parent to itself
        if projDim == 0:
            node.parent = node
            node.idx_marginal_samples_X = nodeIdx
            node.idx_marginal_samples_Y = nodeIdx
            node.n_marginal_samples_X = len(nodeIdx)
            node.n_marginal_samples_Y = len(nodeIdx)

        # Basic node properties
        node.n_samples = len(nodeIdx)
        node.lowerBounds = lowerBounds
        node.upperBounds = upperBounds

        # Compute conditional joint distribution (node samples / parent samples)
        node.condJointDist = node.n_samples / node.parent.n_samples

        # Compute conditional marginal product
        node.condMargProd = node.conditionalMarginalProd(
            data, lowerBounds, upperBounds, Xdim_indicator, dim, projDim
        )

        # If enough samples remain to split further
        if (node.n_samples // 2) >= kn:

            # Project data onto current dimension
            proj_axis = projDim % dim
            projectedData = data[nodeIdx, proj_axis]

            # Find the median index (partial sort)
            medianIdx = node.n_samples // 2
            idx_partitioned = np.argpartition(projectedData, medianIdx)

            # Split indices into two groups around the median
            leftNodeIdx = nodeIdx[idx_partitioned[:medianIdx]]
            rightNodeIdx = nodeIdx[idx_partitioned[medianIdx:]]

            # Compute partition boundary
            left_max = np.max(projectedData[idx_partitioned[:medianIdx]])
            right_min = np.min(projectedData[idx_partitioned[medianIdx:]])
            mean = (left_max + right_min) / 2

            # Update bounds for left and right children
            leftUpper = upperBounds.copy()
            leftUpper[proj_axis] = mean
            rightLower = lowerBounds.copy()
            rightLower[proj_axis] = mean

            # Save the partition info
            node.partitions = [(proj_axis, rightLower, leftUpper)]

            # Recursively grow left and right subtrees
            node.left = self.grow(
                node, leftNodeIdx, data,
                lowerBounds, leftUpper,
                Xdim_indicator, dim, kn, projDim + 1
            )

            node.right = self.grow(
                node, rightNodeIdx, data,
                rightLower, upperBounds,
                Xdim_indicator, dim, kn, projDim + 1
            )

            # Compute mutual information gains for this node
            node.relativeCMIgain = node.getCMIgain()
            node.absoluteCMIgain = node.relativeCMIgain * (node.n_samples / data.shape[0])

        return node

    def conditionalMarginalProd(self, data, lower_bounds, upper_bounds, Xdim_indicator, dim, proj_dim):
        """
        Numba-optimized version: uses the JIT-compiled filter_in_bounds(...) 
        to quickly filter samples for X or Y partitions.

        Parameters
        ----------
        data : np.ndarray
            The full dataset of shape (N, dim).
        lower_bounds : np.ndarray
            The lower bounds for each dimension of the current partition.
        upper_bounds : np.ndarray
            The upper bounds for each dimension of the current partition.
        Xdim_indicator : array-like of bool
            Which dimensions belong to X (True) or Y (False).
        dim : int
            The total dimensionality of the data.
        proj_dim : int
            The current projection (dimension index) used for partitioning.

        Returns
        -------
        float
            The ratio of filtered samples over the parent's marginal samples, 
            serving as the conditional marginal product for this node.
        """

        if proj_dim == 0:
            return 1.0

        # Ensure Xdim_indicator is a NumPy boolean array
        Xdim_indicator = np.asarray(Xdim_indicator, dtype=np.bool_)

        # Check if the partition is on X or Y
        is_x_partition = Xdim_indicator[(proj_dim - 1) % dim]

        if is_x_partition:
            # Use parent's marginal samples for Y
            self.idx_marginal_samples_Y = self.parent.idx_marginal_samples_Y
            self.n_marginal_samples_Y = self.parent.n_marginal_samples_Y

            # Filter parent's X
            parent_idx_X = np.asarray(self.parent.idx_marginal_samples_X, dtype=np.int32).ravel()
            filtered_idx = filter_in_bounds(
                data,
                parent_idx_X,
                lower_bounds,
                upper_bounds,
                Xdim_indicator,
                True  # we are checking X dimensions
            )

            self.idx_marginal_samples_X = filtered_idx
            self.n_marginal_samples_X = len(filtered_idx)

            # Return proportion of samples
            return self.n_marginal_samples_X / self.parent.n_marginal_samples_X

        else:
            # Use parent's marginal samples for X
            self.idx_marginal_samples_X = self.parent.idx_marginal_samples_X
            self.n_marginal_samples_X = self.parent.n_marginal_samples_X

            # Filter parent's Y
            parent_idx_Y = np.asarray(self.parent.idx_marginal_samples_Y, dtype=np.int32).ravel()
            filtered_idx = filter_in_bounds(
                data,
                parent_idx_Y,
                lower_bounds,
                upper_bounds,
                Xdim_indicator,
                False  # we are checking Y dimensions
            )

            self.idx_marginal_samples_Y = filtered_idx
            self.n_marginal_samples_Y = len(filtered_idx)

            return self.n_marginal_samples_Y / self.parent.n_marginal_samples_Y

    def getCMIgain(self):
        """
        Compute the local conditional mutual information (CMI) gain at this node
        by evaluating its left and right children.

        Returns
        -------
        float
            The sum of CMI contributions from left and right child nodes.
        """
        left_CMIgain = self.left.condJointDist * math.log2(self.left.condJointDist / self.left.condMargProd)
        right_CMIgain = self.right.condJointDist * math.log2(self.right.condJointDist / self.right.condMargProd)
        return left_CMIgain + right_CMIgain

    def getPartitions(self):
        """
        Recursively gather all partition boundaries from this node downward.

        Returns
        -------
        list
            A list of partition tuples (dimension, lower_bounds, upper_bounds).
        """
        if self.left is None and self.right is None:
            return []
        else:
            return self.partitions + self.left.getPartitions() + self.right.getPartitions()

    def getEMI(self):
        """
        Recursively compute the empirical mutual information from this node
        by adding the current CMI gain and weighting children's EMI values.

        Returns
        -------
        float
            The accumulated EMI from this node's level downward.
        """
        if self.left is None and self.right is None:
            return 0
        else:
            return (
                self.relativeCMIgain
                + (self.left.condJointDist * self.left.getEMI())
                + (self.right.condJointDist * self.right.getEMI())
            )

    def getSize(self):
        """
        Recursively count the number of leaf nodes in the subtree rooted at this node.

        Returns
        -------
        int
            The count of leaves under this node, including itself if it is a leaf.
        """
        if self.left is None and self.right is None:
            return 1
        else:
            return self.left.getSize() + self.right.getSize()
