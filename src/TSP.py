"""
Tree Structured Partition class, represents a tree (set of nodes) and the values associated with the MI estimation problem, such
as empirical MI, regularized MI and tree size

Maximiliano Rojas - Information and Decision System Group - Universidad de Chile
"""

import numpy as np
import matplotlib.pyplot as plt
from TSPNode import TSPNode


class TSP:
    def __init__(self, l_bn, w_bn, _lambda):
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
        # Convert inputs to numpy arrays if they aren't already
        x = np.array(x)
        y = np.array(y)

        # Data array
        data = np.concatenate((x, y), axis=1)

        # Number of samples and dimensions of data
        self.n_samples = data.shape[0]
        self.dim = data.shape[1]
        
        # Define a root node and determine the critical mass according to its formula
        self.root = TSPNode()
        self.kn = np.ceil(self.w_bn * pow(self.n_samples, 1 - self.l_bn))

        # Initial lower and upper bounds of the whole sample space
        lowerBounds = np.min(data, axis=0) - 0.001
        upperBounds = np.max(data, axis=0) + 0.001

        # Two lists: one which indicates whose dimensions came from the vector X and the other has the indices of every sample in the whole space
        Xdim_indicator = [True] * x.shape[1] + [False] * y.shape[1]
        nodeIdx = list(range(self.n_samples))


        # Tree growth begins at the root node
        self.root = self.root.grow(None, nodeIdx, data, lowerBounds, upperBounds, Xdim_indicator, self.dim, self.kn, projDim=0)
        

        # Once the tree is fully grown obtain the TSP results
        self.tsp_emi = self.root.getEMI()
        self.tsp_size = self.root.getSize()
        self.tsp_partitions = self.root.getPartitions()


    def regularize(self):

        # A grown tree is required
        if self.root is None:
            raise ValueError("Observations not provided.")
        

        # Size of the full grown tree
        full_tree_size = self.tsp_size


        # Regularizer terms that balances the cost and penalization associated with complexity regularization
        bn = self.w_bn * np.power(self.n_samples, -self.l_bn)
        inv_deltan = np.exp(np.power(self.n_samples, 1 / 3.0))
        

        # Initialize array which will have the EMI for every minimum cost tree and fill it with the method
        treesEMI = np.zeros(full_tree_size)
        self.minimum_cost_trees(treesEMI, full_tree_size)
        

        # Initialize cost and tree size for the optimization problem
        optimal_cost = -treesEMI[0]
        optimal_size = 1


        # Solve for every minimum cost tree
        for k in range(2, full_tree_size + 1):

            # Optimization problem
            epsilon = (12 / bn) * np.sqrt((8.0 / self.n_samples) * (np.log(8 * inv_deltan) + k * ((self.dim + 1) * np.log(2) + self.dim * np.log(self.n_samples))))
            cost = -treesEMI[k-1] + self._lambda * epsilon


            # Check if the cost is less than the optimal cost
            if cost < optimal_cost:
                optimal_cost = cost
                optimal_size = k
        
        # Once the tree is prunned obtain the TSP regularization results
        self.tsp_reg_emi = -optimal_cost
        self.tsp_reg_size = optimal_size 


    def minimum_cost_trees(self, treesEMI, full_tree_size):

        # An auxiliary array that will store the leaves of each minimum cost tree in every iteration. 
        # The goal is to sort these leaves in ascending order, according to their absolute CMI gain
        leaves = [None] * full_tree_size
        leaves[0] = self.root


        # Obtain the EMI for every possible minimum cost tree
        for k in range(full_tree_size - 1):

            # Leaf of the tree of size k which has the maximum absolute CMI.
            leaf_maxCMI = leaves[k]


            # The EMI of the tree of size k+1 is calculated by adding the absolute CMI of the leaf with the highest CMI from the
            # tree of size k to the EMI of the tree of size k
            treesEMI[k + 1] = treesEMI[k] + leaf_maxCMI.absoluteCMIgain


            # Next tree is obtained by the split of the maximum absolute CMI leaf

            # The left child will sustitute the parent in the leaves array and the 
            # right child will be simply appended to the leaves array
            self.subadditive_insert(leaf_maxCMI.left, leaves, k)

            self.subadditive_insert(leaf_maxCMI.right, leaves, k + 1)


    def subadditive_insert(self, new_leaf, leaves, j):

        # Storage the new leaf
        leaves[j] = new_leaf


        # Bubble sort algorithm to order the leaves in ascending order according to their absolute CMI gain
        for k in range(j, 0, -1):

            # Check if the previous leaf has a higher absolute CMI; if so, they must swap places
            if  leaves[k-1].absoluteCMIgain > leaves[k].absoluteCMIgain:
                leaves[k], leaves[k-1] = leaves[k-1], leaves[k]  


    def visualize(self, x, y):

        # A grown tree is required
        if self.root is None:
            raise ValueError("Observations not provided.")


        # Check if the problem is two dimensional
        if self.dim != 2:
            raise ValueError("The tree can only be visualized for a two dimensional problem.")


        # Partitions
        partitions = self.partitions()


        # Initialize lists for the partitions in every dimension
        horizontal_partition = [partition[1:] for partition in partitions if partition[0] == 0]
        vertical_partition = [partition[1:] for partition in partitions if partition[0] == 1]


        # Create a figure and axis object
        fig, ax = plt.subplots()


        # Plot sample points
        ax.scatter(x, y, marker='o', edgecolor='k', s=25, alpha=0.5)


        # Plot horizontal lines
        for bound in vertical_partition:
            ax.hlines(y=bound[0][1], xmin=bound[0][0], xmax=bound[1][0], color='black')


        # Plot vertical lines
        for bound in horizontal_partition:
            ax.vlines(x=bound[0][0], ymin=bound[0][1], ymax=bound[1][1], color='black')


        # Set labels and title
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('TSP: n={}, k_n={}, size={}'.format(self.n_samples, int(self.kn), self.size()))


        # Show plot
        plt.grid(True)
        plt.show()


    def emi(self):
        if self.root is None:
            raise ValueError("Observations not provided.")
        return self.tsp_emi
    

    def reg_emi(self):
        if self.root is None:
            raise ValueError("Observations not provided.")
        return self.tsp_reg_emi
    

    def size(self):
        if self.root is None:
            raise ValueError("Observations not provided.")
        return self.tsp_size
    

    def reg_size(self):
        if self.root is None:
            raise ValueError("Observations not provided.")
        return self.tsp_reg_size
    
    
    def partitions(self):
        if self.root is None:
            raise ValueError("Observations not provided.")
        return self.tsp_partitions