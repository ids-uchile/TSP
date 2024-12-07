"""
Tree Structured Partition Node class, represents a node (space partition) in the tree. 

Maximiliano Rojas - Information and Decision System Group - Universidad de Chile
"""

import numpy as np
import math

class TSPNode:
    def __init__(self, parent=None):
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


    def grow(self, parent, nodeIdx, data, lowerBounds, upperBounds, Xdim_indicator, dim, kn, projDim):

        # Define whos the node's parent
        node = TSPNode(parent)


        # Check if the node is the root; if so, set its parent to itself
        if projDim == 0:
            node.parent = node
            node.idx_marginal_samples_X = nodeIdx.copy()
            node.idx_marginal_samples_Y = nodeIdx.copy()
            node.n_marginal_samples_X = len(nodeIdx)
            node.n_marginal_samples_Y = len(nodeIdx)

        # Node's instance variables
        node.n_samples = len(nodeIdx)

        node.lowerBounds = lowerBounds
        node.upperBounds = upperBounds
        
        node.condJointDist = node.n_samples / node.parent.n_samples
        node.condMargProd = node.conditionalMarginalProd(data, lowerBounds, upperBounds, Xdim_indicator, dim, projDim)


        # Check if the node has the minimum samples required to be partitioned
        if node.n_samples // 2 >= kn:

            # Sample points projection into the target dimension
            projectedData = data[nodeIdx, projDim % dim]


            # Indices that sort the node samples
            idx = np.argsort(projectedData)
            

            # Map sorted indices back to the original array's indices
            nodeIdxSorted = np.array(nodeIdx)[idx]


            # Index of projectedData's median 
            medianIdx = len(nodeIdxSorted) // 2 
           

            # Space partition indices
            leftNodeIdx = nodeIdxSorted[:medianIdx]

            rightNodeIdx = nodeIdxSorted[medianIdx:]


            # Space partition samples
            leftNode = data[leftNodeIdx]

            rightNode = data[rightNodeIdx]


            # Limits of each partition
            left_max = max(leftNode[:, projDim % dim])
            right_min = min(rightNode[:, projDim % dim]) 
            mean = (left_max + right_min) / 2
    

            # Partitions
            leftUpperBounds = np.copy(upperBounds)      
            leftUpperBounds[projDim % dim] = mean

            rightLowerBounds = np.copy(lowerBounds)
            rightLowerBounds[projDim % dim] = mean


            # Save partitions
            node.partitions = [(projDim % dim, rightLowerBounds, leftUpperBounds)]


            # Node growth recursion
            node.left = self.grow(node, leftNodeIdx, data, lowerBounds, leftUpperBounds, Xdim_indicator, dim, kn, projDim + 1)
            
            node.right = self.grow(node, rightNodeIdx, data, rightLowerBounds, upperBounds, Xdim_indicator, dim, kn, projDim + 1)
            

            # Conditional mutual information gains
            node.relativeCMIgain = node.getCMIgain()

            node.absoluteCMIgain = node.relativeCMIgain * (node.n_samples / data.shape[0])
            
        return node
    
        
    def conditionalMarginalProd(self, data, lower_bounds, upper_bounds, Xdim_indicator, dim, proj_dim):
        if proj_dim == 0:
            return 1

        # Check if the following partition is on X
        if Xdim_indicator[(proj_dim - 1) % dim]:

            # Use parent's marginal samples for Y
            self.idx_marginal_samples_Y = self.parent.idx_marginal_samples_Y
            self.n_marginal_samples_Y = self.parent.n_marginal_samples_Y


            # Initialize marginal samples for X
            self.idx_marginal_samples_X = np.zeros(self.parent.n_marginal_samples_X, dtype=int)


            # Iterate through the parents marginal samples
            for i in range(self.parent.n_marginal_samples_X):

                # Iterate through dimensions
                for d in range(dim):

                    # Check if the dimension came from variable X
                    if Xdim_indicator[d]:

                        # Check if the sample is not in the 'marginal partition'
                        if (data[self.parent.idx_marginal_samples_X[i], d] < lower_bounds[d]) or (data[self.parent.idx_marginal_samples_X[i], d] > upper_bounds[d]):
                            # Sample is discarded
                            break

                    # Check if the dimension is the last one
                    if d == dim - 1:
                        # Sample is within bounds
                        self.idx_marginal_samples_X[self.n_marginal_samples_X] = self.parent.idx_marginal_samples_X[i]
                        self.n_marginal_samples_X += 1

            # Return proportion of samples
            return self.n_marginal_samples_X / self.parent.n_marginal_samples_X
        

        else:

            # Use parent's marginal samples for X
            self.idx_marginal_samples_X = self.parent.idx_marginal_samples_X
            self.n_marginal_samples_X = self.parent.n_marginal_samples_X


            # Initialize marginal samples for Y
            self.idx_marginal_samples_Y = np.zeros(self.parent.n_marginal_samples_Y, dtype=int)


            # Iterate through the parents marginal samples
            for i in range(self.parent.n_marginal_samples_Y):

                # Iterate through dimensions
                for d in range(dim):

                    # Check if the dimension came from variable Y
                    if not Xdim_indicator[d]:

                        # Check if the sample is not in the 'marginal partition'
                        if (data[self.parent.idx_marginal_samples_Y[i], d] < lower_bounds[d]) or (data[self.parent.idx_marginal_samples_Y[i], d] > upper_bounds[d]):
                            # Sample is discarded
                            break
                    
                    # Check if the dimension is the last one
                    if d == dim - 1:
                        # Sample is within bounds
                        self.idx_marginal_samples_Y[self.n_marginal_samples_Y] = self.parent.idx_marginal_samples_Y[i]
                        self.n_marginal_samples_Y += 1

            # Return proportion of samples
            return self.n_marginal_samples_Y / self.parent.n_marginal_samples_Y
        

    def getCMIgain(self):
        left_CMIgain = self.left.condJointDist * math.log2(self.left.condJointDist / self.left.condMargProd)
        
        right_CMIgain = self.right.condJointDist * math.log2(self.right.condJointDist / self.right.condMargProd)
        
        return left_CMIgain + right_CMIgain
    
        
    def getPartitions(self):
        if self.left is None and self.right is None:
            return []
        else:
            return self.partitions + self.left.getPartitions() + self.right.getPartitions()
        
        
    def getEMI(self):
        if self.left is None and self.right is None:
            return 0
        else:
            return self.relativeCMIgain + (self.left.condJointDist * self.left.getEMI()) + (self.right.condJointDist * self.right.getEMI())
        
        
    def getSize(self):
        if self.left is None and self.right is None:
            return 1
        else:
            return self.left.getSize() + self.right.getSize()