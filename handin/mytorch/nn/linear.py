import numpy as np

class Linear:
    def __init__(self, in_features, out_features):
        """
        Initialize the weights and biases with zeros
        W shape: (out_features, in_features)
        b shape: (out_features,)  # Changed from (out_features, 1) to match PyTorch
        """
        # DO NOT MODIFY
        self.W = np.zeros((out_features, in_features))
        self.b = np.zeros(out_features)


    def init_weights(self, W, b):
        """
        Initialize the weights and biases with the given values.
        """
        # DO NOT MODIFY
        self.W = W
        self.b = b

    def forward(self, A):
        """
        :param A: Input to the linear layer with shape (*, in_features)
        :return: Output Z with shape (*, out_features)
        
        Handles arbitrary batch dimensions like PyTorch
        """
        # TODO: Implement forward pass
        
        # Store input for backward pass
        self.A = A
        
        return np.dot(A, self.W.T) + self.b  # Shape: (*, out_features)

    def backward(self, dLdZ):
        """
        :param dLdZ: Gradient of loss wrt output Z (*, out_features)
        :return: Gradient of loss wrt input A (*, in_features)
        """
        # TODO: Implement backward pass

        dLdZ = dLdZ.reshape(-1, dLdZ.shape[-1])  # Reshape to 2D for matrix operations
        A = self.A.reshape(-1, self.A.shape[-1])  # Reshape to 2D for matrix operations
        # Compute gradients
        self.dLdA = np.dot(dLdZ, self.W)  # Shape: (*, in_features)
        self.dLdW = np.dot(dLdZ.T, A)  # Shape: (out_features, in_features)
        self.dLdb = np.sum(dLdZ, axis=0)  # Shape: (out_features,)
        self.dLdA = np.reshape(self.dLdA, self.A.shape)  # Reshape to match input shape
        
        # Return gradient of loss wrt input
        return self.dLdA
