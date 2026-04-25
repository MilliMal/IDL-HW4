import numpy as np
from .activation import Softmax

class ScaledDotProductAttention:
    """
    Scaled Dot Product Attention
    """ 
    def __init__(self):
        '''
        Initialize the ScaledDotProductAttention class.
        '''
        # Initialize your softmax layer
        # What dimension should you pass to the softmax constructor?
        self.eps = 1e10 # DO NOT MODIFY
        self.softmax = Softmax()
        
    
    def forward(self, Q, K, V, mask=None):
        """
        :param Q: Query matrix of shape (N, ..., H, L, E) where L is target sequence length
        :param K: Key matrix of shape (N, ..., H, S, E) where S is source sequence length
        :param V: Value matrix of shape (N, ..., H, S, Ev) where Ev is value dimension
        :param mask: Boolean mask matrix of shape (N, ..., H, L, S) or broadcastable shape where 1/True indicates a position to ignore
        :return: Output matrix of shape (N, ..., H, L, Ev)
        """
        # TODO: Implement forward pass
        self.Q = Q
        self.K = K
        self.V = V

        dk = K.shape[-1]
        # Calculate attention scores
        scaled_dot_product = np.matmul(Q, np.swapaxes(K, -2, -1)) / np.sqrt(dk)  
        # Apply mask before softmax if provided
        if mask is not None:
            scaled_dot_product += (mask * -self.eps)  # Masked positions will have very low scores after softmax

        # Compute attention scores: 
        # # Think about which dimension you should apply Softmax
        self.attention_scores = self.softmax.forward(scaled_dot_product)  # Shape: (N, ..., H, L, S)

        # Calculate final output
        output = np.matmul(self.attention_scores, V)  # Shape: (N, ..., H, L, Ev)
        # Return final output
        return output

    def backward(self, d_output):
        """
        :param d_output: Gradient of loss wrt output of shape (N, ..., H, L, Ev)
        :return: Gradient of loss wrt input Q, K, V
        """
        # TODO: Implement backward pass

        # Calculate gradients for V
        transformed_attention_scores = np.swapaxes(self.attention_scores, -2, -1)  # Shape: (N, ..., H, S, L)
        d_V = np.matmul(transformed_attention_scores, d_output)  # Shape: (N, ..., H, S, Ev)
        V_transformed = np.swapaxes(self.V, -2, -1)      
        # Calculate gradients for attention scores
        d_attention_scores = np.matmul(d_output, V_transformed)
        d_scaled_dot_product = self.softmax.backward(d_attention_scores)  # Shape: (N, ..., H, L, S)
        
        # Scale gradients by sqrt(d_k)
        d_scaled_dot_product /= np.sqrt(self.K.shape[-1])


        # Calculate gradients for Q and K
        d_Q = np.matmul(d_scaled_dot_product, self.K)  # Shape: (N, ..., H, L, E)
        d_K = np.matmul(np.swapaxes(d_scaled_dot_product, -2, -1), self.Q)  # Shape: (N, ..., H, S, E)
        
        # Return gradients for Q, K, V
        return d_Q, d_K, d_V

