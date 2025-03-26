import numpy as np

class Activations:
    @staticmethod
    def linear(x):
        return x
    
    @staticmethod
    def linear_derivative(x):
        return np.ones_like(x)
    
    @staticmethod
    def relu(x):
        return np.maximum(0, x)
    
    @staticmethod
    def relu_derivative(x):
        return (x > 0).astype(float)
    
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))  # avoiding overflow
    
    @staticmethod
    def sigmoid_derivative(x):
        s = Activations.sigmoid(x)
        return s * (1 - s)
    
    @staticmethod
    def tanh(x):
        return np.tanh(x)
    
    @staticmethod
    def tanh_derivative(x):
        return 1 - np.tanh(x)**2
    
    @staticmethod
    def softmax(x, axis=-1):
        x_max = np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
    
    @staticmethod
    #TODO
    def softmax_derivative(x, axis=-1):
        """
        In neural networks, when softmax is the final layer and is used with 
        categorical cross-entropy loss, the gradient simplifies to (output - target),
        which is handled in the categorical_cross_entropy_derivative function.
        
        This function is not directly used in backpropagation when softmax is combined
        with categorical cross-entropy, as the ActivationLayer.backward method bypasses it.
        
        Args:
            x: Input to the softmax function
            axis: Axis along which softmax is computed
            
        Returns:
            Jacobian-vector product function for backpropagation
        """
        s = Activations.softmax(x, axis=axis)
        return s
    
    @staticmethod
    def leaky_relu(x):
        return np.where(x > 0, x, x * 0.01)
    
    @staticmethod
    def leaky_relu_derivative(x):
        return np.where(x > 0, 1, 0.01)
    
    @staticmethod
    def exponential_relu(x):
        return np.where(x > 0, x, np.exp(x) - 1)
    
    @staticmethod
    def exponential_relu_derivative(x):
        return np.where(x > 0, 1, np.exp(x))
