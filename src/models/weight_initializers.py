import numpy as np

class Initializers:
    @staticmethod
    def zero_init(shape):
        """Zero initialization - all weights set to 0"""
        return np.zeros(shape)

    @staticmethod
    def uniform_init(shape, low=-0.1, high=0.1, seed=None):
        """Uniform distribution initialization"""
        if seed is not None:
            np.random.seed(seed)
        return np.random.uniform(low, high, shape)

    @staticmethod
    def normal_init(shape, mean=0.0, var=0.1, seed=None):
        """Normal distribution initialization"""
        if seed is not None:
            np.random.seed(seed)
        return np.random.normal(mean, np.sqrt(var), shape)
    
    @staticmethod
    def xavier_init(shape, seed=None):
        """
        Xavier/Glorot initialization
        Helps maintain variance across layers for sigmoid/tanh activations
        """
        if seed is not None:
            np.random.seed(seed)
        
        fan_in = shape[0] if len(shape) >= 1 else 1
        fan_out = shape[1] if len(shape) >= 2 else 1
        
        limit = np.sqrt(6 / (fan_in + fan_out))
        return np.random.uniform(-limit, limit, shape)

    @staticmethod
    def he_init(shape, seed=None):
        """
        He initialization, optimize for ReLU activations
        """
        if seed is not None:
            np.random.seed(seed)
        
        fan_in = shape[0] if len(shape) >= 1 else 1
        std = np.sqrt(2 / fan_in)
        return np.random.normal(0.0, std, shape)