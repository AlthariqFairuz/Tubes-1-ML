from activation_functions import Activations
from loss_functions import Losses
from weight_initializers import Initializers
from layer import Layer
import numpy as np

ACTIVATIONS = {
    'linear': (Activations.linear, Activations.linear_derivative),
    'relu': (Activations.relu, Activations.relu_derivative),
    'sigmoid': (Activations.sigmoid, Activations.sigmoid_derivative),
    'tanh': (Activations.tanh, Activations.tanh_derivative),
    'softmax': (Activations.softmax, Activations.softmax_derivative)
}

LOSSES = {
    'mse': (Losses.mse, Losses.mse_derivative),
    'binary_cross_entropy': (Losses.binary_cross_entropy, Losses.binary_cross_entropy_derivative),
    'categorical_cross_entropy': (Losses.categorical_cross_entropy, Losses.categorical_cross_entropy_derivative)
}

INITIALIZERS = {
    'zero': Initializers.zero_init,
    'uniform': Initializers.uniform_init,
    'normal': Initializers.normal_init,
    'xavier': Initializers.xavier_init,
    'he': Initializers.he_init
}

class LinearLayer(Layer):

    def __init__(self, in_features, out_features, bias=True, 
                 init_method='xavier', **init_kwargs):
        super().__init__()
        
        initializer = INITIALIZERS.get(init_method, INITIALIZERS['xavier'])
        
        self.params['W'] = initializer((in_features, out_features), **init_kwargs)
        self.grads['W'] = np.zeros((in_features, out_features))
        
        if bias:
            self.params['b'] = np.zeros(out_features)
            self.grads['b'] = np.zeros(out_features)
        
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias
        self.input = None