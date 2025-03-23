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

    def forward(self, x):
        self.input = x
        y = np.matmul(x, self.params['W'])

        if self.use_bias:
            y += self.params['b']
        
        return y
    
    def backward(self, grad_output):
        # Gradient for weights: dL/dW = x^T @ dL/dy
        self.grads['W'] = np.matmul(self.input.T, grad_output)
        
        # Gradient for bias: dL/db = sum(dL/dy, axis=0)
        if self.use_bias:
            self.grads['b'] = np.sum(grad_output, axis=0)
        
        # Gradient for input: dL/dx = dL/dy @ W^T
        grad_input = np.matmul(grad_output, self.params['W'].T)
        
        return grad_input

class Regularization:
    @staticmethod
    def l1_regularization(model, lambda_val=0.01):
        """
        L1 regularization (Lasso)
        Adds λ * |w| to the loss
        
        Args:
            model: FFNN model
            lambda_val: Regularization strength
            
        Returns:
            Regularization loss and updates gradients in-place
        """
        reg_loss = 0
        
        for layer in model.layers:
            if isinstance(layer, LinearLayer):
                # L1 loss
                reg_loss += lambda_val * np.sum(np.abs(layer.params['W']))
                
                # L1 gradient: λ * sign(w)
                layer.grads['W'] += lambda_val * np.sign(layer.params['W'])
        
        return reg_loss

    @staticmethod
    def l2_regularization(model, lambda_val=0.01):
        """
        L2 regularization (Ridge)
        Adds λ * ||w||² / 2 to the loss
        
        Args:
            model: FFNN model
            lambda_val: Regularization strength
            
        Returns:
            Regularization loss and updates gradients in-place
        """
        reg_loss = 0
        
        for layer in model.layers:
            if isinstance(layer, LinearLayer):
                # L2 loss
                reg_loss += 0.5 * lambda_val * np.sum(layer.params['W']**2)
                
                # L2 gradient: λ * w
                layer.grads['W'] += lambda_val * layer.params['W']
        
        return reg_loss

class ActivationLayer(Layer):
    def __init__(self, activation_name):
        super().__init__()
        if isinstance(activation_name, str):
            self.activation_fn, self.activation_derivative = ACTIVATIONS[activation_name]
        else:
            self.activation_fn = activation_name
            self.activation_derivative = None 
        
        self.input = None
        self.output = None
    
    def forward(self, x):
        self.input = x
        self.output = self.activation_fn(x)
        return self.output
    
    def backward(self, grad_output):
        if self.activation_fn.__name__ == 'softmax':
            # Special case for softmax (combined with cross-entropy loss), when softmax is used with categorical cross-entropy, the gradient simplifies to (output - target)
            return grad_output
        else:
            # Element-wise multiplication of gradient and activation derivative
            return grad_output * self.activation_derivative(self.input)
        
class RMSNorm(Layer):
    """
    Root Mean Square Layer Normalization
    Simplified version of LayerNorm without mean-centering
    """
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.params['scale'] = np.ones(dim)
        self.grads['scale'] = np.zeros(dim)
        self.input = None
        self.rms = None
        self.normalized = None
    
    def forward(self, x):
        self.input = x
        
        # RMS
        variance = np.mean(x**2, axis=-1, keepdims=True)
        self.rms = np.sqrt(variance + self.eps)
        
        # normalize
        self.normalized = x / self.rms
        
        # scale
        return self.normalized * self.params['scale']
    
    def backward(self, grad_output):
        # Gradient for scale parameter
        self.grads['scale'] = np.sum(grad_output * self.normalized, axis=0)
        
        # Gradient for normalized input
        grad_normalized = grad_output * self.params['scale']
        
        # Gradient for input
        n = self.input.shape[-1]
        grad_input = grad_normalized / self.rms
        
        # Additional term from RMS normalization
        grad_rms = -np.sum(grad_normalized * self.input * (self.rms ** -2), axis=-1, keepdims=True)
        grad_input += 2 * grad_rms * self.input / (n * self.rms)
        
        return grad_input

class FFNN:
    def __init__(self, layer_sizes, activations, loss='mse', use_rmsnorm=False,
                    weight_init='xavier', **init_kwargs):
            """
            Initialize a new FFNN model
            
            Args:
                layer_sizes: List of integers specifying the number of neurons in each layer
                            (including input and output layers)
                activations: List of activation functions for each layer (except input)
                loss: Loss function to use
                use_rmsnorm: Whether to use RMSNorm after each layer
                weight_init: Weight initialization method
                **init_kwargs: Additional arguments for weight initialization
            """
            if len(layer_sizes) < 2:
                raise ValueError("Need at least input and output layers")
            
            if len(activations) != len(layer_sizes) - 1:
                raise ValueError(f"Number of activation functions ({len(activations)}) must match "
                                f"number of layers minus input layer ({len(layer_sizes) - 1})")
            
            self.layer_sizes = layer_sizes
            self.layers = []
            
            for i in range(len(layer_sizes) - 1):

                # add linear layer
                self.layers.append(
                    LinearLayer(layer_sizes[i], layer_sizes[i+1], 
                        init_method=weight_init, **init_kwargs)
                )
                
                if use_rmsnorm:
                    self.layers.append(RMSNorm(layer_sizes[i+1]))
                
                act_fn = activations[i] if isinstance(activations[i], str) else activations[i]
                self.layers.append(ActivationLayer(act_fn))
            
            self.loss_fn, self.loss_derivative = LOSSES[loss] if isinstance(loss, str) else loss
        