import numpy as np
from src.models.ffnn import ACTIVATIONS, INITIALIZERS


class Layer:
    # Abstract Class Layer
    def __init__(self):
        self.params = {}
        self.grads = {}
    
    def forward(self, x):
        raise NotImplementedError
    
    def backward(self, grad_output):
        raise NotImplementedError
    
    def get_params(self):
        return self.params
    
    def get_grads(self):
        return self.grads

# Dibagi jadi 2, Linear (Yang penjumlahan Sigma) dan Aktivasi  
class Linear(Layer):
    # y = x @ W + b

    def __init__(self, in_features, out_features, bias=True, 
                 init_method='xavier', **init_kwargs):
        super().__init__()
        
        initializer = INITIALIZERS.get(init_method, INITIALIZERS['xavier'])
        
        # Init bobot
        self.params['W'] = initializer((in_features, out_features), **init_kwargs)
        self.grads['W'] = np.zeros((in_features, out_features))
        
        # Init bias kalo ada
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
            y = y + self.params['b']
        
        return y
    
    def backward(self, grad_output):
        # Gradien bobot: dL/dW = x^T @ dL/dy
        self.grads['W'] = np.matmul(self.input.T, grad_output)
        
        # Gradient for bias: dL/db = sum(dL/dy, axis=0)
        if self.use_bias:
            self.grads['b'] = np.sum(grad_output, axis=0)
        
        # Gradient for input: dL/dx = dL/dy @ W^T
        grad_input = np.matmul(grad_output, self.params['W'].T)
        
        return grad_input

class Activation(Layer):
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
            # Belom kelar
            return grad_output
        else:
            return grad_output * self.activation_derivative(self.input)

class RMSNorm(Layer):
    # Normalisasi RMS
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
        
        # Normalisasi
        self.normalized = x / self.rms
        
        # Scale
        return self.normalized * self.params['scale']
    
    def backward(self, grad_output):
        # Gradien scaling
        self.grads['scale'] = np.sum(grad_output * self.normalized, axis=0)
        
        # Gradien RMS
        grad_normalized = grad_output * self.params['scale']
        
        # Gradien input
        n = self.input.shape[-1]
        grad_input = grad_normalized / self.rms
        
        grad_rms = -np.sum(grad_normalized * self.input * (self.rms ** -2), axis=-1, keepdims=True)
        grad_input += 2 * grad_rms * self.input / (n * self.rms)
        
        return grad_input
