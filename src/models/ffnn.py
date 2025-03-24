from .activation_functions import Activations
from .loss_functions import Losses
from .weight_initializers import Initializers
from .layer import Layer
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

    def forward(self, x):
        # pass through each layer
        activations = [x]
        
        for layer in self.layers:
            x = layer.forward(x)
            activations.append(x)
        
        return x, activations
    
    def backward(self, y_pred, y_true, activations):
        grad = self.loss_derivative(y_pred, y_true)
        
        # backpropagate through layers in reverse order
        for i in range(len(self.layers) - 1, -1, -1):
            grad = self.layers[i].backward(grad)
    
    def get_params(self):
        # get all trainable parameters
        params = {}
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'get_params'):
                for name, param in layer.get_params().items():
                    params[f"layer{i}_{name}"] = param
        return params
    
    def get_gradients(self):
        # get all parameter gradients
        grads = {}
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'get_grads'):
                for name, grad in layer.get_grads().items():
                    grads[f"layer{i}_{name}"] = grad
        return grads

    def train_step(self, x_batch, y_batch, learning_rate):
        # forward pass
        y_pred, activations = self.forward(x_batch)
        
        # compute loss
        loss = self.loss_fn(y_pred, y_batch)
        
        # backward pass
        self.backward(y_pred, y_batch, activations)
        
        # bpdate weights using gradient descent
        for layer in self.layers:
            if isinstance(layer, (LinearLayer, RMSNorm)):
                for param_name in layer.params:
                    layer.params[param_name] -= learning_rate * layer.grads[param_name]
                    # Reset gradients
                    layer.grads[param_name] = np.zeros_like(layer.grads[param_name])
        
        return loss
    
    def train(self, x_train, y_train, batch_size=32, learning_rate=0.01, 
              epochs=100, x_val=None, y_val=None, verbose=1):
        n_samples = len(x_train)
        history = {'train_loss': [], 'val_loss': []}
        
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        for epoch in range(epochs):
            total_loss = 0
            
            # shuffle
            indices = np.random.permutation(n_samples)
            x_shuffled = x_train[indices]
            y_shuffled = y_train[indices]
            
            # mini-batch training
            for batch in range(n_batches):
                start_idx = batch * batch_size
                end_idx = min((batch + 1) * batch_size, n_samples)
                
                x_batch = x_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                
                batch_loss = self.train_step(x_batch, y_batch, learning_rate)
                total_loss += batch_loss * (end_idx - start_idx)
                
                if verbose > 1 and batch % max(1, n_batches // 10) == 0:
                    print(f"Epoch {epoch+1}/{epochs}, Batch {batch+1}/{n_batches}, Loss: {batch_loss:.4f}")
            
            # average training loss
            avg_train_loss = total_loss / n_samples
            history['train_loss'].append(avg_train_loss)
            
            # validation loss if validation data is provided
            val_loss = None
            if x_val is not None and y_val is not None:
                y_val_pred, _ = self.forward(x_val)
                val_loss = self.loss_fn(y_val_pred, y_val)
                history['val_loss'].append(val_loss)
            
            if verbose > 0:
                if val_loss is not None:
                    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")
                else:
                    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}")
        
        return history
    
    def predict(self, x):
        predictions, _ = self.forward(x)
        return predictions