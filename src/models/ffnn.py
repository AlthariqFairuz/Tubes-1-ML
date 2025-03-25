import pickle

from matplotlib import pyplot as plt
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
    
    
    def save(self, filepath):
        # Save  ke file
        data = {
            'layer_sizes': self.layer_sizes,
            'params': {}
        }
        
        # Save all layer param
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'params'):
                for param_name, param in layer.params.items():
                    data['params'][f"layer{i}_{param_name}"] = param
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
    def load(self, filepath):
        # load dari filepath
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # Verif
        if model_data['layer_sizes'] != self.layer_sizes:
            raise ValueError("Model architecture does not match saved model")
        
        # Load
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'params'):
                for param_name in layer.params:
                    param_key = f"layer{i}_{param_name}"
                    if param_key in model_data['params']:
                        layer.params[param_name] = model_data['params'][param_key]

    def plot_weight_distribution(self, layers=None):
        if layers is None:
            layers = range(len(self.layers))
        
        # Cari jumlah layer
        param_layers = [i for i, layer in enumerate(self.layers) if hasattr(layer, 'params')]
        
        plot_layers = [i for i in layers if i in param_layers]
        
        if not plot_layers:
            print("No layers with parameters to plot")
            return
        
        fig, axes = plt.subplots(len(plot_layers), 1, figsize=(10, 3*len(plot_layers)))
        if len(plot_layers) == 1:
            axes = [axes]
        
        for i, layer_idx in enumerate(plot_layers):
            layer = self.layers[layer_idx]
            
            for name, param in layer.params.items():
                # Pemerataan bobot
                weights = param.flatten()
                axes[i].hist(weights, bins=50, alpha=0.5, label=name)
            
            axes[i].set_title(f"Distribusi bobot Layer {layer_idx}")
            axes[i].set_xlabel("Weight")
            axes[i].set_ylabel("Frequency")
            axes[i].legend()
        
        plt.tight_layout()
        plt.show()

    def plot_gradient_distribution(self, layers=None):
        if layers is None:
            layers = range(len(self.layers))
        
        #  Cari jumlah layer
        param_layers = [i for i, layer in enumerate(self.layers) if hasattr(layer, 'grads')]
        
        plot_layers = [i for i in layers if i in param_layers]
        
        if not plot_layers:
            print("No layers with parameters to plot")
            return
        
        fig, axes = plt.subplots(len(plot_layers), 1, figsize=(10, 3*len(plot_layers)))
        if len(plot_layers) == 1:
            axes = [axes]
        
        for i, layer_idx in enumerate(plot_layers):
            layer = self.layers[layer_idx]
            
            for name, grad in layer.grads.items():
                # Pemerataan gradien
                grads = grad.flatten()
                axes[i].hist(grads, bins=50, alpha=0.5, label=name)
            
            axes[i].set_title(f"Distribusi Gradien Layer {layer_idx} ")
            axes[i].set_xlabel("Gradien")
            axes[i].set_ylabel("Frequency")
            axes[i].legend()
        
        plt.tight_layout()
        plt.show()

    def plot_loss(self, history):
        plt.figure(figsize=(10, 5))
        
        # plot training loss
        plt.plot(history['train_loss'], label='Training Loss', marker='o')
        
        # plot validation loss if available
        if 'val_loss' in history and history['val_loss']:
            plt.plot(history['val_loss'], label='Validation Loss', marker='o')
        
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
        return history

    def print_model(self):
        # Display struktur FFNN nya
        print("FFNN")
        print(f"Layer sizes: {self.layer_sizes}")

        # Iterasi tiap layer
        for i, layer in enumerate(self.layers):
            if isinstance(layer, LinearLayer):
                print(f"\nLayer {i} (Linear):")
                print(f"  Input size: {layer.in_features}")
                print(f"  Output size: {layer.out_features}")
                
                # Ingfo bobot
                W = layer.params['W']
                print(f"  Weights shape: {W.shape}")
                print(f"  Weights range: [{W.min():.4f}, {W.max():.4f}]")
                print(f"  Weights mean: {W.mean():.4f}")
                print(f"  Weights std: {W.std():.4f}")
                
                # Ingfo gradien
                if np.any(layer.grads['W']):
                    print(f"  Gradient range: [{layer.grads['W'].min():.4f}, {layer.grads['W'].max():.4f}]")
                    print(f"  Gradient mean: {layer.grads['W'].mean():.4f}")
                    print(f"  Gradient std: {layer.grads['W'].std():.4f}")
                
                # Ingfo bias
                if layer.use_bias:
                    b = layer.params['b']
                    print(f"  Bias shape: {b.shape}")
                    print(f"  Bias range: [{b.min():.4f}, {b.max():.4f}]")
            
            elif isinstance(layer, ActivationLayer):
                print(f"\nLayer {i} (Activation):")
                print(f"  Type: {layer.activation_fn.__name__}")
            
            elif isinstance(layer, RMSNorm):
                print(f"\nLayer {i} (RMSNorm):")
                scale = layer.params['scale']
                print(f"  Scale shape: {scale.shape}")
                print(f"  Scale range: [{scale.min():.4f}, {scale.max():.4f}]")
