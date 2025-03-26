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

    def train_step(self, x_batch, y_batch, learning_rate, reg_type=None, lambda_val=0.01):
        # forward pass
        y_pred, activations = self.forward(x_batch)
        
        # compute loss
        loss = self.loss_fn(y_pred, y_batch)
        
        # backward pass
        self.backward(y_pred, y_batch, activations)
        
        # tambahkan regularisasi
        reg_loss = 0
        if reg_type == 'l1':
            reg_loss = Regularization.l1_regularization(self, lambda_val)
        elif reg_type == 'l2':
            reg_loss = Regularization.l2_regularization(self, lambda_val)
        
        # update bobot dengan gradient descent
        for layer in self.layers:
            if isinstance(layer, (LinearLayer, RMSNorm)):
                for param_name in layer.params:
                    layer.params[param_name] -= learning_rate * layer.grads[param_name]
                    # Reset gradien
                    layer.grads[param_name] = np.zeros_like(layer.grads[param_name])
        
        # total loss (training + regularisasi)
        total_loss = loss + reg_loss
        
        return total_loss
    
    def train(self, x_train, y_train, batch_size=32, learning_rate=0.01, 
            epochs=100, x_val=None, y_val=None, verbose=1, 
            reg_type=None, lambda_val=0.01):
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
                
                # gunakan reg_type yang sesuai
                batch_loss = self.train_step(x_batch, y_batch, learning_rate, 
                                            reg_type, lambda_val)
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
    
    def visualize_model(self):
        """
        Visualisasi FFNN
        """
        import plotly.graph_objects as go
        import numpy as np
        from IPython.display import display, HTML
        
        # Jarak antar layer, bisa diubah kalo mau
        GAP_MULTIPLIER = 50  # makin gede angkanya, makin jauh jaraknya
        
        # Nama-nama layer
        layer_names = ["Input Layer"]
        for i in range(1, len(self.layer_sizes) - 1):
            layer_names.append(f"Hidden Layer {i}")
        layer_names.append("Output Layer")
        
        # Inisialisasi data nodes dan edges
        nodes = []
        edges = []
        node_id_map = {}  # Buat nyimpen mapping indeks node
        node_idx = 0
        
        # Nyari ukuran layer terbesar buat ngatur skala
        max_layer_size = max(self.layer_sizes)
        
        # Bikin node buat tiap neuron di semua layer
        for layer_idx, (layer_size, layer_name) in enumerate(zip(self.layer_sizes, layer_names)):
            # Posisi horizontal pake GAP_MULTIPLIER
            x = layer_idx * GAP_MULTIPLIER
            
            for neuron_idx in range(layer_size):
                # Ngatur jarak vertikal antar neuron
                y_spacing = 2.5  # jarak antar node
                if layer_size == 1:
                    y = 0
                else:
                    y = (neuron_idx - (layer_size - 1) / 2) * y_spacing
                
                # Bikin label buat tiap node
                if layer_idx == 0:  # kalo layer input
                    label = f"Variable #{neuron_idx+1}"
                elif layer_idx == len(self.layer_sizes) - 1:  # kalo layer output
                    label = "Output"
                else:  # kalo hidden layer
                    label = f"H{layer_idx}_{neuron_idx+1}"
                
                # Ngasih warna beda buat tiap jenis layer
                if layer_idx == 0:
                    color = "lightblue"
                elif layer_idx == len(self.layer_sizes) - 1:
                    color = "lightgreen"
                else:
                    color = "lightyellow"
                
                nodes.append({
                    "id": node_idx,
                    "label": label,
                    "x": x,
                    "y": y,
                    "size": 15,
                    "color": color,
                    "layer": layer_name,
                    "neuron_idx": neuron_idx
                })
                
                node_id_map[(layer_idx, neuron_idx)] = node_idx
                node_idx += 1
        
        # Bikin edges dengan info bobot dan gradien
        for layer_idx in range(len(self.layer_sizes) - 1):
            from_size = self.layer_sizes[layer_idx]
            to_size = self.layer_sizes[layer_idx + 1]
            
            # Nyari LinearLayer yang sesuai
            linear_layer = None
            for l in self.layers:
                if isinstance(l, LinearLayer) and l.in_features == from_size and l.out_features == to_size:
                    linear_layer = l
                    break
            
            if linear_layer is None:
                continue
            
            # Ambil bobot dan gradien
            weights = linear_layer.params['W']
            gradients = linear_layer.grads['W'] if 'W' in linear_layer.grads else np.zeros_like(weights)
            
            # Cari activation function
            activation = "None"
            for l_idx, l in enumerate(self.layers):
                if l is linear_layer and l_idx + 1 < len(self.layers):
                    next_layer = self.layers[l_idx + 1]
                    if hasattr(next_layer, 'activation_fn'):
                        activation = next_layer.activation_fn.__name__
                    break
            
            for i in range(from_size):
                for j in range(to_size):
                    source = node_id_map[(layer_idx, i)]
                    target = node_id_map[(layer_idx + 1, j)]
                    weight = float(weights[i, j])
                    gradient = float(gradients[i, j])
                    
                    # Ngatur ketebalan garis berdasarkan bobot
                    width = min(max(0.5, abs(weight) * 3), 5)
                    
                    # Kasih warna beda berdasarkan tanda bobot
                    color = 'rgba(123, 165, 209, 0.8)' if weight >= 0 else 'rgba(217, 123, 106, 0.8)'
                    
                    edges.append({
                        "source": source,
                        "target": target,
                        "weight": weight,
                        "gradient": gradient,
                        "width": width,
                        "color": color,
                        "source_layer": layer_names[layer_idx],
                        "target_layer": layer_names[layer_idx + 1],
                        "source_neuron": nodes[source]["label"],
                        "target_neuron": nodes[target]["label"],
                        "activation": activation
                    })
        
        # Bikin trace buat node
        node_x = [node["x"] for node in nodes]
        node_y = [node["y"] for node in nodes]
        node_text = [f"{node['label']}<br>Layer: {node['layer']}" for node in nodes]
        node_colors = [node["color"] for node in nodes]
        
        node_trace = go.Scatter(
            x=node_x, 
            y=node_y,
            mode='markers',
            hoverinfo='text',
            text=node_text,
            marker=dict(
                color=node_colors,
                size=25,
                line=dict(width=2, color='black')
            )
        )
        
        # Bikin trace buat tiap edge
        edge_traces = []
        
        for edge in edges:
            source_node = nodes[edge["source"]]
            target_node = nodes[edge["target"]]
            
            # Ambil titik awal dan akhir
            x0, y0 = source_node["x"], source_node["y"]
            x1, y1 = target_node["x"], target_node["y"]
            
            # Bikin banyak titik di sepanjang garis biar gampang di-hover
            # Pake 20 titik aja harusnya cukup
            num_points = 20
            edge_x = []
            edge_y = []
            for i in range(num_points):
                ratio = i / (num_points - 1)
                edge_x.append(x0 * (1 - ratio) + x1 * ratio)
                edge_y.append(y0 * (1 - ratio) + y1 * ratio)
            
            # Format nilai bobot dan gradien
            weight_display = f"{edge['weight']:.4f}"
            gradient_display = f"{edge['gradient']:.6f}"
            
            # Bikin text buat hover
            hover_text = (
                f"<b>Edge:</b> {edge['source_neuron']} → {edge['target_neuron']}<br>"
                f"<b>Weight:</b> {weight_display}<br>"
                f"<b>Gradient:</b> {gradient_display}<br>"
                f"<b>Aktivasi:</b> {edge['activation']}"
            )
            
            # Pake hovertemplate biar tampilannya lebih bagus
            edge_trace = go.Scatter(
                x=edge_x, 
                y=edge_y,
                line=dict(width=edge["width"], color=edge["color"]),
                mode='lines',
                hoverinfo='text',
                hovertemplate=hover_text + "<extra></extra>",
                hoverlabel=dict(
                    bgcolor="white",
                    font_size=12,
                    font_family="Arial",
                    bordercolor="black"
                ),
                opacity=0.7
            )
            
            edge_traces.append(edge_trace)
        
        # Gabungin jadi satu figure
        fig = go.Figure(
            data=edge_traces + [node_trace],
            layout=go.Layout(
                title='Visualisasi Arsitektur Neural Network dengan Bobot dan Gradien',
                showlegend=False,
                hovermode='closest',
                hoverdistance=100,  # Naikin sensitivitas hover
                margin=dict(b=20, l=5, r=5, t=40),
                xaxis=dict(
                    showgrid=False, 
                    zeroline=False, 
                    showticklabels=False,
                    range=[-0.5, (len(self.layer_sizes) - 1) * GAP_MULTIPLIER + 0.5]
                ),
                yaxis=dict(
                    showgrid=False, 
                    zeroline=False, 
                    showticklabels=False,
                    scaleanchor="x", 
                    scaleratio=1
                ),
                width=1200,
                height=800,
                plot_bgcolor='rgba(240, 240, 240, 0.2)'
            )
        )
        
        # Tambah label buat tiap layer
        for i, name in enumerate(layer_names):
            fig.add_annotation(
                x=i * GAP_MULTIPLIER,
                y=(max_layer_size - 1) * 2.5 / 2 + 3,
                text=name,
                showarrow=False,
                font=dict(size=16, color="black")
            )
        
        # Nambahin keterangan warna
        fig.add_annotation(
            x=0.02,
            y=0.02,
            xref="paper",
            yref="paper",
            text="Garis biru: Bobot positif | Garis merah: Bobot negatif<br>Ketebalan garis menunjukkan besarnya weight",
            showarrow=False,
            font=dict(size=12),
            bgcolor="rgba(255, 255, 255, 0.7)",
            bordercolor="gray",
            borderwidth=1,
            borderpad=4,
            align="left"
        )
        

        # Tampilin visualisasi
        fig.show(config={
            'displayModeBar': True,
            'scrollZoom': True,
            'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'eraseshape'],
            'toImageButtonOptions': {
                'format': 'png',
                'filename': 'visualisasi_neural_network',
                'height': 800,
                'width': 1200,
                'scale': 2
            }
        })
        

        # # Simpan visualisasi ke file HTML
        # html_file = "visualisasi_neural_network.html"
        # fig.write_html(html_file)
        # print(f"Visualisasi disimpan ke file {html_file} - buka file ini di browser untuk fitur interaktif lengkap")


