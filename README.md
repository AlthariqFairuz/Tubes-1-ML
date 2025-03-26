# Feedforward Neural Network Implementation

## Overview

This project implements a Feedforward Neural Network (FFNN) from scratch for the IF3270 Machine Learning course assignment. The implementation provides a flexible architecture for creating and training neural networks with customizable layers, activation functions, loss functions, and weight initialization methods.

## Features

- **Flexible Architecture**: Configure neural networks with arbitrary depth and width
- **Activation Functions**:
  - Linear
  - ReLU
  - Sigmoid
  - Hyperbolic Tangent (tanh)
  - Softmax
- **Loss Functions**:
  - Mean Squared Error (MSE)
  - Binary Cross-Entropy
  - Categorical Cross-Entropy
- **Weight Initialization Methods**:
  - Zero initialization
  - Random uniform distribution
  - Random normal distribution
  - Xavier/Glorot initialization
  - He initialization
- **Regularization Techniques**:
  - L1 regularization (Lasso)
  - L2 regularization (Ridge)
- **Normalization**:
  - RMSNorm (Root Mean Square Normalization)
- **Training Features**:
  - Mini-batch gradient descent
  - Configurable learning rate
  - Validation during training
  - Progress tracking and history

## Project Structure

```
.
├── __init__.py                # Package initialization file
├── activation_functions.py    # Activation functions implementation
├── ffnn.py                    # Main FFNN implementation and utility classes
├── layer.py                   # Base Layer class
├── loss_functions.py          # Loss functions implementation
└── weight_initializers.py     # Weight initialization methods
```

## Installation

1. Clone this repository:
```bash
git clone https://github.com/AlthariqFairuz/Tubes-1-ML.git
cd Tubes-1-ML
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Creating a Neural Network

```python
from ffnn import FFNN

# Define layer sizes (input, hidden layers, output)
layer_sizes = [784, 128, 64, 10]  # Example for MNIST

# Define activation functions for each layer except input
activations = ['relu', 'relu', 'softmax']

# Create the model
model = FFNN(
    layer_sizes=layer_sizes,
    activations=activations,
    loss='categorical_cross_entropy',
    weight_init='xavier'
)
```

### Training the Model

```python
# Prepare your data (X_train, y_train, X_val, y_val)
# ...

# Train the model
history = model.train(
    x_train=X_train,
    y_train=y_train,
    batch_size=32,
    learning_rate=0.01,
    epochs=10,
    x_val=X_val,
    y_val=y_val,
    verbose=1
)

# Making predictions
predictions = model.predict(X_test)
```

### Using Regularization

```python
from ffnn import FFNN, Regularization

# Create model
model = FFNN(...)

# Training with L2 regularization
history = model.train(...)
reg_loss = Regularization.l2_regularization(model, lambda_val=0.01)
```

## Examples

See the accompanying Jupyter notebooks in the `examples` folder for detailed usage examples including:

1. MNIST classification
2. Analysis of hyperparameters (depth, width, activation functions, learning rate)
3. Comparison with sklearn's MLPClassifier
4. Visualization of weight distributions
