import numpy as np
class Losses:
    @staticmethod
    def mse(y_pred, y_true):
        return np.mean((y_pred - y_true)**2)
    
    @staticmethod
    def mse_derivative(y_pred, y_true):
        return 2 * (y_pred - y_true) / y_true.size
    
    @staticmethod
    def binary_cross_entropy(y_pred, y_true):
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps) # avoid log(0)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    @staticmethod
    def binary_cross_entropy_derivative(y_pred, y_true):
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps) # avoid division by 0
        return (-(y_true / y_pred) + ((1 - y_true) / (1 - y_pred))) / y_pred.shape[0]
    
    @staticmethod
    def categorical_cross_entropy(y_pred , y_true):
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1.0) # avoid log(0)
        return -np.sum(y_true * np.log(y_pred)) / y_pred.shape[0]
    
    @staticmethod
    def categorical_cross_entropy_derivative(y_pred, y_true):
        pass