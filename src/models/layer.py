class Layer:
    """Abstract class for a layer."""
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