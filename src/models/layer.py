class Layer:
    # Abstract Class Layer
    def __init__(self):
        self.params = {}
        self.grads = {}
        self.last_gradients = {} 
    
    def forward(self, x):
        raise NotImplementedError
    
    def backward(self, grad_output):
        raise NotImplementedError
    
    def get_params(self):
        return self.params
    
    def get_grads(self):
        return self.grads
    
    def get_last_gradients(self):
        return self.last_gradients