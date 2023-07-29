import numpy as np

class Activations:
    def sigmoid(x):
        return 1 / ( 1 + np.exp(-x))
    
    def tanh(x):
        return np.tanh(x)
    
    def ReLU(x):
        return np.maximum(0, x)
    
    def ide(x):
        return x