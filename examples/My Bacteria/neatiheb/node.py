import numpy as np
from neatiheb.configuration import Configuration

class Node:
    SENSOR = 0
    HIDDEN = 1
    OUTPUT = -1
    def __init__(self, id, activation, role, layer, bias=None, config = Configuration):
        self.config = config
        if(bias==None):
            bias = np.random.uniform(config.bias_range_start, config.bias_range_end)
        
        self.layer = layer
        self.id = id
        self.activation = activation
        self.bias = bias
        self.outcons = []
        self.role = role
        self.reset()
    
    def forward(self):
        if self.role != Node.SENSOR:
            self.output = self.activation(self.inputSum + self.bias)
        
        for con in self.outcons:
            con.output_node.inputSum += con.weight*self.output
    
    def mutate_bias_set(self):
        self.bias = np.random.uniform(self.config.bias_range_start, self.config.bias_range_end)
    
    def mutate_bias_perturb(self):
        self.bias += np.random.uniform(self.config.bias_range_start, self.config.bias_range_end)

    def reset(self):
        self.inputSum = 0
        self.output = 0

    def __str__(self):
        return f'id={self.id:02d}{["s", "h", "o"][self.role]} layer={self.layer:02d}'
    
    def clone(self):
        return Node(self.id, self.activation, self.role, self.layer, self.bias, self.config)

    def similarto(self, other):
        return self.id == other.id