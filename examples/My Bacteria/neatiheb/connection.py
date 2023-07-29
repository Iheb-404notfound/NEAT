import numpy as np
from neatiheb.configuration import Configuration
from neatiheb.node import Node

class Connection:
    def __init__(self, input_node, output_node, weight=None, enabled=True, config = Configuration):
        self.config = config

        if(weight==None):
            weight = np.random.uniform(config.weight_range_start, config.weight_range_end)
        
        self.input_node = input_node
        self.output_node = output_node
        self.enabled = enabled
        self.weight = weight
        self.input_node.outcons.append(self)
    
    def mutate_weight_set(self):
        self.weight = np.random.uniform(self.config.weight_range_start, self.config.weight_range_end)
    
    def mutate_weight_perturb(self):
        self.weight += np.random.uniform(self.config.weight_range_start, self.config.weight_range_end)
    
    def get_innovation_number(self):
        # Cantor Pairing Function gives unique identifiers, pi(n1, n2) = n3, where n1,n2,n3 are integers
        return int(0.5*(self.input_node.id+self.output_node.id)*(self.input_node.id+self.output_node.id + 1) + self.output_node.id)

    def find_common(self, others):
        for i in range(len(others)):
            if others[i].get_innovation_number() == self.get_innovation_number():
                return i
        return -1

    def clone(self, input_node, output_node):
        return Connection(input_node, output_node, self.weight, self.enabled, self.config)

    def __str__(self):
        return f'innov={self.get_innovation_number():03d}  in={self.input_node.id:02d}{"i" if self.input_node.role==Node.SENSOR else "h"}  out={self.output_node.id:02d}{"o" if self.output_node.role==Node.OUTPUT  else "h"}'