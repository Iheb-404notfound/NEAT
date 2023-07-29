import numpy as np
from neat_iheb.node import Node
from neat_iheb.connection import Connection
from neat_iheb.configuration import Configuration

class Genome:
    def __init__(self, input_size, output_size, config = Configuration):
        self.input_size = input_size
        self.output_size = output_size
        self.config = config
        
        self.nodes = []
        self.connections = []

        self.fitness = 0
        self.adj_fitness = 0
    
    def generate(input_size, output_size, config = Configuration):
        # Generate minimum connections network (linking input nodes to output nodes without hidden layers)
        g = Genome(input_size, output_size, config)
        for i in range(g.input_size):
            g.nodes.append(Node(len(g.nodes), config.default_activation, Node.SENSOR, 0))
        
        for i in range(g.output_size):
            g.nodes.append(Node(len(g.nodes), config.default_activation, Node.OUTPUT, 1))
            for j in range(g.input_size):
                g.connections.append(Connection(g.nodes[j],g.nodes[-1]))
        return g
    
    def forward(self, inputs):
        self.reset()

        for i in range(len(inputs)):
            self.nodes[i].output = inputs[i]
        
        key = lambda x: x.layer
        sorted_nodes = sorted(self.nodes, key=key)
        
        result = []
        for node in sorted_nodes:
            node.forward()
            if node.role == Node.OUTPUT:
                result.append(node.output)
            
        return result

    def mutate(self):
        choices = [self.set_weight, self.set_bias, self.perturb_weight, self.perturb_bias, self.add_node, self.add_connection]
        probabilities = [self.config.mutate_weight_set, self.config.mutate_bias_set, self.config.mutate_weight_perturb, self.config.mutate_bias_perturb, self.config.mutate_add_node, self.config.mutate_add_connection]
        choice = np.random.choice(choices, p=probabilities)
        choice()


    def set_weight(self):
        enabled_cons = [con for con in self.connections if con.enabled]
        enabled_cons[np.random.randint(0, len(enabled_cons)-1)].mutate_weight_set()
    
    def perturb_weight(self):
        enabled_cons = [con for con in self.connections if con.enabled]
        enabled_cons[np.random.randint(0, len(enabled_cons)-1)].mutate_weight_perturb()
    
    def set_bias(self):
        self.nodes[np.random.randint(self.input_size, len(self.nodes)-1)].mutate_bias_set()
    
    def perturb_bias(self):
        self.nodes[np.random.randint(self.input_size, len(self.nodes)-1)].mutate_bias_perturb()
    
    def add_node(self):
        index = np.random.randint(0, len(self.connections))
        self.connections[index].enabled = False

        node = Node(len(self.nodes), self.config.default_activation, Node.HIDDEN, self.connections[index].input_node.layer+1, 0, config=self.config)

        for i in range(len(self.nodes)):
            if(self.nodes[i].layer > self.connections[index].input_node.layer):
                self.nodes[i].layer += 1

        self.nodes.append(node)
        self.connections.append(Connection(self.connections[index].input_node, node, 1, config=self.config))
        self.connections.append(Connection(node, self.connections[index].output_node, config=self.config))
    
    def add_connection(self):
        if not self.isFullyConnected():
            index1, index2  = np.random.randint(0, len(self.nodes)), np.random.randint(self.input_size, len(self.nodes))

            while self.nodes[index1].layer>=self.nodes[index2].layer or self.connected(index1, index2):
                index1, index2  = np.random.randint(0, len(self.nodes)), np.random.randint(self.input_size, len(self.nodes))
            self.connections.append(Connection(self.nodes[index1], self.nodes[index2], config=self.config))

    def connected(self, index1, index2):
        for con in self.connections:
            if con.input_node == self.nodes[index1] and con.output_node == self.nodes[index2]:
                return True
        return False
    
    def get_con(self, index1, index2):
        for con in self.connections:
            if con.input_node == self.nodes[index1] and con.output_node == self.nodes[index2]:
                return str(con)

    def isFullyConnected(self):
        maxcons = 0
        nodecount = []
        for node in self.nodes:
            if node.layer >= len(nodecount):
                nodecount += [0]*(node.layer - len(nodecount)+1)
            nodecount[node.layer]+=1
        
        for i in range(len(nodecount)-1):
            for j in range(i+1, len(nodecount)):
                maxcons += nodecount[i] * nodecount[j]
        
        return maxcons == len(self.connections)

    def reset(self):
        for node in self.nodes:
            node.reset()
    
    def correct_layers(self):
        max_layer = 0
        for con in self.connections:
            if con.input_node.layer >= con.output_node.layer:
                con.output_node.layer = con.input_node.layer + 1
        for node in self.nodes:
            if max_layer < node.layer and node.role!=Node.OUTPUT:
                max_layer = node.layer
        max_layer += 1
        for node in self.nodes:
            if node.role == Node.OUTPUT:
                node.layer = max_layer

    def clone(self):
        c = Genome(self.input_size, self.output_size, self.config)
        for node in self.nodes:
            c.nodes.append(node.clone())
        
        for con in self.connections:
            c.connections.append(con.clone(c.nodes[con.input_node.id], c.nodes[con.output_node.id]))
        return c

    def __str__(self):
        s = ''
        for con in self.connections:
            s += str(con) + '\n'
        return s

    def visualize(self):
        s = ''
        for con in self.connections:
            s+= f'{con.input_node.id} {con.output_node.id} {con.weight:.1f}\n'
        print(s,'\n')