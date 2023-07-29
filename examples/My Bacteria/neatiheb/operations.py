from neatiheb.genome import Genome
from neatiheb.configuration import Configuration
import numpy as np
from neatiheb.node import Node
from neatiheb.connection import Connection

class Operations:
    def distance(parent1: Genome, parent2: Genome, config=Configuration):
        inter = []
        ED = 0
        for i in range(len(parent1.connections)):
            j = parent1.connections[i].find_common(parent2.connections)
            if j>=0:
                inter.append((i,j))
            else:
                ED += 1
        for i in range(len(parent2.connections)):
            if parent2.connections[i].find_common(parent1.connections)<0:
                ED += 1

        nodes = min(len(parent1.nodes), len(parent2.nodes))
        cons = len(max(parent1.connections, parent2.connections, key=len))

        W, B= 0, 0
        for (i,j) in inter:
            W += abs(parent1.connections[i].weight - parent2.connections[j].weight)

        for i in range(nodes):
            B += abs(parent1.nodes[i].bias - parent1.nodes[i].bias)

        c1 = config.excess_disjoint_distance_coefficient/cons
        c2 = config.bias_distance_coefficient/nodes
        c3 = config.weight_distance_coefficient/len(inter)
        return c1*ED + c2*B + c3*W

    def crossover(parent1: Genome, parent2: Genome, config=Configuration):
        offspring = Genome(parent1.input_size, parent1.output_size, config)

        #copy mutual nodes
        num_nodes = min(len(parent1.nodes), len(parent2.nodes))
        for i in range(num_nodes):
            parent = np.random.choice([parent1, parent2])
            offspring.nodes.append(parent.nodes[i].clone())

        #copy excess disjoint nodes
        parent = max(parent1, parent2, key= lambda p: len(p.nodes))
        excess = []
        for i in range(num_nodes, len(parent.nodes)):
            excess += [parent.nodes[i].clone()]
        offspring.nodes += excess

        #copy mutual connections from random parent
        for con in parent1.connections:
            i = con.find_common(parent2.connections)
            if i>=0:
                #offspring.connections.append(con.clone(offspring.nodes[con.input_node.id], offspring.nodes[con.output_node.id]))
                offspring.connections.append(Connection(offspring.nodes[con.input_node.id], offspring.nodes[con.output_node.id], 
                                                        np.random.choice([con.weight, parent2.connections[i].weight]), 
                                                        np.random.choice([con.enabled, parent2.connections[i].enabled]), config))

        #inherit the excess genes from the fittest parent
        fitter, fittless = max(parent1, parent2, key=lambda p: p.fitness), min(parent1, parent2, key=lambda p: p.fitness)
        for con in fitter.connections:
            i = con.find_common(fittless.connections)
            if i<0:
                offspring.connections.append(con.clone(offspring.nodes[con.input_node.id], offspring.nodes[con.output_node.id]))

        offspring.correct_layers()
        offspring.reset()
        return offspring