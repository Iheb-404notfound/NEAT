from neatiheb.configuration import Configuration
import numpy as np
from neatiheb.operations import Operations

class Specie:
    
    def __init__(self, genomes, config= Configuration):
        self.config = config
        self.genomes = genomes
        self.fitness_history = []
        self.global_fitness = 0
    
    def breed(self):
        if np.random.uniform(0,1) < self.config.specie_procreation:
            # mutate an existing genome
            result = np.random.choice(self.genomes).clone()
            result.mutate()
            return result
        else:
            dad, papa = np.random.choice(self.genomes, size=2, replace=False)
            result = Operations.crossover(dad, papa)
            return result
    
    def natural_selection(self, fittest=-1):
        byfitness = sorted(self.genomes, key= lambda x: x.fitness, reverse=True)
        if fittest > 0 and fittest < len(byfitness):
            selected = byfitness[:fittest]
        else:
            selected = byfitness[:int(np.ceil(self.config.specie_procreation * len(byfitness)))]
        self.genomes = selected
    
    def fitness_sum(self):
        s = 0
        for genome in self.genomes:
            s += genome.fitness
        return s

    def update_fitness(self):
        for m in self.genomes:
            m.adj_fitness = m.fitness / len(self.genomes)
        

        self.fitness_history.append(self.fitness_sum())

        if len(self.fitness_history) > self.config.specie_max_age:
            self.fitness_history.pop(0)

        self.global_fitness = self.fitness_sum()

    def top(self):
        return max(self.genomes, key= lambda x: x.fitness)

    def stagnated(self):
        avg = sum(self.fitness_history) / len(self.fitness_history)
        return avg < self.fitness_history[0] and len(self.fitness_history) >= self.config.specie_max_age

    def reset(self):
        for g in self.genomes:
            g.fitness = 0
            g.adj_fitness = 0
            g.reset()