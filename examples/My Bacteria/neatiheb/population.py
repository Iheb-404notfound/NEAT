from neatiheb.configuration import Configuration
from neatiheb.specie import Specie
from neatiheb.operations import *
import numpy as np

class Population:
    def __init__(self, input_size, output_size, pop_size, config= Configuration):
        self.input_size = input_size
        self.output_size = output_size
        self.pop_size = pop_size
        self.config = config

        self.top = None

        self.species = []
        self.global_fitness = 0
        self.generation = 0
    
    def speciate(self, genome):
        if len(self.species) == 0:
            self.species.append(Specie([genome], self.config))
        else:
            for specie in self.species:
                element = np.random.choice(specie.genomes)
                if Operations.distance(element, genome, self.config) <= self.config.genomic_delta_threshold:
                    specie.genomes.append(genome)
                    return

            #Genome doesn't correspond to any specie
            self.species.append(Specie([genome], self.config))
    
    def generate(self):
        for i in range(self.pop_size):
            genome = Genome.generate(self.input_size, self.output_size, self.config)
            self.speciate(genome)
        self.top = self.species[0].genomes[0]
    
    def update_top(self):
        self.top = max(max([specie.top() for specie in self.species], key=lambda x: x.fitness), self.top, key=lambda x: x.fitness)

    def evolve(self):
        
        self.global_fitness = 0
        for specie in self.species:
            specie.update_fitness()
            self.global_fitness += specie.global_fitness
        
        if self.global_fitness == 0:
            for specie in self.species:
                for genome in specie.genomes:
                    genome.mutate()
        else:
            # find the survived species
            survivors = []
            for specie in self.species:
                if not specie.stagnated():
                    survivors.append(specie)
            self.species = survivors

            # apply natural selection
            for specie in self.species:
                specie.natural_selection()

            # repopulate, species with highest fitnesses will survive more
            for specie in self.species:
                r = specie.global_fitness / self.global_fitness
                rest = self.pop_size - sum([len(s.genomes) for s in self.species])
                add = int(np.ceil(r * rest))
                for i in range(add):
                    self.speciate(specie.breed())
            
            # no species survived
            if not self.species:
                if np.random.uniform(0, 1) < self.config.mutate_top:
                    self.speciate(self.top.clone().mutate())
                else:
                    self.speciate(Genome.generate(self.input_size, self.output_size, self.config))
            
        self.generation += 1
    
    def done(self):
        return self.top.fitness >= self.config.max_fitness or self.generation >= self.config.max_generations

    def evaluate(self, evaluation_function):
        self.generate()
        while not self.done():
            print(f'########################{self.generation+1}#######################')
            genomes = []
            for s in self.species:
                genomes += s.genomes
            evaluation_function(genomes)
            self.update_top()
            self.evolve()
            self.reset()
        self.update_top()
        return self.top
    
    def reset(self):
        for s in self.species:
            s.reset()