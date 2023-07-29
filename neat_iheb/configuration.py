from neat_iheb.activations import *

class Configuration:
    
    bias_range_start = -1
    bias_range_end = 1
    weight_range_start = -1
    weight_range_end = 1

    default_activation = Activations.tanh

    #mutation probabilities
    mutate_weight_set = 0.1
    mutate_bias_set = 0.1
    mutate_weight_perturb = 0.4
    mutate_bias_perturb = 0.3
    mutate_add_node = 0.02
    mutate_add_connection = 0.08

    #genomic distance coefficients
    bias_distance_coefficient = 0.03
    weight_distance_coefficient = 0.03
    excess_disjoint_distance_coefficient = 0.04
    genomic_delta_threshold = 0.5

    #specie constants
    specie_procreation = 0.4
    extinction_of_genomes = 0.25
    specie_max_age = 30

    #population constants
    mutate_top = 0.4
    max_fitness = 300
    max_generations = 20