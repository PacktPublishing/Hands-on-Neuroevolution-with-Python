#
# This file provides source code of double-pole balancing experiment in
# its Markovian version, i.e., when velocity information is available to
# the solver.
#

# The Python standard library import
import os
import shutil
import math
# The NEAT-Python library imports
import neat
# The helper used to visualize experiment results
import visualize
# The cart-2-pole simulator
import cart_two_pole as cart

# The current working directory
local_dir = os.path.dirname(__file__)
# The directory to store outputs
out_dir = os.path.join(local_dir, 'out')


def eval_fitness(net, max_bal_steps=100000):
    """
    Evaluates fitness of the genome that was used to generate 
    provided net
    Arguments:
        net: The feed-forward neural network generated from genome
        max_bal_steps: The maximum nubmer of time steps to
            execute simulation.
    Returns:
        The phenotype fitness score in range [0, 1]
    """
    # First we run simulation loop returning number of successfull
    # simulation steps
    steps = cart.run_markov_simulation(net, max_bal_steps)

    if steps == max_bal_steps:
        # the maximal fitness
        return 1.0
    elif steps == 0: # needed to avoid math error when taking log(0)
        # the minimal fitness
        return 0.0
    else:
        # we use logarithmic scale because most cart-pole runs fails 
        # too early - within ~100 steps, but we are testing against 
        # 100'000 balancing steps
        log_steps = math.log(steps)
        log_max_steps = math.log(max_bal_steps)
        # The loss value is in range [0, 1]
        error = (log_max_steps - log_steps) / log_max_steps
        # The fitness value is a complement of the loss value
        return 1.0 - error

def eval_genomes(genomes, config):
    """
    The function to evaluate the fitness of each genome in 
    the genomes list. 
    The provided configuration is used to create feed-forward 
    neural network from each genome and after that created
    the neural network evaluated in its ability to solve
    XOR problem. As a result of this function execution, the
    the fitness score of each genome updated to the newly
    evaluated one.
    Arguments:
        genomes: The list of genomes from population in the 
                 current generation
        config:  The configuration settings with algorithm
                 hyper-parameters
    """
    for genome_id, genome in genomes:
        genome.fitness = 0.0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        fitness = eval_fitness(net)
        