#
# This file provides source code of double-pole balancing experiment in
# its Markovian version, i.e., when velocity information is available to
# the solver.
#

# The Python standard library import
import os
import shutil
import math
import random
import time
# The NEAT-Python library imports
import neat
# The helper used to visualize experiment results
import visualize
# The cart-2-pole simulator
import cart_two_pole as cart

import utils

# The current working directory
local_dir = os.path.dirname(__file__)
# The directory to store outputs
out_dir = os.path.join(local_dir, 'out')
out_dir = os.path.join(out_dir, 'two_pole_markov')

# The number of additional simulation runs for the winner genome
additional_num_runs = 1
# The number os steps in additional simulation runs
additional_steps = 100000

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
    Arguments:
        genomes: The list of genomes from population in the 
                 current generation
        config:  The configuration settings with algorithm
                 hyper-parameters
    """
    for _, genome in genomes:
        genome.fitness = 0.0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        genome.fitness = eval_fitness(net)
        
def run_experiment(config_file, n_generations=100, silent=False):
    """
    The function to run the experiment against hyper-parameters 
    defined in the provided configuration file.
    The winner genome will be rendered as a graph as well as the
    important statistics of neuroevolution process execution.
    Arguments:
        config_file: the path to the file with experiment 
                    configuration
    Returns:
        True if experiment finished with successful solver found. 
    """
    # set random seed
    seed = 1559231616#int(time.time())#
    random.seed(seed)

    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5, filename_prefix='out/tpbm-neat-checkpoint-'))

    # Run for up to N generations.
    best_genome = p.run(eval_genomes, n=n_generations)

    # Display the best genome among generations.
    print('\nBest genome:\n{!s}'.format(best_genome))

    # Check if the best genome is a winning Double-Pole-Markov balancing controller 
    net = neat.nn.FeedForwardNetwork.create(best_genome, config)
    print("\n\nEvaluating the best genome in random runs")
    success_runs = evaluate_best_net(net, config, additional_num_runs)
    print("Runs successful/expected: %d/%d" % (success_runs, additional_num_runs))
    if success_runs == additional_num_runs:
        print("SUCCESS: The stable Double-Pole-Markov balancing controller found!!!")
    else:
        print("FAILURE: Failed to find the stable Double-Pole-Markov balancing controller!!!")

    print("Random seed:", seed)

    # Visualize the experiment results
    if not silent or success_runs == additional_num_runs:
        node_names = {-1:'x', -2:'dot_x', -3:'θ_1', -4:'dot_θ_1', -5:'θ_2', -6:'dot_θ_2', 0:'action'}
        visualize.draw_net(config, best_genome, True, node_names=node_names, directory=out_dir, fmt='svg')
        visualize.plot_stats(stats, ylog=False, view=True, filename=os.path.join(out_dir, 'avg_fitness.svg'))
        visualize.plot_species(stats, view=True, filename=os.path.join(out_dir, 'speciation.svg'))
    
    return success_runs == additional_num_runs

def evaluate_best_net(net, config, num_runs):
    """
    The function to evaluate the ANN of the best genome in
    specified number of sequetial runs. It is aimed to test it
    against various random initial states that checks if it is
    implementing stable control strategy or just a special case
    for particular initial state.
    Arguments:
        net:        The ANN to evaluate
        config:     The hyper-parameters configuration
        num_runs:   The number of sequential runs
    Returns:
        The number of succesful runs 
    """
    for run in range(num_runs):
        fitness = eval_fitness(net, max_bal_steps=additional_steps)
        if fitness < config.fitness_threshold:
            return run
    return num_runs

if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    config_path = os.path.join(local_dir, 'two_pole_markov_config.ini')

    # Clean results of previous run if any or init the ouput directory
    utils.clear_output(out_dir)

    # Run the experiment
    pole_length = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    num_runs = len(pole_length)
    for i in range(num_runs):
        cart.LENGTH_2 = pole_length[i] / 2.0
        solved = run_experiment(config_path, n_generations=100, silent=True)
        print("run: %d, solved: %s, half-length: %f" % (i + 1, solved, cart.LENGTH_2))
        if solved:
            print("Solution found in: %d run, short pole length: %f" % (i + 1, pole_length[i]))
            break
        
