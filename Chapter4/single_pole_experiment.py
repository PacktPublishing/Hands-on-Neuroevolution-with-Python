#
# This file provides the source code of the Single-Pole balancing experiment using on NEAT-Python library
#

# The Python standard library import
import os

# The NEAT-Python library imports
import neat
# The helper used to visualize experiment results
import visualize
# The cart-pole simulator
import cart_pole as cart

import utils

# The current working directory
local_dir = os.path.dirname(__file__)
# The directory to store outputs
out_dir = os.path.join(local_dir, 'out')
out_dir = os.path.join(out_dir, 'single_pole')

# The number of additional simulation runs for the winner genome
additional_num_runs = 100
# The number os steps in additional simulation runs
additional_steps = 200

def eval_genomes(genomes, config):
    """
    The function to evaluate the fitness of each genome in 
    the genomes list.
    Arguments:
        genomes: The list of genomes from population in the 
                current generation
        config: The configuration settings with algorithm
                hyper-parameters
    """
    for genome_id, genome in genomes:
        genome.fitness = 0.0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        fitness = cart.eval_fitness(net)
        if fitness >= config.fitness_threshold:
            # do additional steps of evaluation with random initial states
            # to make sure that we found stable control strategy rather than
            # special case for particular initial state
            success_runs = evaluate_best_net(net, config, additional_num_runs)
            # adjust fitness
            fitness = 1.0 - (additional_num_runs - success_runs) / \
                      additional_num_runs

        genome.fitness = fitness

def run_experiment(config_file, n_generations=100):
    """
    The function to run the experiment against hyper-parameters 
    defined in the provided configuration file.
    The winner genome will be rendered as a graph as well as the
    important statistics of neuroevolution process execution.
    Arguments:
        config_file: the path to the file with experiment 
                    configuration
    """
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
    p.add_reporter(neat.Checkpointer(5, filename_prefix='out/spb-neat-checkpoint-'))

    # Run for up to N generations.
    best_genome = p.run(eval_genomes, n=n_generations)

    # Display the best genome among generations.
    print('\nBest genome:\n{!s}'.format(best_genome))

    # Check if the best genome is a winning Single-Pole balancing controller 
    net = neat.nn.FeedForwardNetwork.create(best_genome, config)
    print("\n\nEvaluating the best genome in random runs")
    success_runs = evaluate_best_net(net, config, additional_num_runs)
    print("Runs successful/expected: %d/%d" % (success_runs, additional_num_runs))
    if success_runs == additional_num_runs:
        print("SUCCESS: The stable Single-Pole balancing controller found!!!")
    else:
        print("FAILURE: Failed to find the stable Single-Pole balancing controller!!!")

    # Visualize the experiment results
    node_names = {-1:'x', -2:'dot_x', -3:'θ', -4:'dot_θ', 0:'action'}
    visualize.draw_net(config, best_genome, True, node_names=node_names, directory=out_dir, fmt='svg')
    visualize.plot_stats(stats, ylog=False, view=True, filename=os.path.join(out_dir, 'avg_fitness.svg'))
    visualize.plot_species(stats, view=True, filename=os.path.join(out_dir, 'speciation.svg'))

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
        fitness = cart.eval_fitness(net, max_bal_steps=additional_steps)
        if fitness < config.fitness_threshold:
            return run
    return num_runs

if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    config_path = os.path.join(local_dir, 'single_pole_config.ini')

    # Clean results of previous run if any or init the ouput directory
    utils.clear_output(out_dir)

    # Run the experiment
    run_experiment(config_path)