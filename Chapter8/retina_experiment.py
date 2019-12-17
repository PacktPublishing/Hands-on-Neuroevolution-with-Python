#
# The script to run Modular Retinal experiment using ES-HyperNEAT method
# from the MultiNEAT library
#

# The Python standard library import
import os
import shutil
import math
import random
import time
import copy
import argparse
import pickle

import numpy as np

# The MultiNEAT specific
import MultiNEAT as NEAT

# The test environment
import retina_environment as rt_env

# The helper used to visualize experiment results
import utils
from utils import Statistics
import visualize

# The fitness threshold
MAX_FITNESS = 1000.0
FITNESS_THRESHOLD = MAX_FITNESS

def eval_individual(genome, substrate, rt_environment, params):
    """
    The funtciton to evaluate fitness of the individual CPPN genome by creating
    the substrate with topology based on the CPPN output.
    Arguments:
        genome:         The CPPN genome
        substrate:      The substrate to build control ANN
        rt_environment: The test visual retina environment
        params:         The ES-HyperNEAT hyper-parameters
    Returns:

    """
    #substrate.PrintInfo()
    # Create ANN from provided CPPN genome and substrate
    net = NEAT.NeuralNetwork()
    genome.BuildESHyperNEATPhenotype(net, substrate, params)

    fitness, dist, total_count, false_detections = rt_environment.evaluate_net(net, max_fitness=MAX_FITNESS)
    return fitness, dist, total_count, false_detections

def eval_genomes(genomes, substrate, rt_environment, params):
    """
    The function to evaluate fitness of the entire population against test 
    retina environment using the provided substrate 
    configuration of the descriminatorANN
    Arguments:
        genomes:        The list of genomes in the population
        substrate:      The substrate configuration of the descriminatorANN
        rt_environment: The test visual retina environment
        params:         The ES-HyperNEAT hyper-parameters
    Returns:
        the tuple (best_genome, max_fitness, errors) with best CPPN genome, 
        the maximal fitness score value and the list of error values for each genome
    """
    best_genome = None
    max_fitness = 0
    errors = []
    for genome in genomes:
        fitness, error, total_count, false_detections = eval_individual(genome, substrate, rt_environment, params)
        genome.SetFitness(fitness)
        errors.append(error)

        if fitness > max_fitness:
            max_fitness = fitness
            best_genome = genome
    
    return best_genome, max_fitness, errors

def run_experiment(params, rt_environment, trial_out_dir, n_generations=100, 
                    save_results=False, silent=False, args=None):
    """
    The function to run the experiment against hyper-parameters 
    defined in the provided configuration file.
    The winner genome will be rendered as a graph as well as the
    important statistics of neuroevolution process execution.
    Arguments:
        params:             The NEAT parameters
        rt_environment:     The test environment for detector ANN evaluations
        trial_out_dir:      The directory to store outputs for this trial
        n_generations:      The number of generations to execute.
        save_results:       The flag to control if intermdiate results will be saved.
        silent:             If True than no intermediary outputs will be
                            presented until solution is found.
        args:               The command line arguments holder.
    Returns:
        True if experiment finished with successful solver found. 
    """
    # random seed
    seed = 1569777981#int(time.time())

    # Create substrate
    substrate = create_substrate()

    # Create CPPN genome and population
    g = NEAT.Genome(0,
                    substrate.GetMinCPPNInputs(),
                    2, # hidden units
                    substrate.GetMinCPPNOutputs(),
                    False,
                    NEAT.ActivationFunction.TANH,
                    NEAT.ActivationFunction.SIGNED_GAUSS, # The initial activation type for hidden 
                    1, # hidden layers seed
                    params, 
                    1) # one hidden layer

    pop = NEAT.Population(g, params, True, 1.0, seed)
    pop.RNG.Seed(seed)

    # Run for up to N generations.
    start_time = time.time()
    best_genome_ser = None
    best_ever_goal_fitness = 0
    best_id = -1
    solution_found = False

    stats = Statistics()
    for generation in range(n_generations):
        print("\n****** Generation: %d ******\n" % generation)
        gen_time = time.time()
        # get list of current genomes
        genomes = NEAT.GetGenomeList(pop)

        # evaluate genomes
        genome, fitness, errors = eval_genomes(genomes, rt_environment=rt_environment, 
                                                substrate=substrate, params=params)

        stats.post_evaluate(max_fitness=fitness, errors=errors)
        solution_found = fitness >= FITNESS_THRESHOLD
        # store the best genome
        if solution_found or best_ever_goal_fitness < fitness:
            best_genome_ser = pickle.dumps(genome) # dump to pickle to freeze the genome state
            best_ever_goal_fitness = fitness
            best_id = genome.GetID()
        
        if solution_found:
            print('Solution found at generation: %d, best fitness: %f, species count: %d' % (generation, fitness, len(pop.Species)))
            break

        # advance to the next generation
        pop.Epoch()

        # print statistics
        gen_elapsed_time = time.time() - gen_time
        print("Best fitness: %f, genome ID: %d" % (fitness, best_id))
        print("Species count: %d" % len(pop.Species))
        print("Generation elapsed time: %.3f sec" % (gen_elapsed_time))
        print("Best fitness ever: %f, genome ID: %d" % (best_ever_goal_fitness, best_id))

    # Find the experiment elapsed time
    elapsed_time = time.time() - start_time

    # Restore the freezed best genome from pickle
    best_genome = pickle.loads(best_genome_ser)

    # write best genome to the file
    best_genome_file = os.path.join(trial_out_dir, "best_genome.pickle")
    with open(best_genome_file, 'wb') as genome_file:
        pickle.dump(best_genome, genome_file)

    # Print experiment statistics
    print("\nBest ever fitness: %f, genome ID: %d" % (best_ever_goal_fitness, best_id))
    print("\nTrial elapsed time: %.3f sec" % (elapsed_time))
    print("Random seed:", seed)

    # Visualize the experiment results
    show_results = not silent
    if save_results or show_results:
        # Draw CPPN network graph
        net = NEAT.NeuralNetwork()
        best_genome.BuildPhenotype(net)
        visualize.draw_net(net, view=False, node_names=None, filename="cppn_graph.svg", directory=trial_out_dir, fmt='svg')
        print("\nCPPN nodes: %d, connections: %d" % (len(net.neurons), len(net.connections)))

         # Draw the substrate network graph
        net = NEAT.NeuralNetwork()
        best_genome.BuildESHyperNEATPhenotype(net, substrate, params)
        visualize.draw_net(net, view=False, node_names=None, filename="substrate_graph.svg", directory=trial_out_dir, fmt='svg')
        print("\nSubstrate nodes: %d, connections: %d" % (len(net.neurons), len(net.connections)))
        inputs = net.NumInputs()
        outputs = net.NumOutputs()
        hidden = len(net.neurons) - net.NumInputs() - net.NumOutputs()
        print("\n\tinputs: %d, outputs: %d, hidden: %d" % (inputs, outputs, hidden))

        # Test against random retina configuration
        l_index = random.randint(0, 15)
        r_index = random.randint(0, 15)
        left = rt_environment.visual_objects[l_index]
        right = rt_environment.visual_objects[r_index]
        err, outputs = rt_environment._evaluate(net, left, right, 3)
        print("Test evaluation error: %f" % err)
        print("Left flag: %f, pattern: %s" % (outputs[0], left))
        print("Right flag: %f, pattern: %s" % (outputs[1], right))

        # Test against all visual objects
        fitness, avg_error, total_count, false_detections = rt_environment.evaluate_net(net, debug=True)
        print("Test evaluation against full data set [%d], fitness: %f, average error: %f, false detections: %f" % 
                (total_count, fitness, avg_error, false_detections))

        # Visualize statistics
        visualize.plot_stats(stats, ylog=False, view=show_results, filename=os.path.join(trial_out_dir, 'avg_fitness.svg'))


    return solution_found


def create_substrate():
    """
    The function to create appropriate substrate configuration with 16 inputs and 2 outputs.
    """
    # The input layer
    x_space = np.linspace(-1.0, 1.0, num=4)
    inputs = [
        (x_space[0], 0.0, 1.0), (x_space[1], 0.0, 1.0), (x_space[0], 0.0, -1.0), (x_space[1], 0.0, -1.0), # the left side
        (x_space[2], 0.0, 1.0), (x_space[3], 0.0, 1.0), (x_space[2], 0.0, -1.0), (x_space[3], 0.0, -1.0),  # the right side
        (0,0,0) # the bias
        ]
    # The output layer
    outputs = [(-1.0, 1.0, 0.0), (1.0, 1.0, 0.0)]

    substrate = NEAT.Substrate( inputs,
                                [], # hidden
                                outputs)

    # Allow connections: input-to-hidden, hidden-to-output, and hidden-to-hidden
    substrate.m_allow_input_hidden_links = True
    substrate.m_allow_hidden_output_links = True
    substrate.m_allow_hidden_hidden_links = True

    substrate.m_allow_input_output_links = False
    substrate.m_allow_output_hidden_links = False
    substrate.m_allow_output_output_links = False
    substrate.m_allow_looped_hidden_links = False
    substrate.m_allow_looped_output_links = False

    substrate.m_hidden_nodes_activation = NEAT.ActivationFunction.SIGNED_SIGMOID
    substrate.m_output_nodes_activation = NEAT.ActivationFunction.UNSIGNED_SIGMOID

    substrate.m_with_distance = True # send connection length to the CPPN as a parameter
    substrate.m_max_weight_and_bias = 8.0

    return substrate

def create_params():
    params = NEAT.Parameters()
    params.PopulationSize = 300

    params.DynamicCompatibility = True
    params.CompatTreshold = 3.0
    params.YoungAgeTreshold = 15
    params.SpeciesMaxStagnation = 100
    params.OldAgeTreshold = 35
    params.MinSpecies = 5
    params.MaxSpecies = 15
    params.RouletteWheelSelection = False

    params.MutateRemLinkProb = 0.02
    params.RecurrentProb = 0
    params.OverallMutationRate = 0.15
    params.MutateAddLinkProb = 0.03
    params.MutateAddNeuronProb = 0.03
    params.MutateWeightsProb = 0.90
    params.MaxWeight = 8.0
    params.WeightMutationMaxPower = 0.2
    params.WeightReplacementMaxPower = 1.0

    params.MutateActivationAProb = 0.0
    params.ActivationAMutationMaxPower = 0.5
    params.MinActivationA = 0.05
    params.MaxActivationA = 6.0

    params.MutateNeuronActivationTypeProb = 0.3
    params.ActivationFunction_SignedGauss_Prob = 1.0
    params.ActivationFunction_SignedStep_Prob = 1.0
    params.ActivationFunction_Linear_Prob = 1.0
    params.ActivationFunction_SignedSine_Prob = 1.0
    params.ActivationFunction_SignedSigmoid_Prob = 1.0

    params.ActivationFunction_Tanh_Prob = 0.0
    params.ActivationFunction_UnsignedSigmoid_Prob = 0.0
    params.ActivationFunction_TanhCubic_Prob = 0.0
    params.ActivationFunction_UnsignedStep_Prob = 0.0
    params.ActivationFunction_UnsignedGauss_Prob = 0.0
    params.ActivationFunction_Abs_Prob = 0.0
    params.ActivationFunction_UnsignedSine_Prob = 0.0

    params.MutateNeuronTraitsProb = 0
    params.MutateLinkTraitsProb = 0

    params.DivisionThreshold = 0.5
    params.VarianceThreshold = 0.03
    params.BandThreshold = 0.3
    params.InitialDepth = 2
    params.MaxDepth = 3
    params.IterationLevel = 1
    params.Leo = False
    params.GeometrySeed = False
    params.LeoSeed = False
    params.LeoThreshold = 0.3
    params.CPPN_Bias = 0.33#-1.0#
    params.Qtree_X = 0.0
    params.Qtree_Y = 0.0
    params.Width = 1.0
    params.Height = 1.0
    params.Elitism = 0.1

    return params

if __name__ == '__main__':
    # read command line parameters
    parser = argparse.ArgumentParser(description="The Modular Retina experiment runner (ES-HyperNEAT).")
    parser.add_argument('-g', '--generations', default=500, type=int, 
                        help='The number of generations for the evolutionary process.')
    parser.add_argument('-t', '--trials', type=int, default=1, help='The number of trials to run')
    parser.add_argument('-s', '--silent', type=bool, default=False, help='If True than no intermediary outputs will be presented until solution is found.')
    args = parser.parse_args()

    # The current working directory
    local_dir = os.path.dirname(__file__)
    # The directory to store outputs
    out_dir = os.path.join(local_dir, 'out')
    out_dir = os.path.join(out_dir, 'rt_multineat')

    # Clean results of previous run if any or init the ouput directory
    utils.clear_output(out_dir)

    # create the test environment
    env = rt_env.RetinaEnvironment()

    # Run the maze experiment trials
    print("Starting Modular Retina experiment with MultiNEAT, for %d trials" % (args.trials))
    for t in range(args.trials):
        print("\n\n----- Starting Trial: %d ------" % (t))
        trial_out_dir = os.path.join(out_dir, str(t))
        os.makedirs(trial_out_dir, exist_ok=True)
        soulution_found = run_experiment(params=create_params(),
                                        rt_environment = env,
                                        trial_out_dir=trial_out_dir,
                                        n_generations=args.generations,
                                        args=args,
                                        save_results=True,
                                        silent=args.silent)
        print("\n------ Trial %d complete, solution found: %s ------\n" % (t, soulution_found))