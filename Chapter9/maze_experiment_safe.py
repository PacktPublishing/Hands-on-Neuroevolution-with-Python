#
# The script to run maze navigation experiment with Novelty Search optimization
# using the MultiNEAT library
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

# The MultiNEAT specific
import MultiNEAT as NEAT

# The helper used to visualize experiment results
import visualize
import utils
from utils import Statistics

# The maze environment
import maze_environment as maze
import agent
import novelty_archive as archive

# The number of maze solving simulator steps
SOLVER_TIME_STEPS = 400

class ANN:
    """
    The wrapper of MultiNEAT NeuralNetwork class
    """
    def __init__(self, multi_neat_nn, depth):
        """
        Creates new instance of the wrapper for a given NeuralNetwork
        """
        self.nn = multi_neat_nn
        self.depth = depth

    def activate(self, inputs):
        """
        Function to activate associated NeuralNetwork with given inputs
        Argumnets:
            inputs: the array with network inputs.
        Returns:
            The control signal outputs.
        """
        # append bias
        inputs.append(0.5)
        # activate and get outputs
        self.nn.Input(inputs)
        [self.nn.Activate() for _ in range(self.depth)]
        return self.nn.Output()

class Robot:
    """
    The maze nivigating robot
    """
    def __init__(self, maze_env, archive, genome, population):
        # The record store for evaluated maze solver agents
        self.record_store = agent.AgentRecordStore()
        # The initial maze simulation environment
        self.orig_maze_environment = maze_env
        # The NoveltyItem archive
        self.archive = archive
        # The initial genome
        self.genome = genome
        # The current population of robot genomes
        self.population = population

    def get_species_id(self, genome):
        genome_id = genome.GetID()
        for s in self.population.Species:
            for gen in s.Individuals:
                if gen.GetID() == genome_id:
                    return s.ID()
        return -1

class ObjectiveFun:
    """
    The coevolving objective function
    """
    def __init__(self, archive, genome, population):
        # The NoveltyItem archive
        self.archive = archive
        # The initial genome
        self.genome = genome
        # The current population of objective function genomes
        self.population = population

def evaluate_individual_solution(genome, generation, robot):
    """
    The function to evaluate individual solution against maze environment.
    Arguments:
        genome:         The genome to evaluate.
        generation:     The current generation.
        robot:          The object encapsulating the robots population
    Return:
        The tuple specifying if solution was found and the distance from maze exit of final robot position.
    """
    # create NoveltyItem for genome and store it into map
    genome_id = genome.GetID()
    n_item = archive.NoveltyItem(generation=generation, genomeId=genome_id)
    # run the simulation
    maze_env = copy.deepcopy(robot.orig_maze_environment)
    multi_net = NEAT.NeuralNetwork()
    genome.BuildPhenotype(multi_net)
    depth = 8
    try:
        genome.CalculateDepth()
        depth = genome.GetDepth()
    except:
        pass
    control_net = ANN(multi_net, depth=depth)
    distance = maze.maze_simulation_evaluate(
        env=maze_env, net=control_net, time_steps=SOLVER_TIME_STEPS, n_item=n_item)

    # Store simulation results into the agent record
    record = agent.AgenRecord(generation=generation, agent_id=genome_id)
    record.distance = distance
    record.x = maze_env.agent.location.x
    record.y = maze_env.agent.location.y
    record.hit_exit = maze_env.exit_found
    record.species_id = robot.get_species_id(genome)
    robot.record_store.add_record(record)

    return (maze_env.exit_found, distance, n_item)

def evaluate_solutions(robot, obj_func_coeffs, generation):
    best_robot_genome = None
    solution_found = False
    distances = []
    n_items_list = []
    # evaluate robot genomes against maze simulation
    robot_genomes = NEAT.GetGenomeList(robot.population)
    for genome in robot_genomes:
        found, distance, n_item = evaluate_individual_solution(
            genome=genome, generation=generation, robot=robot)
        # store returned values
        distances.append(distance)
        n_items_list.append(n_item)

        if found:
            best_robot_genome = genome
            solution_found = True

    # evaluate novelty scores of robot genomes and calculate fitness
    max_fitness = 0
    best_coeffs = None
    best_distance = 1000
    best_novelty = 0
    for i, n_item in enumerate(n_items_list):
        novelty = robot.archive.evaluate_novelty_score(item=n_item, n_items_list=n_items_list)
        # The sanity check
        assert robot_genomes[i].GetID() == n_item.genomeId

        # calculate fitness
        fitness, coeffs = evaluate_solution_fitness(distances[i], novelty, obj_func_coeffs)
        robot_genomes[i].SetFitness(fitness)

        if not solution_found:
            # find the best genome in population
            if max_fitness < fitness:
                max_fitness = fitness
                best_robot_genome = robot_genomes[i]
                best_coeffs = coeffs
                best_distance = distances[i]
                best_novelty = novelty
        elif best_robot_genome.GetID() == n_item.genomeId:
            # store fitness of winner solution
            max_fitness = fitness
            best_coeffs = coeffs
            best_distance = distances[i]
            best_novelty = novelty

    return best_robot_genome, solution_found, max_fitness, distances, best_coeffs, best_distance, best_novelty
        

def evaluate_solution_fitness(distance, novelty, obj_func_coeffs):
    """
    The function to evaluate fitness of solution. The solution fitness
    is based on the results of evaluation of the objective functions population.
    Arguments:
        distance:           The final distance to the goal (maze exit) of the maze solver
        novelty:            The novelty score of maze solver
        obj_func_coeffs:    The objective function coefficients from evaluated population of evolved objective
                            functions.
    Returns:
        The maximum fitness score eveluated using all provided objective function coefficients and objective function
        coefficients used to get max fitness score.
    """
    normalized_novelty = novelty
    if novelty >= 1.00:
        normalized_novelty = math.log(novelty)
    norm_distance = math.log(distance)

    max_fitness = 0
    best_coeffs = [-1, -1]
    for coeff in obj_func_coeffs:
        fitness = coeff[0] / norm_distance + coeff[1] * normalized_novelty
        if fitness > max_fitness:
            max_fitness = fitness
            best_coeffs[0] = coeff[0]
            best_coeffs[1] = coeff[1]

    # print("Solution fitness: %f -> d: %f, ns: %f, a: %f, b: %f" % (max_fitness, distance, novelty, best_coeffs[0], best_coeffs[1]))

    return max_fitness, best_coeffs

def evaluate_individ_obj_function(genome, generation):
    """
    The function to evaluate individual objective function
    Arguments:
        genome:     The objective function genome
        generation: The current generation of evolution
    Returns:
        The NoveltyItem created using evaluation results.
    """
    # create NoveltyItem for genome and store it into map
    genome_id = genome.GetID()
    n_item = archive.NoveltyItem(generation=generation, genomeId=genome_id)
    # run the simulation
    multi_net = NEAT.NeuralNetwork()
    genome.BuildPhenotype(multi_net)
    depth = 2
    try:
        genome.CalculateDepth()
        depth = genome.GetDepth()
    except:
        pass
    obj_net = ANN(multi_net, depth=depth)

    # set inputs and get ouputs ([a, b])
    output = obj_net.activate([0.5])

    # store coefficients
    n_item.data.append(output[0])
    n_item.data.append(output[1])

    return n_item

def evaluate_obj_functions(obj_function, generation):
    """
    The function to perform evaluation of the objective functions population
    Arguments:
        obj_function:   The population of objective functions
        generation:     The current generation of evolution
    """
    obj_func_coeffs = []
    n_items_list = []
    # evaluate objective function genomes and collect novelty items
    obj_func_genomes = NEAT.GetGenomeList(obj_function.population)
    for genome in obj_func_genomes:
        n_item = evaluate_individ_obj_function(genome=genome, generation=generation)
        n_items_list.append(n_item)
        obj_func_coeffs.append(n_item.data)

    # evaluate collected novelty items and set genomes fitness scores
    max_fitness = 0
    for i, genome in enumerate(obj_func_genomes):
        fitness = obj_function.archive.evaluate_novelty_score(item=n_items_list[i], n_items_list=n_items_list)
        genome.SetFitness(fitness)
        max_fitness = max(max_fitness, fitness)

    return obj_func_coeffs, max_fitness

def run_experiment(maze_env, trial_out_dir, args=None, n_generations=100, 
                    save_results=False, silent=False):
    """
    The function to run the experiment against hyper-parameters 
    defined in the provided configuration file.
    The winner genome will be rendered as a graph as well as the
    important statistics of neuroevolution process execution.
    Arguments:
        maze_env:           The maze environment to use in simulation.
        trial_out_dir:      The directory to store outputs for this trial
        n_generations:      The number of generations to execute.
        save_results:       The flag to control if intermdiate results will be saved.
        silent:             If True than no intermediary outputs will be
                            presented until solution is found.
        args:               The command line arguments holder.
    Returns:
        True if experiment finished with successful solver found. 
    """
    # set random seed
    seed = int(time.time())#1571021768#
    print("Random seed:           %d" % seed)

    # Create Population of Robots and objective functions
    robot = create_robot(maze_env, seed=seed)
    obj_func = create_objective_fun(seed)

    # Run for up to N generations.
    start_time = time.time()
    best_robot_genome_ser = None
    best_robot_id = -1
    solution_found = False
    best_obj_func_coeffs = None
    best_solution_novelty = 0
    best_solution_distance = 0

    stats = Statistics()
    for generation in range(n_generations):
        print("\n****** Generation: %d ******\n" % generation)
        gen_time = time.time()

        # evaluate objective function population
        obj_func_coeffs, max_obj_func_fitness = evaluate_obj_functions(obj_func, generation)

        # evaluate robots population
        robot_genome, solution_found, robot_fitness, distances, \
        obj_coeffs, best_distance, best_novelty = evaluate_solutions(
            robot=robot, obj_func_coeffs=obj_func_coeffs, generation=generation)

        stats.post_evaluate(max_fitness=robot_fitness, errors=distances)
        # store the best genome
        if solution_found or robot.population.GetBestFitnessEver() < robot_fitness:
            best_robot_genome_ser = pickle.dumps(robot_genome)
            best_robot_id = robot_genome.GetID()
            best_obj_func_coeffs = obj_coeffs
            best_solution_novelty = best_novelty
            best_solution_distance = best_distance
        
        if solution_found:
            print('\nSolution found at generation: %d, best fitness: %f, species count: %d\n' % 
                    (generation, robot_fitness, len(robot.population.Species)))
            break

        # advance to the next generation
        robot.population.Epoch()
        obj_func.population.Epoch()

        # print statistics
        gen_elapsed_time = time.time() - gen_time
        print("Generation fitness -> solution: %f, objective function: %f" % (robot_fitness, max_obj_func_fitness))
        print("Gen. species count -> solution: %d, objective function: %d" % (len(robot.population.Species), len(obj_func.population.Species)))
        print("Gen. archive size  -> solution: %d, objective function: %d" % (robot.archive.size(), obj_func.archive.size()))
        print("Objective function coeffts:     %s" % obj_coeffs)
        print("Gen. best solution genome ID:   %d, distance to exit: %f, novelty: %f" % (robot_genome.GetID(), best_distance, best_novelty))
        print("->")
        print("Best fitness ever  -> solution: %f, objective function: %f" % (robot.population.GetBestFitnessEver(), obj_func.population.GetBestFitnessEver()))
        print("Best ever solution genome ID:   %d, distance to exit: %f, novelty: %f" % (best_robot_id, best_solution_distance, best_solution_novelty))
        print("------------------------------")
        print("Generation elapsed time:        %.3f sec\n" % (gen_elapsed_time))
        
    elapsed_time = time.time() - start_time
    # Load serialized best robot genome
    best_robot_genome = pickle.loads(best_robot_genome_ser)

    # write best genome to the file
    best_genome_file = os.path.join(trial_out_dir, "best_robot_genome.pickle")
    with open(best_genome_file, 'wb') as genome_file:
        pickle.dump(best_robot_genome, genome_file)

    # write the record store data
    rs_file = os.path.join(trial_out_dir, "data.pickle")
    robot.record_store.dump(rs_file)

    print("==================================")
    print("Record store file:     %s" % rs_file)
    print("Random seed:           %d" % seed)
    print("............")
    print("Best solution fitness: %f, genome ID: %d" % (robot.population.GetBestFitnessEver(), best_robot_genome.GetID()))
    print("Best objective func coefficients: %s" % best_obj_func_coeffs)
    print("------------------------------")

    # Visualize the experiment results
    show_results = not silent
    if save_results or show_results:
        if args is None:
            visualize.draw_maze_records(maze_env, robot.record_store.records, view=show_results)
        else:
            visualize.draw_maze_records(maze_env, robot.record_store.records, 
                                        view=show_results, 
                                        width=args.width,
                                        height=args.height,
                                        filename=os.path.join(trial_out_dir, 'maze_records.svg'))
        # store NoveltyItems archive data
        robot.archive.write_to_file(path=os.path.join(trial_out_dir, 'ns_items_all.txt'))

        # create the best genome simulation path and render
        maze_env = copy.deepcopy(robot.orig_maze_environment)
        multi_net = NEAT.NeuralNetwork()
        best_robot_genome.BuildPhenotype(multi_net)
        depth = 8
        try:
            best_robot_genome.CalculateDepth()
            depth = genome.GetDepth()
        except:
            pass
        control_net = ANN(multi_net, depth=depth)
        path_points = []
        distance = maze.maze_simulation_evaluate(
                                    env=maze_env, 
                                    net=control_net, 
                                    time_steps=SOLVER_TIME_STEPS,
                                    path_points=path_points)
        print("Best solution distance to maze exit: %.2f, novelty: %.2f" % (distance, best_solution_novelty))
        visualize.draw_agent_path(robot.orig_maze_environment, path_points, best_robot_genome,
                                    view=show_results, 
                                    width=args.width,
                                    height=args.height,
                                    filename=os.path.join(trial_out_dir, 'best_solver_path.svg'))

        # Draw the best agent phenotype ANN
        visualize.draw_net(multi_net, view=show_results, filename="best_solver_net", directory=trial_out_dir)

        # Visualize statistics
        visualize.plot_stats(stats, ylog=False, view=show_results, filename=os.path.join(trial_out_dir, 'avg_fitness.svg'))

    print("------------------------")
    print("Trial elapsed time:    %.3f sec" % (elapsed_time))
    print("==================================")

    return solution_found

def create_objective_fun(seed):
    """
    The function to create population of objective functions
    """
    params = create_objective_fun_params()
    # Genome has one input (0.5) and two outputs (a and b)
    genome = NEAT.Genome(0, 1, 1, 2, False, 
        NEAT.ActivationFunction.TANH, # hidden layer activation
        NEAT.ActivationFunction.UNSIGNED_SIGMOID, # output layer activation
        1, params, 0)
    pop = NEAT.Population(genome, params, True, 1.0, seed)
    pop.RNG.Seed(seed)

    obj_archive = archive.NoveltyArchive(metric=maze.maze_novelty_metric_euclidean)
    obj_fun = ObjectiveFun(archive=obj_archive, genome=genome, population=pop)
    return obj_fun

def create_objective_fun_params():
    """
    The function to create NEAT hyper-parameters for population of objective functions
    """
    params = NEAT.Parameters()
    params.PopulationSize = 100
    params.DynamicCompatibility = True
    params.AllowClones = False
    params.AllowLoops = True
    params.CompatTreshold = 2.0
    params.CompatTresholdModifier = 0.3
    params.YoungAgeTreshold = 15
    params.SpeciesMaxStagnation = 20
    params.OldAgeTreshold = 200
    params.MinSpecies = 3
    params.MaxSpecies = 20
    params.RouletteWheelSelection = True

    params.RecurrentProb = 0.2
    params.OverallMutationRate = 0.4

    params.LinkTries = 40
    params.SpeciesDropoffAge = 100
    params.DisjointCoeff = 1.0
    params.ExcessCoeff = 1.0

    params.MutateWeightsProb = 0.90
    params.WeightMutationMaxPower = 0.8
    params.WeightReplacementMaxPower = 5.0
    params.MutateWeightsSevereProb = 0.5
    params.WeightMutationRate = 0.75

    params.MutateAddNeuronProb = 0.03
    params.MutateAddLinkProb = 0.05
    params.MutateRemLinkProb = 0.1

    params.Elitism = 0.2

    params.CrossoverRate = 0.8
    params.MultipointCrossoverRate = 0.6
    params.InterspeciesCrossoverRate = 0.01

    params.MutateNeuronTraitsProb = 0.1
    params.MutateLinkTraitsProb = 0.1

    return params

def create_robot(maze_env, seed):
    """
    The function to create population of robots.
    """
    params = create_robot_params()
    # Genome has 11 inputs and two outputs
    genome = NEAT.Genome(0, 11, 0, 2, False, NEAT.ActivationFunction.UNSIGNED_SIGMOID, 
                        NEAT.ActivationFunction.UNSIGNED_SIGMOID, 0, params, 0)
    pop = NEAT.Population(genome, params, True, 1.0, seed)
    pop.RNG.Seed(seed)

    robot_archive = archive.NoveltyArchive(metric=maze.maze_novelty_metric)
    robot = Robot(maze_env=maze_env, archive=robot_archive, genome=genome, population=pop)
    return robot

def create_robot_params():
    """
    The function to create NEAT hyper-parameters for population of robots
    """
    params = NEAT.Parameters()
    params.PopulationSize = 250
    params.DynamicCompatibility = True
    params.AllowClones = False
    params.AllowLoops = True
    params.CompatTreshold = 2.0
    params.CompatTresholdModifier = 0.3
    params.YoungAgeTreshold = 15
    params.SpeciesMaxStagnation = 20
    params.OldAgeTreshold = 200
    params.MinSpecies = 3
    params.MaxSpecies = 20
    params.RouletteWheelSelection = True

    params.RecurrentProb = 0.2
    params.OverallMutationRate = 0.4

    params.LinkTries = 40
    params.SpeciesDropoffAge = 200
    params.DisjointCoeff = 1.0
    params.ExcessCoeff = 1.0

    params.MutateWeightsProb = 0.90
    params.WeightMutationMaxPower = 0.8
    params.WeightReplacementMaxPower = 5.0
    params.MutateWeightsSevereProb = 0.5
    params.WeightMutationRate = 0.75
    params.MaxWeight = 30.0
    params.MinWeight = -30.0

    params.MutateAddNeuronProb = 0.03
    params.MutateAddLinkProb = 0.05
    params.MutateRemLinkProb = 0.1

    params.Elitism = 0.1

    params.CrossoverRate = 0.8
    params.MultipointCrossoverRate = 0.6
    params.InterspeciesCrossoverRate = 0.01

    params.MutateNeuronTraitsProb = 0.1
    params.MutateLinkTraitsProb = 0.1

    return params

if __name__ == '__main__':
    # read command line parameters
    parser = argparse.ArgumentParser(description="The maze experiment runner (SAFE).")
    parser.add_argument('-m', '--maze', default='medium', 
                        help='The maze configuration to use.')
    parser.add_argument('-g', '--generations', default=500, type=int, 
                        help='The number of generations for the evolutionary process.')
    parser.add_argument('-t', '--trials', type=int, default=1, help='The number of trials to run')
    parser.add_argument('--width', type=int, default=300, help='The width of the records subplot')
    parser.add_argument('--height', type=int, default=150, help='The height of the records subplot')
    args = parser.parse_args()

    if not (args.maze == 'medium' or args.maze == 'hard'):
        print('Unsupported maze configuration: %s' % args.maze)
        exit(1)

    # The current working directory
    local_dir = os.path.dirname(__file__)
    # The directory to store outputs
    out_dir = os.path.join(local_dir, 'out')
    out_dir = os.path.join(out_dir, 'maze_%s_safe' % args.maze)

    # Clean results of previous run if any or init the ouput directory
    utils.clear_output(out_dir)

    # Run the experiment
    maze_env_config = os.path.join(local_dir, '%s_maze.txt' % args.maze)
    maze_env = maze.read_environment(maze_env_config)

    # Run the maze experiment trials
    print("Starting the %s maze experiment (SAFE) with MultiNEAT, for %d trials" % (args.maze, args.trials))
    for t in range(args.trials):
        print("\n\n----- Starting Trial: %d ------" % (t))
        # Create novelty archive
        trial_out_dir = os.path.join(out_dir, str(t))
        os.makedirs(trial_out_dir, exist_ok=True)
        solution_found = run_experiment(
                                        maze_env=maze_env,
                                        trial_out_dir=trial_out_dir,
                                        n_generations=args.generations,
                                        args=args,
                                        save_results=True,
                                        silent=True)

        print("\n------ Trial %d complete, solution found: %s ------\n" % (t, solution_found))

        if solution_found:
            break