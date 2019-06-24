#
# The script to run maze navigation experiment for both medium and hard
# maze configurations.
#

# The Python standard library import
import os
import shutil
import math
import random
import time
import copy

# The NEAT-Python library imports
import neat
# The helper used to visualize experiment results
import visualize

# The maze environment
import maze_environment as maze
import agent

# The current working directory
local_dir = os.path.dirname(__file__)
# The directory to store outputs
out_dir = os.path.join(local_dir, 'out')
out_dir = os.path.join(out_dir, 'maze_objective')

class MazeSimulation:
    """
    The class to hold maze simulator execution parameters and results.
    """
    def __init__(self, maze_env):
        """
        Creates new instance and initialize fileds.
        Arguments:
            maze_env: The maze environment as loaded from configuration file.
        """
        # The initial maze simulation environment
        self.orig_maze_environment = maze_env
        # The record store for evaluated maze solver agents
        self.record_store = agent.RecordStore()
        # The couter to assign unique numbers to evaluated individuals
        self.individ_counter = 0
        # The NEAT population object
        self.population = None

# The simulation results holder for a one trial.
# It must be initialized before start of each trial.
trialSim = None

def eval_fitness(genome, config, time_steps=400):
    """
    Evaluates fitness of the provided genome.
    Arguments:
        genome:     The genome to evaluate.
        config:     The NEAT configuration holder.
        time_steps: The number of time steps to execute for maze solver simulation.
    Returns:
        The phenotype fitness score in range [0, 1]
    """
    record = agent.AgenRecord(
        generation=trialSim.population.generation,
        agent_id=trialSim.individ_counter)
    # increment the unique individuals counter
    trialSim.individ_counter += 1
