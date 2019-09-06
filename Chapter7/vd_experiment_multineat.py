#
# The script to run Visual Discrimination experiment using HyperNEAT method
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

import numpy as np

# The MultiNEAT specific
import MultiNEAT as NEAT
from MultiNEAT.viz import Draw

# The helper used to visualize experiment results
import utils

def run_experiment(params, trial_out_dir, args=None, n_generations=100, 
                    save_results=False, silent=False):
    """
    The function to run the experiment against hyper-parameters 
    defined in the provided configuration file.
    The winner genome will be rendered as a graph as well as the
    important statistics of neuroevolution process execution.
    Arguments:
        params:             The NEAT parameters
        trial_out_dir:      The directory to store outputs for this trial
        n_generations:      The number of generations to execute.
        save_results:       The flag to control if intermdiate results will be saved.
        silent:             If True than no intermediary outputs will be
                            presented until solution is found.
        args:               The command line arguments holder.
    Returns:
        True if experiment finished with successful solver found. 
    """

    substrate = create_substrate(11)
    substrate.PrintInfo()

    print(substrate.GetMinCPPNInputs(), substrate.GetMinCPPNOutputs(), " Max Dimensions:", len(substrate.m_output_coords[0]))

def create_substrate(dim):
    """
    The function to create two-sheets substrate configuration with specified
    dimensions of each sheet.
    Arguments:
        dim:    The dimensions accross X, Y axis of the sheet
    """
    # Building sheet configurations of inputs and outputs
    inputs = create_sheet_space(-1, 1, 11, -1)
    outputs = create_sheet_space(-1, 1, 11, 1)

    substrate = NEAT.Substrate( inputs,
                                [], # hidden
                                outputs)

    substrate.m_allow_input_output_links = True

    substrate.m_allow_input_hidden_links = False
    substrate.m_allow_hidden_hidden_links = False
    substrate.m_allow_hidden_output_links = False
    substrate.m_allow_output_hidden_links = False
    substrate.m_allow_output_output_links = False
    substrate.m_allow_looped_hidden_links = False
    substrate.m_allow_looped_output_links = False

    substrate.m_hidden_nodes_activation = NEAT.ActivationFunction.SIGNED_SIGMOID
    substrate.m_output_nodes_activation = NEAT.ActivationFunction.UNSIGNED_SIGMOID

    substrate.m_with_distance = True
    substrate.m_max_weight_and_bias = 3.0

    return substrate

def create_sheet_space(start, stop, dim, z):
    """
    The function to create list with coordinates for a specific sheet of the substrate.
    Arguments:
        start:  The start value by particular coordinate axis
        stop:   The stop value by particular coordinate axis
        dim:    The dimensions accross X, Y axis
        z:      The Z coordinatre of this sheet in the substrate.
    """
    lin_sp = np.linspace(start, stop, num=dim)
    space = []
    for x in range(dim):
        for y in range(dim):
            space.append((lin_sp[x], lin_sp[y], z))

    return space

def create_params():
    params = NEAT.Parameters()
    params.PopulationSize = 150

    params.DynamicCompatibility = True
    params.CompatTreshold = 2.0
    params.YoungAgeTreshold = 15
    params.SpeciesMaxStagnation = 100
    params.OldAgeTreshold = 35
    params.MinSpecies = 5
    params.MaxSpecies = 10
    params.RouletteWheelSelection = False

    params.MutateRemLinkProb = 0.02
    params.RecurrentProb = 0
    params.OverallMutationRate = 0.15
    params.MutateAddLinkProb = 0.1
    params.MutateAddNeuronProb = 0.03
    params.MutateWeightsProb = 0.90
    params.MaxWeight = 3.0
    params.WeightMutationMaxPower = 0.2
    params.WeightReplacementMaxPower = 1.0

    params.MutateActivationAProb = 0.0
    params.ActivationAMutationMaxPower = 0.5
    params.MinActivationA = 0.05
    params.MaxActivationA = 6.0

    params.MutateNeuronActivationTypeProb = 0.03

    params.ActivationFunction_SignedSigmoid_Prob = 0.0
    params.ActivationFunction_UnsignedSigmoid_Prob = 0.0
    params.ActivationFunction_Tanh_Prob = 1.0
    params.ActivationFunction_TanhCubic_Prob = 0.0
    params.ActivationFunction_SignedStep_Prob = 1.0
    params.ActivationFunction_UnsignedStep_Prob = 0.0
    params.ActivationFunction_SignedGauss_Prob = 1.0
    params.ActivationFunction_UnsignedGauss_Prob = 0.0
    params.ActivationFunction_Abs_Prob = 0.0
    params.ActivationFunction_SignedSine_Prob = 1.0
    params.ActivationFunction_UnsignedSine_Prob = 0.0
    params.ActivationFunction_Linear_Prob = 1.0

    params.MutateNeuronTraitsProb = 0
    params.MutateLinkTraitsProb = 0

    params.AllowLoops = False

    return params

if __name__ == '__main__':
    # read command line parameters
    parser = argparse.ArgumentParser(description="The Visual Discrimination experiment runner (Novelty Search).")
    parser.add_argument('-g', '--generations', default=500, type=int, 
                        help='The number of generations for the evolutionary process.')
    parser.add_argument('-t', '--trials', type=int, default=1, help='The number of trials to run')
    args = parser.parse_args()

    # The current working directory
    local_dir = os.path.dirname(__file__)
    # The directory to store outputs
    out_dir = os.path.join(local_dir, 'out')
    out_dir = os.path.join(out_dir, 'vd_multineat')

    # Clean results of previous run if any or init the ouput directory
    utils.clear_output(out_dir)

    # Run the maze experiment trials
    print("Starting Visual Discrimination experiment with MultiNEAT, for %d trials" % (args.trials))
    for t in range(args.trials):
        print("\n\n----- Starting Trial: %d ------" % (t))
        trial_out_dir = os.path.join(out_dir, str(t))
        os.makedirs(trial_out_dir, exist_ok=True)
        soulution_found = run_experiment( params=create_params(),
                                        trial_out_dir=trial_out_dir,
                                        n_generations=args.generations,
                                        args=args,
                                        save_results=True,
                                        silent=True)
    print("\n------ Trial %d complete, solution found: %s ------\n" % (t, soulution_found))