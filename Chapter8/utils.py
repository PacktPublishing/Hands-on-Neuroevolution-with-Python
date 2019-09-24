#
# The collection of utilities
# 
import os
import shutil

import numpy as np

class Statistics:
    def __init__(self):
        self.most_fit_scores = []
        self.generation_statistics = []

    def post_evaluate(self, errors, max_fitness):
        self.generation_statistics.append(errors)
        self.most_fit_scores.append(max_fitness)

    def get_error_mean(self):
        avg_error = np.array([np.array(xi).mean() for xi in self.generation_statistics])
        return avg_error

    def get_error_stdev(self):
        stdev_error = np.array([np.array(xi).std() for xi in self.generation_statistics])
        return stdev_error

def clear_output(out_dir):
    """
    Function to clear output directory.
    Arguments:
        out_dir: The directory to be cleared
    """
    if os.path.isdir(out_dir):
        # remove files from previous run
        shutil.rmtree(out_dir)

    # create the output directory
    os.makedirs(out_dir, exist_ok=False)