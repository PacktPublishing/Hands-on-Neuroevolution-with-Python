#
# The collection of utilities
# 
import os
import shutil

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