import os
import sys

# set avaliable GPUs to 1 in environment variable
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from ann_automl.lm.lm_funcs import run


if __name__ == "__main__":
    # takes 1 argument: the request (as a string)
    run(sys.argv[1])
