import os
import torch
import random
import numpy as np


def set_seed(env, seed=0):
    os.environ["PYTHONHASHSEED"] = str(seed)
    env.action_space.seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def device(force_cpu=True):
    return "cuda" if torch.cuda.is_available() and not force_cpu else "cpu"
