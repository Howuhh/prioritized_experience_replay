import os
import torch
import random
import numpy as np


def linear_schedule(max_value, total_steps):
    def inner(value):
        inner.calls += 1
        return value + (max_value - value) * inner.calls / total_steps
    inner.calls = 0
    return inner



def set_seed(env, seed=0):
    os.environ["PYTHONHASHSEED"] = str(seed)
    env.seed(seed)
    env.action_space.seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def device(force_cpu=True):
    return "cuda" if torch.cuda.is_available() and not force_cpu else "cpu"
