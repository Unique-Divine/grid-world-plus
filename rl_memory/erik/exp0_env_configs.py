import numpy as np
from IPython.display import clear_output
import matplotlib.pyplot as plt
from typing import Iterable, List

config = {
    "grid_shape": (5, 5),
    "num_goals": 1,
    "hole_pct": .99,
    "view_length": 2
}

config0 = {
    "grid_shape": (5, 5),
    "num_goals": 1,
    "hole_pct": .2,
    "view_length": 2
}

def epsilon(current_episode, num_episodes):
    """
    epsilon decays as the current episode gets higher because we want the agent to
    explore more in earlier episodes (when it hasn't learned anything)
    explore less in later episodes (when it has learned something)
    i.e. assume that episode number is directly related to learning
    """
    # return 1 - (current_episode/num_episodes)
    return .5 * .9**current_episode