import os, sys
from typing import List

def access_root_dir(depth = 1):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    parent_dir = os.path.dirname(current_dir)
    args: list = [parent_dir]
    for _ in range(depth):
        args.append('..')
    
    rel_path = os.path.join(*args)
    sys.path.append(rel_path) 
    print(current_dir, parent_dir)

access_root_dir(depth = 0)

import rl_memory
from rl_memory.rlm_env import environment
from rl_memory.rl_algos import trackers

Env = environment.Env
Observation = environment.Observation
EnvStep = environment.EnvStep
EpisodeTracker = trackers.EpisodeTracker
SceneTracker = trackers.SceneTracker