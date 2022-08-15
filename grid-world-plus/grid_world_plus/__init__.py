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

from grid_world_plus import env
from grid_world_plus import replay

Env = env.Env
Observation = env.Observation
EnvStep = env.EnvStep
Memory = replay.Memory
Experience = replay.Experience