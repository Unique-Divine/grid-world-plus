import os, sys

def access_root_dir(depth = 1):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    parent_dir = os.path.dirname(current_dir)
    args: list = [parent_dir]
    for _ in range(depth):
        args.append('..')
    
    rel_path = os.path.join(*args)
    sys.path.append(rel_path) 
    print(current_dir, parent_dir)

access_root_dir()

from rl_memory.custom_env.agents import Agent
from rl_memory.custom_env.environment import (
    Point, Env, PathMaker, Observation, State)

__all__ = [
    'Point',
    'Env',
    'PathMaker',
    'Observation',
    'State',
    'Agent',
]