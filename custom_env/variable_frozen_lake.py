#%%
import sys
import numpy as np
from io import StringIO

import gym.utils
import random

class Environment:
    
    env_objects = {'frozen': '0', 'agent': 'a', 'goal': 'g', 'hole': 'h', 
                   'blocked': 'b'}

    def __init__(self, board_dim = (50, 50), frozen_pct = 0.8, n_goals = 1):
        # Set board dimensions. 
        if len(board_dim) != 2:
            raise ValueError("'board_dim' must be a list-like of lenght 2.")
        self.board_dim = board_dim
        self.grid = np.zeros(board_dim, dtype='int').astype(str)
        assert self.grid.shape == self.board_dim

        if (frozen_pct < 0) or (frozen_pct >= 1):
            raise ValueError("'frozen_pct' must be between 0 and 1.") 
        self.frozen_pct = frozen_pct

        self.n_goals = n_goals
        self.env_matrix = np.empty(5) # TODO                        
        
    def set_agent_goal(self):
        grid_len, grid_height = self.board_dim
        def randomly_select_positions():
            positions = []
            # Randomly select starting point for agent.
            agent_x = random.randint(0, grid_len)
            agent_y = random.randint(0, grid_height)
            positions.append([agent_x, agent_y])
            # Randomly select starting point for each goal.
            for goal in self.n_goals:
                goal_x = random.randint(0, grid_len) 
                goal_y = random.randint(0, grid_height)
                positions.append([goal_x, goal_y])
            assert len(positions) >= 2
            return positions 
        
        agent_position, goal_position = randomly_select_positions()
        # Re- sample if the positions map. 
        while agent_position == goal_position:
            positions = randomly_select_positions()
        self.grid[positions[0]] = env_objects['agent']
        for goal_idx in self.n_goals:
            self.grid[positions[goal_idx + 1]] = env_objects['goal'] 



    def update(self, action):
        pass

def generate_random_grid( frozen_pct, num_goals):
    """Generates a random grid that has a path from start to goal.
    """
    pass
#%%
from gym.envs.toy_text import frozen_lake

print(frozen_lake.generate_random_map())
# %%
