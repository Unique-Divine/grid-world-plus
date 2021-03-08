#%%
import sys
import numpy as np
from io import StringIO

import gym.utils
import random

class Environment:

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

        self.objects = {'frozen': '0', 'agent': 'a', 'goal': 'g', 'hole': 'h', 
                        'blocked': 'b'}        
        self.n_goals = n_goals
        self.env_matrix = np.empty(5) # TODO                        
        
    def set_agent_goal(self):
        grid_len, grid_height = self.board_dim
        def randomly_select_positions():
            positions = []
            # Randomly select starting point for agent.
            agent_x, agent_y = [random.randrange(l) for l in self.board_dim]
            positions.append([agent_x, agent_y])
            # Randomly select starting point for each goal.
            for goal_idx in np.arange(self.n_goals):
                goal_x, goal_y = [random.randrange(l) for l in self.board_dim] 
                positions.append([goal_x, goal_y])
            assert len(positions) >= 2
            return positions 
        
        positions = randomly_select_positions()

        for position in positions:
            assert len(position) == 2

        # Re- sample if the positions map. 
        while np.all(positions):
            positions = randomly_select_positions()

        x, y = positions[0]
        self.grid[x, y] = self.objects['agent']
        for goal_idx in np.arange(self.n_goals):
            x, y = positions[goal_idx + 1]
            self.grid[x, y] = self.objects['goal'] 

    def update(self, action):
        pass

def generate_random_grid( frozen_pct, num_goals):
    """Generates a random grid that has a path from start to goal.
    """
    pass


def test_set_agent_goal():
    env = Environment()
    env.set_agent_goal()

    # Find objects such as the agent and goal
    nonfrozen_spots: np.ndarray = np.argwhere(env.grid != env.objects['frozen'])
    assert nonfrozen_spots.ndim == 2
    assert nonfrozen_spots.size >= 4
    assert nonfrozen_spots.size % 2 == 0
    nonfrozen_spots = list(nonfrozen_spots)

    env_objects: list  = list(env.objects.keys())
    assert 'frozen' in env_objects
    env_objects.remove('frozen')
    
    nonfrozen: list = env_objects
    nonfrozen: dict = {k: v for k, v in env.objects.items() if k in nonfrozen}

    # Verify that each nonfrozen spot is actually not frozen.
    for spot in nonfrozen_spots:
        assert spot.size == 2
        x, y = spot
        env_object = env.grid[x, y]
        assert env_object in list(nonfrozen.values())
    
test_set_agent_goal()
#%%
from gym.envs.toy_text import frozen_lake

print(frozen_lake.generate_random_map())
# %%
