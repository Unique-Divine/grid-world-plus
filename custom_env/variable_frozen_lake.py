#%%
import sys
import numpy as np
import pandas as pd
from io import StringIO
from typing import List
import gym.utils
import random
import itertools

class Environment:
    """A variable Frozen Lake environment. It's the Frozen Lake from AI Gym with
    a varying starting position for the agent.
    Args:
        grid_shape (list-like): 
        difficulty (float): The probability of any open spot, i.e. one that 
            isn't an agent, goal, or blocked, to be hole.  
        n_goals (int): Defaults to 1.
    
    Attributes:
        interactables (dict): key-value pairs for the various items that can 
            take up space on the frozen lake. This would be the agent, goal, 
            holes, etc. The 'blocked' key refers to spaces that can't 
            be traversed.
        grid (np.ndarray): A matrix with the encodings for each interactable. 
    """
    def __init__(self, grid_shape = (10, 10), difficulty = 0.2, n_goals = 1):
        self.interactables = {'frozen': '0', 'agent': 'a', 'goal': 'g', 
            'hole': 'h', 'blocked': 'b'} 

        # Set board dimensions and initalize to an "empty" grid. 
        if len(grid_shape) != 2:
            raise ValueError("'grid_shape' must be a list-like of lenght 2.")
        self.grid = np.zeros(grid_shape, dtype='int').astype(str)
        assert self.grid.shape == grid_shape

        # TODO: Implement blocked pathway
        if (difficulty < 0) or (difficulty >= 1):
            raise ValueError("'difficulty' must be between 0 and 1.") 
        self.difficulty = difficulty
       
        self.n_goals = n_goals
        
        self.position_space: List[list] = self.get_position_space()
        self.open_positions: List[list] = self.position_space

    def get_position_space(self) -> set:
        row_dim, col_dim = self.grid.shape
        position_space: List[list] = []
        for i in range(row_dim):
            for j in range(col_dim):
                position_space.append([i, j])
        return position_space

    def randomly_select_open_position(self) -> List[int]:
        position: List[int] = random.choice(self.open_positions)
        return position

    def set_agent_goal(self):
        # positions_ag: The positions for the agent and goal(s)
        positions_ag: List[list] = []

        # Randomly select starting point for agent.
        agent_position = self.randomly_select_open_position()
        self.open_positions.remove(agent_position) 
        positions_ag.append(agent_position)

        # Randomly select starting point for each goal.
        for goal in np.arange(self.n_goals):
            goal_position = self.randomly_select_open_position()
            self.open_positions.remove(goal_position)
            positions_ag.append(goal_position)
        assert len(positions_ag) >= 2, "We expect at least 1 agent and 1 goal."
        
        # Label the agent on the grid.
        x, y = positions_ag[0]
        self.grid[x, y] = self.interactables['agent']

        # Label the goals on the grid.
        for goal_idx in np.arange(self.n_goals):
            x, y = positions_ag[goal_idx + 1]
            self.grid[x, y] = self.interactables['goal'] 

    def (self, action):
        act
        pass

    def generate_valid_path(self):
        """Generates a random grid that has a path from start to goal.
        """
        # TODO Use maze-generation algo to verify that a valid path exists.
        
        self.grid

        pass


def test_set_agent_goal():
    env = Environment(grid_shape=(3, 3), n_goals=6)
    env.set_agent_goal()

    # Find interactables such as the agent and goal
    nonfrozen_spots: np.ndarray = np.argwhere(
        env.grid != env.interactables['frozen'])
    assert nonfrozen_spots.ndim == 2
    assert nonfrozen_spots.size >= 4
    assert nonfrozen_spots.size % 2 == 0
    nonfrozen_spots = list(nonfrozen_spots)

    # Subset the dictionary of possible environment interactables
    env_interactables: list  = list(env.interactables.keys())
    assert 'frozen' in env_interactables
    env_interactables.remove('frozen')    
    nonfrozen: list = env_interactables
    nonfrozen: dict = {
        k: v for k, v in env.interactables.items() if k in nonfrozen}

    # Verify that each nonfrozen spot is actually not frozen.
    for spot in nonfrozen_spots:
        assert spot.size == 2
        x, y = spot
        env_object = env.grid[x, y]
        assert env_object in list(nonfrozen.values())
    
    print(env.grid)
    print(env.open_positions)
    
test_set_agent_goal()
print("Test passed.")
#%%
from gym.envs.toy_text import frozen_lake

print(frozen_lake.generate_random_map())
# %%
