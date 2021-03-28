#%%
import sys
import numpy as np
import copy
from io import StringIO
from typing import List
import gym.utils
import random
import pathmaker
try: 
    import cPickle as pickle 
except:
    import pickle 

class Env:
    """A variable Frozen Lake environment. It's the Frozen Lake from AI Gym with
    a varying starting position for the agent.
    
    Args:
        grid_shape (list-like): The matrix dimensions of the environment. 
        hole_pct (float): The probability of any open spot to be a hole.
            An "open spot" is any spot on the grid that is not an agent, 
            goal, or blocked.   
        n_goals (int): Defaults to 1.
    
    Attributes:
        interactables (dict): key-value pairs for the various items that can 
            take up space on the frozen lake. This would be the agent, goal, 
            holes, etc. The 'blocked' key refers to spaces that can't 
            be traversed.
        grid (np.ndarray): A matrix with the encodings for each interactable. 
    """
    def __init__(self, grid_shape = (10, 10), hole_pct = 0.2, n_goals = 3):
        self.interactables = {'frozen': '_', 'agent': 'A', 'goal': 'G', 
            'hole': 'o', 'blocked': 'b'} 

        # Set board dimensions and initalize to an "empty" grid. 
        if len(grid_shape) != 2:
            raise ValueError("'grid_shape' must be a list-like of length 2.")
        self.grid = np.full(grid_shape, self.interactables['frozen'])
        assert self.grid.shape == grid_shape
        # Initialize grid helper parameters  
        self._position_space: List[list] = self.position_space
        self.open_positions: List[list] = self._position_space
        self._agent_position: List[int] = self.agent_position 
        self.goal_position: List[int] = None

        # Initial grid - for env.reset()
        self.agent_start: List[int] = None
        self.env_start = None

        # Declare board paramteres as class attributes
        if (hole_pct < 0) or (hole_pct >= 1):
            raise ValueError("'hole_pct' must be between 0 and 1.") 
        self.hole_pct = hole_pct
        self.n_goals = n_goals

    # --------------------------------------------------------------------
    # Properties 
    # --------------------------------------------------------------------

    @property
    def position_space(self) -> List[List[int]]:
        try:
            return self.position_space
        except:
            row_dim, col_dim = self.grid.shape
            position_space: List[list] = []
            for i in range(row_dim):
                for j in range(col_dim):
                    position_space.append([i, j])
            return position_space
    
    @position_space.deleter
    def position_space(self):
        raise AttributeError("`position_space` attribute of class "
            + "`Env` is read-only.")

    @property
    def agent_position(self) -> List[int]:
        is_agent: np.ndarray = (self.grid == self.interactables['agent'])
        if np.any(is_agent):
            return np.argwhere(is_agent)[0].tolist() 
        else:
            return None

    # --------------------------------------------------------------------
    # Helper functions for creating an env from scratch
    # --------------------------------------------------------------------
    
    def randomly_select_open_position(self) -> List[int]:
        position: List[int] = random.choice(self.open_positions)
        return position

    def set_agent_goal(self):
        # positions_ag: The positions for the agent and goal(s)
        positions_ag: List[list] = []

        # Randomly select starting point for agent.
        agent_start = self.randomly_select_open_position()
        self.agent_start = agent_start
        self.open_positions.remove(agent_start) 
        positions_ag.append(agent_start)

        # Randomly select starting point for each goal.
        for _ in np.arange(self.n_goals):
            goal_position = self.randomly_select_open_position()
            self.open_positions.remove(goal_position)
            positions_ag.append(goal_position)
        self.goal_position = goal_position
        assert len(positions_ag) >= 2, "We expect at least 1 agent and 1 goal."
        
        # Label the agent on the grid.
        x, y = positions_ag[0]
        self.grid[x, y] = self.interactables['agent']

        # Label the goals on the grid.
        for goal_idx in np.arange(self.n_goals):
            x, y = positions_ag[goal_idx + 1]
            self.grid[x, y] = self.interactables['goal'] 

    def set_holes(self, hole_pct: float = None):
        """[summary]

        Args:
            hole_pct (float, optional): The probability that any open spot is a 
                hold. An "open spot" is any spot on the grid that is not an 
                agent, goal, or blocked. Defaults to 'env.hole_pct' attribute.
                See the first line of this method to understand the default 
                behavior.                 
        """
        hole_pct = self.hole_pct if (hole_pct == None) else hole_pct
        n_holes: int = int(len(self.open_positions) * self.hole_pct)
        for _ in range(n_holes):
            hole_position = self.randomly_select_open_position()
            self.open_positions.remove(hole_position)
            x, y = hole_position
            self.grid[x, y] = self.interactables['hole']

    # --------------------------------------------------------------------
    # Functions for the user
    # --------------------------------------------------------------------

    def create(self):
        """Place all of the interactables on the grid to create a new env."""
        self.set_agent_goal() # Create agent and goals
        
        # Clear a path for the agent
        valid_path = pathmaker.PathMaker(self).make_valid_path()
        for position in valid_path:
            if position in self.open_positions:
                self.open_positions.remove(position)
        
        # Place holes in some of the empty spaces
        self.set_holes()
        
        # Save initial state if this is the first time create() has been called.
        if self.env_start == None:
            self.env_start: Env = Env()
            self.env_start.grid = self.grid
            self.env_start.open_positions = self.open_positions
            self.env_start.env_start = self.env_start
            assert self.env_start != None

        # TODO: Check that there are holes on the grid.
        # TODO: Check that none of the positions in valid path now have holes.

    def reset(self):
        """Returns the initial env if it has been created. 

        Returns:
            Env: The initial environment.
        """        
        if self.env_start != None:
            return self.env_start

        elif self.env_start == None:
            self.create()
            assert self.env_start != None
        # TODO Test that this works as intended 
        

# Useful implementation links: 
# https://en.wikipedia.org/wiki/Depth-first_search
# https://docs.python.org/3/library/random.html

# def frozen_lake_original_map():
#     from gym.envs.toy_text import frozen_lake
#     print(frozen_lake.generate_random_map())