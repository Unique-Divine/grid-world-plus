#%%
import os, sys
try:
    import rl_memory
except:
    exec(open('__init__.py').read()) 
    import rl_memory
import numpy as np
import torch
import random
import copy
import warnings; warnings.filterwarnings("ignore")
from rl_memory import rlm_env
import rl_memory.rlm_env.environment

# Type imports
from typing import List, Tuple
from torch import Tensor
Array = np.ndarray
Env = rlm_env.Env
PathMaker = rl_memory.rlm_env.environment.PathMaker

def init_env() -> Tuple[Env, PathMaker]:
    """Helper function for setting up a random environments for tests. 
    Returns:
        env: An empty environment without any agents, goals, or holes. 
        pm: An instance of the Pathmaker class.  
    """
    env: Env = rlm_env.Env(grid_shape=(15,15), n_goals=4, 
                                  hole_pct = 0.3)
    pm: PathMaker = rl_memory.rlm_env.environment.PathMaker(env)
    return env, pm

class TestEnvInit:
    def test_set_agent_goal(self):
        env = rlm_env.Env(grid_shape=(50, 50), n_goals=60)
        for item_name in ['agent', 'hole', 'goal', 'blocked']:
            n_items = int((env.grid == env.interactables[item_name]).sum())
            expected_count = 0
            assert n_items == expected_count, \
                f"Too many '{item_name}' items on the board."

        env.set_agent_goal()
        # Find interactables such as the agent and goal
        nonfrozen_spots: Array = np.argwhere(
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
        
    def test_set_holes(self):
        env = rlm_env.Env(grid_shape=(50, 50), n_goals=20)

        env.set_agent_goal()
        n_holes: int = int((env.grid == env.interactables['hole']).sum())
        assert n_holes == 0, "Nonzero number of holes before creation." 

        # Calculate expected number of holes before they are placed on the board.
        n_agents = int((env.grid == env.interactables['agent']).sum())
        n_goals = int((env.grid == env.interactables['goal']).sum())
        n_frozen = int((env.grid == env.interactables['frozen']).sum())
        n_positions = np.product(env.grid.shape) 
        assert n_positions == np.sum([n_agents, n_goals, n_holes, n_frozen])
        n_previously_frozen = n_frozen
        expected_n_holes = int(n_previously_frozen * env.hole_pct)

        # Place holes on the board
        env.set_holes() 
        n_holes: int = int((env.grid == env.interactables['hole']).sum())
        assert n_holes != 0, "No holes were created by 'set_holes()'. Odd."
        n_frozen = int((env.grid == env.interactables['frozen']).sum())

        assert n_holes == expected_n_holes, \
            "Mistakes were made in 'env.set_holes()'"

class TestPathMaker:
    """ """
    def test_generate_shifted_spots(self):
        env, pm = init_env()
        spot: List[int] = random.choice(env.position_space)
        for shifted_spot in pm.generate_shifted_spots(spot):
            # Verify that 'shifted_spot' is on the grid.
            assert shifted_spot in env.position_space, (
                "Invalid move in random walk")
            
            # Verify that 'shifted_spot' is only 1 space away from 'spot'.
            positions = np.vstack([spot, shifted_spot]) 
            abs_displacement = np.abs(positions[0] - positions[1])
            assert np.all(abs_displacement <= 1), (
                "'shifted spot' is too far away.")

    def test_random_walk(self):
        env, pm = init_env() # ignore type: Env, PathMaker 
        spot: List[int] = random.choice(env.position_space)
        n_steps =  env.grid.shape[0] // 2
        random_path = pm.random_walk(n_steps = n_steps, start = spot)
        assert len(random_path) == n_steps + 1, "'path' is too short."

    def test_shortest_path(self):
        env, pm = init_env()
        env.set_agent_goal()
        path = pm.shortest_path(env.agent_position, env.goal_position)
        assert path[0] == env.agent_position
        assert path[-1] == env.goal_position

    def test_make_valid_path(self):
        env, pm = init_env()
        env.set_agent_goal()
        valid_path = pm.make_valid_path()
        assert valid_path[0] == env.agent_position
        assert valid_path[-1] == env.goal_position

class TestStateObservation:
    """Unit tests for initialization of Observation and State instances."""
    def test_obs_init(self):
        env, pm = init_env()
        env.reset()
        obs = rlm_env.Observation(env = env)
        assert isinstance(obs, torch.Tensor)
        # TODO 

    def test_state_start(self):
        pass # TODO
    
    def test_state_higher_k(self):
        pass # TODO

class TestEnvIntegration:
    """Tests for the environment updates. """
    def test_create_reset(self):
        env, pm = init_env()
        assert np.all(env.env_start == env.empty_grid), (
            "'env_start' attribute should init to a zero matrix.")

        # Fresh env
        env.create_new()
        assert np.any(env.env_start != env.empty_grid)
        assert np.all(env.grid == env.env_start), \
            "'env' should be the intial env after first call of 'env.create()'"

        # After a reset
        env.reset()  
        assert np.any(env.env_start != env.empty_grid)
        assert np.all(env.grid == env.env_start), \
            "After a reset, 'env.grid' and 'env.env_start' should match"

        # After another create, 'env' and 'env.env_start' may be different,
        env.create_new()
        assert np.any(env.env_start != env.empty_grid)
        env.reset() # but now they should match again.
        
        assert np.all(env.grid == env.env_start), \
            "After a reset, 'env.grid' and 'env.env_start' should match"
        
    def test_auto_win(self):
        """Test the agent on a 3 by 3  with 8 goals so that any action should 
        result in a terminal state and give reward 1. """

        # Initialize an environment where it's impossible to lose. 
        env = rlm_env.Env(grid_shape=(3,3), n_goals=8, hole_pct = 0.0)
        env.create_new()
        auto_win_grid = np.full(shape = env.grid.shape, 
                                fill_value = env.interactables['goal'],
                                dtype = np.int32)
        auto_win_grid[1, 1] = env.interactables['agent']
        env.env_start = auto_win_grid
        env.grid = auto_win_grid
        
        NUM_EPISODES: int = 25
        MAX_NUM_SCENES: int = 1

        episodes = []      
        for _ in range(NUM_EPISODES): 
            env.reset(); assert not np.all(env.grid == env.empty_grid)
            ep_steps: list = []
            done: bool = False

            for _ in range(MAX_NUM_SCENES):
                # Start scene
                obs = rlm_env.Observation(env=env)
                step = env.step(action_idx = random.randrange(8), 
                                obs = obs)
                observation, reward, done, info = step.values
                ep_steps.append(step)
                if done:
                    break
            # Episode complete
            if not done:
                assert np.all([step.reward == 0 for step in ep_steps])
            assert (done == True) or (len(ep_steps) == MAX_NUM_SCENES)
            episodes.append(ep_steps)

        ep_rewards = [traj[-1].reward for traj in episodes]
        assert np.all([r == 1 for r in ep_rewards]), ""

    def test_step(self):
        env: rlm_env.Env = init_env()[0]
        env.reset()

        done = False
        steps = []
        # while done != True:
        for _ in range(50):
            obs = rlm_env.Observation(env=env)
            step = env.step(
                action_idx = random.randrange(len(env.action_space)),
                obs = obs)
            steps.append(step)

        # TODO: 