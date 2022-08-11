#%%
import os, sys
import grid_world_plus
import numpy as np
import torch
import random
import copy
import warnings; warnings.filterwarnings("ignore")
from grid_world_plus import env

# Type imports
from typing import List, Tuple
from torch import Tensor
Array = np.ndarray
Env = env.Env
PathMaker = env.PathMaker

def init_env() -> Tuple[Env, PathMaker]:
    """Helper function for setting up a random environments for tests. 
    Returns:
        env: An empty environment without any agents, goals, or holes. 
        pm: An instance of the Pathmaker class.  
    """
    world: Env = env.Env(grid_shape=(15,15), n_goals=4, 
                                  hole_pct = 0.3)
    pm: PathMaker = grid_world_plus.env.PathMaker(world)
    return world, pm

class TestEnvInit:
    def test_set_agent_goal(self):
        world = env.Env(grid_shape=(50, 50), n_goals=60)
        for item_name in ['agent', 'hole', 'goal', 'blocked']:
            n_items = int((world.grid == world.interactables[item_name]).sum())
            expected_count = 0
            assert n_items == expected_count, \
                f"Too many '{item_name}' items on the board."

        world.set_agent_goal()
        # Find interactables such as the agent and goal
        nonfrozen_spots: Array = np.argwhere(
            world.grid != world.interactables['frozen'])
        assert nonfrozen_spots.ndim == 2
        assert nonfrozen_spots.size >= 4
        assert nonfrozen_spots.size % 2 == 0
        nonfrozen_spots = list(nonfrozen_spots)

        # Subset the dictionary of possible environment interactables
        env_interactables: list  = list(world.interactables.keys())
        assert 'frozen' in env_interactables
        env_interactables.remove('frozen')    
        nonfrozen: list = env_interactables
        nonfrozen: dict = {
            k: v for k, v in world.interactables.items() if k in nonfrozen}

        # Verify that each nonfrozen spot is actually not frozen.
        for spot in nonfrozen_spots:
            assert spot.size == 2
            x, y = spot
            env_object = world.grid[x, y]
            assert env_object in list(nonfrozen.values())
        
    def test_set_holes(self):
        world = env.Env(grid_shape=(50, 50), n_goals=20)

        world.set_agent_goal()
        n_holes: int = int((world.grid == world.interactables['hole']).sum())
        assert n_holes == 0, "Nonzero number of holes before creation." 

        # Calculate expected number of holes before they are placed on the board.
        n_agents = int((world.grid == world.interactables['agent']).sum())
        n_goals = int((world.grid == world.interactables['goal']).sum())
        n_frozen = int((world.grid == world.interactables['frozen']).sum())
        n_positions = np.product(world.grid.shape) 
        assert n_positions == np.sum([n_agents, n_goals, n_holes, n_frozen])
        n_previously_frozen = n_frozen
        expected_n_holes = int(n_previously_frozen * world.hole_pct)

        # Place holes on the board
        world.set_holes() 
        n_holes: int = int((world.grid == world.interactables['hole']).sum())
        assert n_holes != 0, "No holes were created by 'set_holes()'. Odd."
        n_frozen = int((world.grid == world.interactables['frozen']).sum())

        assert n_holes == expected_n_holes, \
            "Mistakes were made in 'world.set_holes()'"

class TestPathMaker:
    """ """
    def test_generate_shifted_spots(self):
        world, pm = init_env()
        spot: List[int] = random.choice(world.position_space)
        for shifted_spot in pm.generate_shifted_spots(spot):
            # Verify that 'shifted_spot' is on the grid.
            assert shifted_spot in world.position_space, (
                "Invalid move in random walk")
            
            # Verify that 'shifted_spot' is only 1 space away from 'spot'.
            positions = np.vstack([spot, shifted_spot]) 
            abs_displacement = np.abs(positions[0] - positions[1])
            assert np.all(abs_displacement <= 1), (
                "'shifted spot' is too far away.")

    def test_random_walk(self):
        world, pm = init_env() # ignore type: Env, PathMaker 
        spot: List[int] = random.choice(world.position_space)
        n_steps =  world.grid.shape[0] // 2
        random_path = pm.random_walk(n_steps = n_steps, start = spot)
        assert len(random_path) == n_steps + 1, "'path' is too short."

    def test_shortest_path(self):
        world, pm = init_env()
        world.set_agent_goal()
        path = pm.shortest_path(world.agent_position, world.goal_position)
        assert path[0] == world.agent_position
        assert path[-1] == world.goal_position

    def test_make_valid_path(self):
        world, pm = init_env()
        world.set_agent_goal()
        valid_path = pm.make_valid_path()
        assert valid_path[0] == world.agent_position
        assert valid_path[-1] == world.goal_position

class TestStateObservation:
    """Unit tests for initialization of Observation and State instances."""
    def test_obs_init(self):
        world, pm = init_env()
        world.reset()
        obs = env.Observation(env = world)
        assert isinstance(obs, torch.Tensor)
        # TODO 

    def test_state_start(self):
        pass # TODO
    
    def test_state_higher_k(self):
        pass # TODO

class TestEnvIntegration:
    """Tests for the environment updates. """
    def test_create_reset(self):
        world, pm = init_env()
        assert np.all(world.env_start == world.empty_grid), (
            "'env_start' attribute should init to a zero matrix.")

        # Fresh env
        world.create_new()
        assert np.any(world.env_start != world.empty_grid)
        assert np.all(world.grid == world.env_start), \
            "'world' should be the intial env after first call of 'world.create()'"

        # After a reset
        world.reset()  
        assert np.any(world.env_start != world.empty_grid)
        assert np.all(world.grid == world.env_start), \
            "After a reset, 'world.grid' and 'world.env_start' should match"

        # After another create, 'env' and 'env.env_start' may be different,
        world.create_new()
        assert np.any(world.env_start != world.empty_grid)
        world.reset() # but now they should match again.
        
        assert np.all(world.grid == world.env_start), \
            "After a reset, 'world.grid' and 'world.env_start' should match"
        
    def test_auto_win(self):
        """Test the agent on a 3 by 3  with 8 goals so that any action should 
        result in a terminal state and give reward 1. """

        # Initialize an environment where it's impossible to lose. 
        world = env.Env(grid_shape=(3,3), n_goals=8, hole_pct = 0.0)
        world.create_new()
        auto_win_grid = np.full(shape = world.grid.shape, 
                                fill_value = world.interactables['goal'],
                                dtype = np.int32)
        auto_win_grid[1, 1] = world.interactables['agent']
        world.env_start = auto_win_grid
        world.grid = auto_win_grid
        
        NUM_EPISODES: int = 25
        MAX_NUM_SCENES: int = 1

        episodes = []      
        for _ in range(NUM_EPISODES): 
            world.reset(); assert not np.all(world.grid == world.empty_grid)
            ep_steps: list = []
            done: bool = False

            for _ in range(MAX_NUM_SCENES):
                # Start scene
                obs = env.Observation(env=world)
                step = world.step(action_idx = random.randrange(8), 
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
        world: env.Env = init_env()[0]
        world.reset()

        done = False
        steps = []
        # while done != True:
        for _ in range(50):
            obs = env.Observation(env=world)
            step = world.step(
                action_idx = random.randrange(len(world.action_space)),
                obs = obs)
            steps.append(step)

        # TODO: 