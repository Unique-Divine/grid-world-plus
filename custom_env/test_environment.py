#%%
import numpy as np
import random
from typing import List
from agent import Agent
import environment



def init_env():
    """Helper function for setting up a random environments for tests. 
    Returns:
        env: An empty environment without any agents, goals, or holes. 
        pm: An instance of the Pathmaker class.  
    """
    env = environment.Env(grid_shape=(10,10), n_goals=2, 
                                  hole_pct = 0.5)
    pm = environment.PathMaker(env)
    return env, pm

# -------------------------------------------------------
# envrionment.py tests
# -------------------------------------------------------

def test_set_agent_goal():
    env = environment.Env(grid_shape=(50, 50), n_goals=60)
    for item_name in ['agent', 'hole', 'goal', 'blocked']:
        n_items = int((env.grid == env.interactables[item_name]).sum())
        expected_count = 0
        assert n_items == expected_count, \
            f"Too many '{item_name}' items on the board."

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
    
def test_set_holes():
    env = environment.Env(grid_shape=(50, 50), n_goals=20)

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

def test_step():
    env: environment.Env = init_env()[0]
    env.reset()

    james_bond = Agent(4)

    done = False
    steps = []
    # while done != True:
    for _ in range(5):
        s = environment.State(env, james_bond)
        step = env.step(action_idx = 0, state = s)
        steps.append(step)
    breakpoint()
        
# -------------------------------------------------------
# pathmaker.py tests
# -------------------------------------------------------

def test_generate_shifted_spots():
    env, pm = init_env()
    spot: List[int] = random.choice(env.position_space)
    for shifted_spot in pm.generate_shifted_spots(spot):
        # Verify that 'shifted_spot' is on the grid.
        assert shifted_spot in env.position_space, "Invalid move in random walk"
        
        # Verify that 'shifted_spot' is only 1 space away from 'spot'.
        positions = np.vstack([spot, shifted_spot]) 
        abs_displacement = np.abs(positions[0] - positions[1])
        assert np.all(abs_displacement <= 1), "'shifted spot' is too far away."

def test_random_walk():
    env, pm = init_env()
    spot: List[int] = random.choice(env.position_space)
    n_steps =  env.grid.shape[0] // 2
    random_path = pm.random_walk(n_steps = n_steps, start = spot)
    assert len(random_path) == n_steps + 1, "'path' is too short."

def test_shortest_path():
    env, pm = init_env()
    env.set_agent_goal()
    path = pm.shortest_path(env.agent_position, env.goal_position)
    assert path[0] == env.agent_position
    assert path[-1] == env.goal_position

def test_make_valid_path():
    env, pm = init_env()
    env.set_agent_goal()
    valid_path = pm.make_valid_path()
    assert valid_path[0] == env.agent_position
    assert valid_path[-1] == env.goal_position

# -------------------------------------------------------
# Environment update tests
# -------------------------------------------------------

def test_create_reset():
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

    # After another create, 'env' and 'env.env_start' are probably different,
    env.create_new()
    assert np.any(env.env_start != env.empty_grid)
    env.reset() # but now they should match again.
    
    assert np.all(env.grid == env.env_start), \
        "After a reset, 'env.grid' and 'env.env_start' should match"
    
def test_auto_win():
    """Test the agent on a 3 by 3  with 8 goals so that any action should result
    in a terminal state and give reward 1. """
    env = environment.Env(grid_shape=(3,3), n_goals=8, hole_pct = 0.0)
    james_bond = Agent(4)
    env.create_new()
    
    NUM_EPISODES = 20
    MAX_SCENE_IDX = 2

    episodes = []      
    for _ in range(NUM_EPISODES): 
        env.reset(); print(f'reset | episode {_}')
        ep_steps: list = []
        scene_idx: int = 0 
        done: bool = False
        while not done:
            # Start scene
            state = environment.State(env, james_bond)
            step = env.step(action_idx = random.randrange(8), 
                            state = state)
            observation, reward, done, info = step
            ep_steps.append(step)

            if scene_idx == MAX_SCENE_IDX:
                break   
            scene_idx += 1 

        # Episode complete
        if not done:
            assert np.all([step.reward == 0 for step in ep_steps])
        assert (done == True) or (len(ep_steps) == MAX_SCENE_IDX)
 
        episodes.append(ep_steps)
        print(f'Episode {_} complete.')
    
    assert np.all([e[-1].reward == 1 for e in episodes]), ""

# ------------------------------------------------------------------
# Run all 
# ------------------------------------------------------------------

def run_all_tests(verbose = True):
    tests = [test_set_agent_goal, test_set_holes, test_generate_shifted_spots, 
             test_random_walk, test_shortest_path, test_make_valid_path,
             test_create_reset, test_auto_win]
    for test in tests:
        test()
        if verbose:
            print(f"Test passed: '{test.__name__}'")
    print("\nAll tests passed." if verbose else "")
 
if __name__ == "__main__":
    for _ in range(1):
        run_all_tests()