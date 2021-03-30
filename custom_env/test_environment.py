#%%
import numpy as np
import random
from typing import List
import pathmaker
import environment


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

# -------------------------------------------------------
# pathmaker.py tests
# -------------------------------------------------------

def init_env():
    env = environment.Env(grid_shape=(50,50), n_goals=10, 
                                  hole_pct = 0.5)
    pm = pathmaker.PathMaker(env)
    return env, pm

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
    try:
        assert valid_path[0] == env.agent_position
        assert valid_path[-1] == env.goal_position
    except:
        breakpoint()

def test_create_reset():
    env, pm = init_env()
    passes = []
    passes.append(env.env_start == None) # T

    # Fresh env
    env.create()
    assert env.env_start != None
    assert np.all(env.grid == env.env_start.grid), \
        "'env' should be the intial env after first call of 'env.create()'"

    # After a reset
    env = env.reset()  
    assert env.env_start != None
    assert np.all(env.grid == env.env_start.grid), \
        "After a reset, 'env' and 'env.env_start' should match"

    # After another create, 'env' and 'env.env_start' are probably different,
    env.create()
    assert env.env_start != None
    env = env.reset() # but now they should match again.
    assert np.all(env.grid == env.env_start.grid), \
        "After a reset, 'env' and 'env.env_start' should match"
    
def run_all_tests(verbose = True):
    tests = [test_set_agent_goal, test_set_holes, test_generate_shifted_spots, 
             test_random_walk, test_shortest_path, test_make_valid_path,
             test_create_reset]
    for test in tests:
        test()
        if verbose:
            print(f"Test passed: '{test.__name__}'")
    print("\nAll tests passed." if verbose else "")
 
if __name__ == "__main__":
    run_all_tests()