#%%
import numpy as np
import pathfinder
import environment

def test_set_agent_goal():
    env = environment.Environment(grid_shape=(50, 50), n_goals=60)
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
    env = environment.Environment(grid_shape=(50, 50), n_goals=20)

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

def test_pathfinder():
    env = environment.Environment(grid_shape=(100,100), n_goals=30, 
                                  hole_pct = 0.5)
    env.set_agent_goal()
    env.set_holes(0.2)
    pf = pathfinder.PathFinder(env)
    pf.pathfind()
    valid = pf.valid
    valid_path = pf.valid_path
    breakpoint()

def run_all_tests(verbose = True):
    tests = [test_set_agent_goal, test_set_holes, test_pathfinder]
    for test in tests:
        test()
        print(f"Test passed: '{test}'" if verbose else "")
 
if __name__ == "__main__":
    run_all_tests()
