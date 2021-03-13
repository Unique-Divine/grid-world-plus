#%%
import environment
import numpy as np

def test_set_agent_goal():
    env = environment.Environment(grid_shape=(50, 50), n_goals=60)
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
    
    # TODO: Make sure n_goals is actually 60.

    print(env.grid)
    # print(env.open_positions)
    

def test_set_holes():
    env = environment.Environment(grid_shape=(50, 50), n_goals=20)
    env.set_agent_goal()
    env.set_holes()

    n_agents: int = int((env.grid == env.interactables['agent']).sum())
    n_goals: int = int((env.grid == env.interactables['goal']).sum())
    n_holes: int = int((env.grid == env.interactables['hole']).sum())
    n_frozen: int = int((env.grid == env.interactables['frozen']).sum())
    n_positions = np.product(env.grid.shape)
    assert n_positions == np.sum([n_agents, n_goals, n_holes, n_frozen])

    n_previously_frozen = n_positions - n_agents - n_goals 
    expected_n_holes = int(n_previously_frozen * env.hole_pct)
    assert n_holes == expected_n_holes
    
    print(env.grid[:10,:10])

test_set_agent_goal()
print("Test passed.")
test_set_holes()
print("Test passed.")

