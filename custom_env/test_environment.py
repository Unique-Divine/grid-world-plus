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
    
    print(env.grid)
    # print(env.open_positions)
    
test_set_agent_goal()
print("Test passed.")


def test_set_holes():
    env = environment.Environment(grid_shape=(50, 50), n_goals=60)
    env.set_agent_goal()

    is_hole: np.ndarry = (env.grid == env.interactables['hole'])
    n_holes: int = is_hole.sum()
    n_positions = np.product(env.grid.shape)
    assert n_holes == int(n_positions * env.hole_pct)