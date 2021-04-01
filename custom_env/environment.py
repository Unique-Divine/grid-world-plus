#%%
import numpy as np
import sys
import copy
import random
import collections
import copy
from agent import Agent
from typing import List, Union, Generator, NamedTuple
import warnings
warnings.filterwarnings("ignore")
import gym.utils

class Point(np.ndarray):
    """A 1D np.ndarray of size 2 that contains the row and column
    indices for a point in the environment.  

    Examples
    --------
    >>> p1 = Point(1, 2)
    >>> p1
    array([1, 2], dtype=int16)
    >>> p2 = Point([1, 3])
    >>> p2
    array([1, 3], dtype=int16)
    >>> p1 == p2
    array([ True, False])
    """
    def __new__(cls, *args):
        if len(args) == 2:
            self = np.asarray([*args], dtype=np.int16)
        elif len(args) == 1:
            self = np.asarray(args[0], dtype=np.int16)
        else:
            raise ValueError("oof")
        return self

class Env:
    """A variable Frozen Lake environment. It's the Frozen Lake from AI Gym with
    a varying starting position for the agent, holes, and goal(s). Movements are
    deterministic rather than stochastic and each environment is solvable, so a
    'perfect' agent can get reward 1 on every episode.
    
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

    Examples:
    ---------
    >>> from agent import Agent
    >>> env = Env() # initializes an environment
    >>> env.reset() # creates or resets the environment
    >>> james = Agent(4)
    # An episode could then look like:
    ```
    while done!= True:  
        s = State(env, james) # the state of James in the environment
        random_action = random.randrange(8)
        step = env.step(action_idx = random_action, s)
        observation, reward, done, info = step
    replay_buffer.append( ... )
    ```
    """
    def __init__(self, grid_shape = (10, 10), hole_pct = 0.2, n_goals = 3):
        self.interactables = {'frozen': 0, 'hole': 1, 'goal': 2, 
                              'agent': 7, 'blocked': 3} 

        # Set board dimensions and initalize to an "empty" grid. 
        if len(grid_shape) != 2:
            raise ValueError("'grid_shape' must be a list-like of length 2.")
        self.grid = np.full(shape = grid_shape, 
                            fill_value = self.interactables['frozen'], 
                            dtype = np.int32)
        assert self.grid.shape == grid_shape
        # Initialize grid helper parameters  
        self._position_space: List[list] = self.position_space
        self.action_space: List[Point] = self.get_action_space()
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

    def __repr__(self) -> str:
        return f"Env:\n{self.render_as_char(self.grid)}"
    def __str__(self) -> str:
        return str(self.grid)

    def render(self):
        raise NotImplementedError
        pass

    def render_as_char(self, grid) -> np.ndarray:
        interactables_to_char = {
            self.interactables['frozen']: "_", 
            self.interactables['hole']: "o", 
            self.interactables['goal']: "G", 
            self.interactables['agent']: "A",
            self.interactables['blocked']: "'"} 
        char_grid = np.asarray(
            [interactables_to_char[e] for e in grid.flatten()],
            dtype = str).reshape(grid.shape)
        return char_grid

    # --------------------------------------------------------------------
    # Properties 
    # --------------------------------------------------------------------
    def get_action_space(self) -> List[Point]:
        action_space: List[list] = [[-1, 1], [-1, 0], [-1, -1], [0, -1],
                                    [1, -1], [1, 0], [1, 1], [0, 1]]
        action_space: List[Point] = [Point(p) for p in action_space]
        return action_space

    @property
    def position_space(self) -> List[List[int]]:
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

    def set_env_start(self, env):
        self.env_start = Env(
            grid_shape = env.grid.shape,
            hole_pct = self.hole_pct,
            n_goals = env.n_goals
        )
        self.env_start.grid = env.grid
        self.env_start.open_positions = env.open_positions
        self.env_start.env_start = self.env_start
        assert self.env_start != None

    # --------------------------------------------------------------------
    # Functions for the user
    # --------------------------------------------------------------------

    def create(self):
        """Place all of the interactables on the grid to create a new env."""
        self.set_agent_goal() # Create agent and goals
        
        # Clear a path for the agent
        valid_path = PathMaker(self).make_valid_path()
        for position in valid_path:
            if position in self.open_positions:
                self.open_positions.remove(position)
        
        # Place holes in some of the empty spaces
        self.set_holes()
        
        # Save initial state if this is the first time create() has been called.
        if self.env_start == None:
            self.set_env_start(env = self)


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
    
    def step(self, action_idx: int, state) -> NamedTuple:
        action: Point = self.action_space[action_idx]
        desired_position: Point = state.center + action
        new_x, new_y = desired_position
        interactable: int = state.observation[new_x, new_y]
        
        def move():
            x, y = self.agent_position
            new_x, new_y = Point(self.agent_position) + action
            self.grid[x, y] = self.interactables['frozen']
            self.grid[new_x, new_y] = self.interactables['agent']
        def unable_to_move():
            pass

        observation: np.ndarray
        reward: float
        done: bool 
        info: str
        
        if interactable == self.interactables['frozen']:
            move()
            reward = 0 
            done = False
        elif interactable == self.interactables['hole']:
            move()
            reward = -1
            done = True
        elif interactable == self.interactables['goal']:
            move()
            reward = 1
            done = True
        elif interactable == self.interactables['blocked']:
            unable_to_move()
            reward = 0
            done = False
        elif interactable == self.interactables['agent']:
            raise NotImplementedError("There shouldn't be two agents yet.")
            # TODO
        else:
            raise ValueError(f"interactable: '{interactable}' is not in "
                +f"interactables: {self.interactables}")
        
        observation = State(self, state.agent).observation 
        info = ""
        Step = collections.namedtuple(
            "Step", ["observation", "reward", "done", "info"])
        return Step(observation, reward, done, info)
        
class PathMaker:
    def __init__(self, env: Env) -> None:
        self.env = env
        self.valid_path: list = None 
    
    def init_unexplored_spots(self) -> List[np.ndarray]: 
        """Initialize the `unexplored_spots` attribute for the pathfinding
        algorithm. Unexplored spots are everything on the board that isn't an
        agent, hole, or blocked.
        
        Returns:
            unexplored_spots (List[list]): List of coordinate pairs to be used
                as indices of the env.grid matrix."""       
        env = self.env
        # Explored spots: Locations in the grid with agent or hole
        is_agent: np.ndarray = (env.grid == env.interactables['agent'])
        is_hole: np.ndarray = (env.grid == env.interactables['hole'])
        is_explored = (is_agent | is_hole)
        explored_spots: List[list] = [A.tolist() for A in np.argwhere(is_explored)]
        assert len(env.position_space) >= len(explored_spots)

        # Store unexplored spots 
        unexplored_spots: list = []
        unexplored_spots[:] = [p for p in env.position_space
                               if (p not in explored_spots)]
        return [np.array(spot) for spot in unexplored_spots] 

    def generate_shifted_spots(self, spot) -> Generator[List[int], None, None]:
        """Generator for a viable position adjacent to the input position.

        Args:
            spot (list): An ordered pair (x, y) for a particular matrix element
                on the grid. 
        Returns:
            shifted_spot (List[list]): A position that neighbors the input 
                'spot' argument. This shifted coordinate is randomly selected
                from the available options on the 'env.grid'. 
        """
        nsew_shifts = [[1, 0], [0, 1], [0, -1], [-1, 0]]
        cross_shifts = [[1, 1], [1, -1], [-1, 1], [-1, -1]]  
        shifts: List[list] = nsew_shifts + cross_shifts
        shifted_spots = []
        x_0, y_0 = spot
        for shift in shifts:
            dx, dy = shift
            x, y = x_0 + dx, y_0 + dy 
            shifted_spot = [x, y]
            if shifted_spot in self.env.position_space:
                shifted_spots.append(shifted_spot)
        random.shuffle(shifted_spots) # randomize the order of the shifts
        for shifted_spot in shifted_spots:
            yield shifted_spot
    
    def random_steps(self, n_steps: int, starting_spot):
        """Helper function for 'random_walk()'. This generates a step in the
        discrete random walk.

        Args:
            n_steps (int): Number of steps
            starting_spot (List[int]): A position (x, y) on the env.grid 

        Yields:
            shifted_spot (List[int]): Position of the next random step. 
        """
        spot = starting_spot
        for _ in range(n_steps):
            shifted_spot: List[int]
            for shifted_spot in self.generate_shifted_spots(spot):
                yield shifted_spot
                spot = shifted_spot
                break
    
    def random_walk(self, n_steps: int, 
                    start: List[Union[int, List[int]]]) -> List[List[int]]:
        assert isinstance(start, list), "'start' must be a list."
        assert len(start) > 0, \
            "'start' cannot be empty. The random walk needs an starting point."
        if isinstance(start[0], int):
            assert len(start) == 2, "..." # TODO
            spot = start
            path = []; path.append(spot)
        elif isinstance(start[0], list):
            assert np.all([len(pt) == 2 for pt in start]), (
                "The current value for 'start' has type List(list). As a list "
                + "of ordered pairs, each of element of 'start' should have a "
                + "length of 2. ")
            spot = start[-1]
            path = copy.deepcopy(start)
        else:
            raise ValueError("'start' must have type List[int] or List[list]")

        starting_spot = spot        
        for step in self.random_steps(n_steps, starting_spot):
            path.append(step)
            
        # for _ in range(n_steps):
        #     shifted_spot: List[int]
        #     for shifted_spot in self.generate_shifted_spots(spot):
        #         path.append(shifted_spot)
        #         spot = shifted_spot
        #         break
        proper_path_length: bool = ((len(path) == n_steps + 1) 
                                    or (len(path) == n_steps + len(start)))
        assert proper_path_length, ("'path' is too short. "
            + f"len(path): {len(path)}, n_steps: {n_steps}")
        return path

    def diag_path(self, starting_pt: List[int], ending_pt: List[int]):
        """[summary] TODO

        Args:
            starting_pt (List[int]): [description]
            ending_pt (List[int]): [description]

        Returns:
            [type]: [description]
        """
        displacement = np.array(ending_pt) - np.array(starting_pt)
        if np.all(displacement == 0):
            # Case 1: 'ending_pt' has already been reached
            return [starting_pt]
        elif np.any(displacement == 0):
            # Case 2: 'displacement' is vertical or horizontal
            return self.straight_shot([starting_pt], ending_pt)
        
        directions = (displacement / np.abs(displacement)).astype(int) 
        magnitude: int = np.min(np.abs(displacement))
        diag = np.full(shape = (magnitude + 1, 2), fill_value = starting_pt)
        for row_idx in range(1, magnitude + 1):
            diag[row_idx] = diag[row_idx - 1] + directions
        diag_path = [pt.tolist() for pt in diag]

        assert diag_path[0] == starting_pt, \
            "'diag_path[0]' should be the starting point." 
        assert np.any(np.array(diag_path[-1]) == np.array(ending_pt)), \
            ("At least one component of the last pt in 'diag_path' should "
            + "match the corresponding component in 'ending_pt'")
        return diag_path

    @staticmethod
    def straight_shot(diag_path: List[List[int]], 
                        ending_pt: List[int]) -> List[List[int]]:
        """[summary] TODO

        Args:
            diag_path (List[List[int]]): [description]
            ending_pt (List[int]): [description]

        Returns:
            List[List[int]]: [description]
        """
        starting_pt = diag_path[-1]
        displacement = np.array(ending_pt) - np.array(starting_pt)
        assert np.any(displacement == 0), \
            "At least one of the displacement components should be 0."
        if np.all(displacement == 0):
            # 'ending_pt' has already been reached on 'diag_path'.
            return diag_path[1:]
        directions = np.where(
            displacement == 0, 0, 
            displacement / np.abs(displacement)).astype(int)
        magnitude = np.max(np.abs(displacement))
        straight = np.full(shape = (magnitude + 1, 2), 
                            fill_value = starting_pt)
        for row_idx in range(1,  magnitude + 1):
            straight[row_idx] = straight[row_idx - 1] + directions
        straight_path = [pt.tolist() for pt in straight]
        assert straight_path[-1] == ending_pt, ("'straight_path' is not "
            + "ending at 'ending_pt'.")
        return straight_path

    def shortest_path(self, path_a: list, 
                      path_b: list) -> List[Union[List, int]]:
        """Find the shortest path between the ends of two paths on the env.grid.

        Args:
            path_a (list): A position of type List[int] or list of positions of 
                type List[List[int]] on the env.grid. 
            path_b (list): A position of type List[int] or list of positions of 
                type List[List[int]] on the env.grid.
                
        Raises:
            ValueError: If 'path_a' and 'path_b' is not a list
            ValueError: If the elements of the paths have the wrong type

        Returns:
            List[Union[List, int]]: The shortest path between the endpoints of 
                'path_a' and 'path_b'. 
        """
        # Verify that both paths are lists.
        assert np.all([isinstance(path, list) for path in [path_a, path_b]]), \
            "Both 'path_a' and 'path_b' must be lists."  
        # Verify that path_a is type List[int] or List[List[int]]
        if isinstance(path_a[0], int):
            pt_a = path_a
        elif isinstance(path_a[0], list):
            pt_a = path_a[-1]
        else:
            raise ValueError("'path_a' must be a position or list of positions")
        # Verify that path_b is type List[int] or List[List[int]]
        if isinstance(path_b[0], int):
            pt_b = path_b
        elif isinstance(path_b[0], list):
            pt_b = path_b[-1]
        else:
            raise ValueError("'path_b' must be a position or list of positions")

        # Compute shortest path
        diag = self.diag_path(pt_a, pt_b)
        straight = self.straight_shot(diag, pt_b)
        if [diag[0], diag[-1]] == [pt_a, pt_b]:
            shortest_path = diag
        elif [straight[0], straight[-1]] == [pt_a, pt_b]:
            shortest_path = straight
        else:
            shortest_path = diag + straight[1:]
        try:
            assert [shortest_path[0], shortest_path[-1]] == [pt_a, pt_b]
        except:
            breakpoint()
        return shortest_path

    def make_valid_path(self, rw_pct = 0.15, sp_pct = 0.15) -> np.ndarray:
        """Specifies a guaranteed path without holes between the agent and a 
        goal. By setting the holes on the environment outside of 'valid_path', 
        we can guarantee that the environment is solvable.

        Args:
            rw_pct (float): "Random walk percentage". The percentage of the 
                length of 'env.grid' that will be taken as random walk steps.
                Directly affects the variable 'rw_steps'.
            sp_pct (float): "Shortest path percentage". The percentage of the 
                length of 'env.grid' that will be taken as shortest path steps.
                Directly affects the variable 'sp_steps'.  
        Returns:
            valid_path (List[List[int]]): List of ordered pairs that consistute a 
                guaranteed successful path for the agent. 
        """
        # TODO: Generate valid path
        agent_position = self.env.agent_position
        goal_position = self.env.goal_position 
        path_a, path_g = agent_position, goal_position
        rw_steps: int = round(rw_pct * len(self.env.grid))
        sp_steps: int = round(0.5 * sp_pct * len(self.env.grid))
        rw_steps = 1 if rw_steps < 1 else rw_steps
        sp_steps = 1 if sp_steps < 1 else sp_steps

        done: bool = False
        while done != True: # Run until 'path_a' reaches the goal
            
            # Random walk from both agent and goal starting positions
            path_a, path_g = [self.random_walk(n_steps = rw_steps , start = path) 
                            for path in [path_a, path_g]]

            # Get shortest path b/w the endpts of both paths
            shortest = self.shortest_path(path_a, path_g)
            if len(shortest) <= 2:
                path_a.append(shortest[-1])
                path_a += path_g[::-1]
                done = True
                try:
                    assert [path_a[0], path_a[-1]] == [agent_position, goal_position] 
                except:
                    print('case 1')
                    breakpoint()
            elif (len(shortest) - 2) <= (2 * sp_steps):
            # If shortest path steps 'sp_steps' spans shortest 
                path_a += shortest[1:-1]
                path_a += path_g[::-1]
                done = True
                try:
                    assert [path_a[0], path_a[-1]] == [agent_position, goal_position] 
                except:
                    print('case 2')
                    breakpoint()
            else:
                # Follow the shortest path for sp_steps
                front_of_shortest = shortest[1:1 + sp_steps]
                back_of_shortest = shortest[-(1 + sp_steps): -1]
                path_a += front_of_shortest
                path_g += back_of_shortest[::-1]
        
        # TODO: Verify that optimal_g connects to path_g and optimal_a connects to path_a
        
        # TODO: Check that the valid path is actually valid -> write test:
        # 1. Verify that valid_path starts with agent position and ends with goal
        # 2. Verify that the shifts between each position in the path are <= 1.
        valid_path: List[List[int]] = path_a
        return valid_path

class State:
    """[summary]
    
    Args:
        env (Env): An environment with an agent in it. The environment contains 
            all information needed to get a state for reinforcement learning. 

    Attributes:
        center_abs (Point): The agent's on the 'env.grid'.
        center (Point): The agent's position on the current sight window.
    """
    def __init__(self, env: Env, agent: Agent) -> None:
        self.env = env
        self.agent: Agent = agent
        self._center: Point = self.center
        self.center_abs: Point = Point(env.agent_position)
        self._observation: np.ndarray = self.observation
    
    @property
    def center(self) -> Point:
        return Point([self.agent.sight_distance] * 2)
    
    @property 
    def observation(self) -> np.ndarray:
        sd: int = self.agent.sight_distance
        center_abs: Point = self.center_abs
        observation = np.empty(
            shape= [self.agent.sight_distance * 2 + 1] * 2, 
            dtype = np.int16)
        row_view: range = range(center_abs[0] - sd, center_abs[0] + sd + 1)
        col_view: range = range(center_abs[1] - sd, center_abs[1] + sd + 1)
        def views(row_view, col_view) -> Generator:
            for row_idx in row_view:
                for col_idx in col_view:
                    displacement = Point(row_idx, col_idx) - center_abs
                    relative_position: Point = self.center + displacement
                    rel_row, rel_col = relative_position
                    yield row_idx, col_idx, rel_row, rel_col
        
        for view in views(row_view, col_view):
            row_idx, col_idx, rel_row, rel_col = view 
            if [row_idx, col_idx] in self.env.position_space:
                observation[rel_row, rel_col] = self.env.grid[row_idx, col_idx]
            else:
                observation[rel_row, rel_col] = self.env.interactables[
                    'blocked']
        return observation

    def __repr__(self):
        return f"{self.env.render_as_char(self.observation)}"

def toy():
    def init_env():
        env = Env(grid_shape=(10,10), n_goals=2, 
                                    hole_pct = 0.5)
        pm = PathMaker(env)
        return env, pm

    env, pm = init_env()
    env.reset()

    Alex = Agent(2)
    s0 = State(env, Alex)

    p1  = Point(1, 2)
    p2 = Point([1, 3])

    breakpoint()
    print('code')

toy()

# Useful implementation links: 
# https://en.wikipedia.org/wiki/Depth-first_search
# https://docs.python.org/3/library/random.html

# def frozen_lake_original_map():
#     from gym.envs.toy_text import frozen_lake
#     print(frozen_lake.generate_random_map())