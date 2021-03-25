from typing import List, Union, Generator
import numpy as np
import environment
import random


class PathMaker:
    def __init__(self, env: environment.Environment) -> None:
        self.env = env
        self.branches = self.init_branches()
        self.valid_path: list = None

        # unexplored_spots init to everything except the agent the holes.
        pass    
        
    def init_branches(self) -> List[list]:
        agent_spot = np.argwhere(
            self.env.grid == self.env.interactables['agent']).flatten()
        (branch := []).append(agent_spot)
        (branches := []).append(branch)
        return branches
    
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
        explored_spots: List[list] = [list(A) for A in np.argwhere(is_explored)]
        assert len(env.position_space) >= len(explored_spots)

        # Store unexplored spots 
        unexplored_spots: list = []
        unexplored_spots[:] = [p for p in env.position_space
                               if (p not in explored_spots)]
        return [np.array(spot) for spot in unexplored_spots] 

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
            path = start
        else:
            raise ValueError("'start' must have type List[int] or List[list]")

        for _ in range(n_steps):
            shifted_spot: List[int]
            for shifted_spot in self.generate_shifted_spots(spot):
                if shifted_spot not in path:
                    path.append(shifted_spot)
                    spot = shifted_spot
                    break
                else:
                    continue
        assert len(path) == n_steps + 1, "'path' is too short."
        return path

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

    def make_path(self) -> np.ndarray:
        """A recursive pathfinding method to see whether it is possible for 
        the agent to solve the given randomly generated environmen. 

        Returns:
            valid_path (List[list]): List of ordered pairs that consistute a 
                guaranteed successful path for the agent. 
        """
        # self.env.grid[x, y] == self.env.interactables['goal']
        
        # TODO: Generate valid path
        # 
        # TODO: Check that the valid path is actually valid 
        raise NotImplementedError
        return self.valid_path

    # ----------------------
    # Valid path generation:
    # ----------------------

    def take_random_step(self) -> List[int]:
        # self.grid
        agent_spot = np.argwhere(
            self.env.grid == self.env.interactables['agent']).flatten()
        breakpoint()

        raise NotImplementedError
        next_spot: tuple = x, y
        return next_spot

    def force_valid_path(self):
        """Generates a random grid that has a path from start to goal.
        """
        # TODO Get out the whiteboard and write an algorithm to do this. 
        raise NotImplementedError