from typing import List, Union, Generator
import numpy as np
import environment
import random
import networkx as nx

class PathMaker:
    def __init__(self, env: environment.Environment) -> None:
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
        explored_spots: List[list] = [list(A) for A in np.argwhere(is_explored)]
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
                path.append(shifted_spot)
                spot = shifted_spot
                break
        assert len(path) == n_steps + 1, "'path' is too short."
        return path

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
        
        def diag_path(starting_pt, ending_pt):
            displacement = np.array(ending_pt) - np.array(starting_pt)
            directions = (displacement / np.abs(displacement)).astype(int) 
            magnitude = np.abs(np.min(displacement))
            diag = np.full(shape = (magnitude + 1, 2), fill_value = starting_pt)
            for row_idx in range(1, magnitude + 1):
                diag[row_idx] = diag[row_idx - 1] + directions
            diag_path = [pt.tolist() for pt in diag]

            assert diag_path[0] == starting_pt, \
                "'diag_path[]' " 
            assert np.any(np.array(diag_path[-1]) == np.array(ending_pt)), \
                ("At least one component of the last pt in 'diag_path' should "
                + "match the corresponding component in 'ending_pt'")
            return diag_path

        def straight_shot(diag_path: List[List[int]], ending_pt):
            starting_pt = diag_path[-1]
            displacement = np.array(ending_pt) - np.array(starting_pt)
             
            assert np.any(displacement == 0)
            directions = np.where(
                displacement == 0, 0, 
                displacement / np.abs(displacement)).astype(int)
            magnitude = np.abs(np.max(displacement))
            straight = np.full(shape = (magnitude + 1, 2), 
                               fill_value = starting_pt)
            for row_idx in range(1,  magnitude + 1):
                straight[row_idx] = straight[row_idx - 1] + directions
            straight_path = [pt.tolist() for pt in straight]
            return straight_path[1:]

        diag = diag_path(pt_a, pt_b)
        shortest_path = diag + straight_shot(diag, pt_b)
        return shortest_path

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

    def force_valid_path(self):
        """Generates a random grid that has a path from start to goal.
        """
        # TODO Get out the whiteboard and write an algorithm to do this. 
        raise NotImplementedError