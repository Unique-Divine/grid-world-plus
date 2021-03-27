from typing import List, Union, Generator
import numpy as np
import environment
import random
import copy
import warnings
warnings.filterwarnings("ignore")

class PathMaker:
    def __init__(self, env: environment.Env) -> None:
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
                "'diag_path[0]' should be the starting point." 
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
        path_a, path_g = self.env.agent_position, self.env.goal_position
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
                done = True
            elif (len(shortest) - 2) <= (2 * sp_steps):
            # If shortest path steps 'sp_steps' spans shortest 
                path_a += shortest[1:-1]
                path_a += path_g[::-1]
                done = True
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

    # ----------------------
    # Valid path generation:
    # ----------------------

    def force_valid_path(self):
        """Generates a random grid that has a path from start to goal.
        """
        # TODO Get out the whiteboard and write an algorithm to do this. 
        raise NotImplementedError