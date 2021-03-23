from typing import List
import numpy as np
import environment
import random

class PathMaker:
    def __init__(self, env: environment.Environment) -> None:
        self.env = env
        self.branches = self.init_branches()
        self.unexplored_spots: List[np.ndarray] = self.init_unexplored_spots()
        self.valid: bool = False
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

    def generate_shifted_spots(self, spot) -> List[list]:
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

    def explore(self, spot, branch):
        branch.append(spot) # Add spot to the end of the branch
        self.branches.append(branch) # Update tree with branch extension
        # Make spot "explored", i.e. remove it from unexplored_spots
        self.unexplored_spots = [x for x in self.unexplored_spots 
                                 if not (spot == x).all()]
        return branch

    def possible_unexplored(self):
        """Generator -> yields spot that is both possible to reach 
        and unexplored in a at the end of a branch. 

        Args:
            possible ([type]): [description]
            unexplored ([type]): [description]
            branch 
        """
        for branch in self.branches:
            branch[-1].size == 2, \
                "'spot' has too many dimensions to be (x, y)"
            possible_spots = self.generate_shifted_spots(branch[-1])        
            for spot in possible_spots:
                for u_spot in self.unexplored_spots:
                    if np.array_equal(spot, u_spot):
                        extended_branch = self.explore(spot, branch)
                        yield spot, extended_branch

    def pathfind(self) -> np.ndarray:
        """A recursive pathfinding method to see whether it is possible for 
        the agent to solve the given randomly generated environmen. 

        Returns:
            (bool): 
        """
        for (x, y), branch in self.possible_unexplored():
            if self.env.grid[x, y] == self.env.interactables['goal']:
                # Case 1: Goal reached
                self.valid = True
                self.valid_path = np.vstack(branch)
                break
            elif len(self.unexplored_spots) == 0: 
                # Case 2: all spots explored
                self.valid = False
                break
            else: 
                # Case 3: Keep exploring through recursion
                self.pathfind()
            
            # TODO: Check the valid fn works
            # TODO: Check that the valid path is actually valid 
            print('yes', branch)
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