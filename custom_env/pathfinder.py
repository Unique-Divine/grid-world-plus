from typing import List
import numpy as np
import environment
import random

class PathFinder:
    def __init__(self, env: environment.Environment) -> None:
        self.branches = self.init_branches()
        self.env = env
        self.unexplored_spots: list = self.init_unexplored_spots()
        self.valid: bool = False
        self.valid_path: list = None

        # unexplored_spots init to everything except the agent the holes.
        pass    
        
    def init_branches(self) -> List[list]:
        agent_spot = list(np.argwhere(
            self.env.grid == self.env.interactables['agent']))
        (branch := []).append(agent_spot)
        (branches := []).append(branch)
        return branches
    
    def init_unexplored_spots(self) -> List[list]: 
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
        explored_spots: list = list(np.argwhere(is_explored))
        
        # Store unexplored spots 
        unexplored_spots: list = []
        unexplored_spots[:] = [p for p in env.position_space
                               if (p not in explored_spots)]
        assert len(set(unexplored_spots).intersection(set(explored_spots))) == 0
        return unexplored_spots 

    @staticmethod
    def generate_shifted_spots(spot) -> List[list]:
        """
        Args:
            spot (list): An ordered pair (x, y) for a particular matrix element
                on the grid. 
        Returns:
            shifted_spots (List[list]): A list containing the coordinates for 
                each position that neighbors the input 'spot' argument. 
                These shifted coordinates are then shuffled to give more variety
                to the search path. 
        """
        nsew_shifts = [[1, 0], [0, 1], [0, -1], [-1, 0]]
        cross_shifts = [[1, 1], [1, -1], [-1, 1], [-1, -1]]  
        shifts = nsew_shifts + cross_shifts 
        shifted_spots = []
        for shift in shifts:
            dx, dy = shift
            shifted_spot = [spot[0] + dx, spot[1] + dy]
            shifted_spots.append(shifted_spot)
        random.shuffle(shifted_spots) # randomize the order of the shifts
        return shifted_spots

    def explore(self, spot, branch):
        branch.append(spot) # Add spot to the end of the branch
        self.branches.append(branch) # Update tree with branch extension
        self.unexplored_spots.remove(spot) # make spot "explored"

    def pathfind(self) -> bool:
        """A recursive pathfinding method to see whether it is possible for 
        the agent to solve the given randomly generated environmen. 

        Returns:
            (bool): 
        """
        for branch in self.branches:
            spot: list = branch[-1]
            assert len(spot) == 2
            possible_spots = self.generate_shifted_spots(spot)
            for spot in possible_spots:                
                if spot in self.unexplored_spots:
                    self.explore(spot, branch)

        branch_tips = [self.env.grid[branch[-1]] for branch in self.branches]
        if self.env.interactables['goal'] in branch_tips: # goal discovered
            self.valid = True
            is_valid_path = [tip == self.env.interactables['goal'] 
                for tip in branch_tips]
            self.valid_path = self.branches[is_valid_path.index(True)] 
            # TODO: Check the valid fn works
            # TODO: Check that the valid path is actually valid
            
        elif len(self.unexplored_spots) == 0: # all spots explored
            self.valid = False
        else: # keep exploring through recursion
            self.pathfind()

        return self.valid