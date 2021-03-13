from typing import List
import numpy as np
import environment

class PathFinder:
    def __init__(self, env: environment.Environment) -> None:
        self.branches = self.init_branches()
        self.valid = None
        self.unexplored_spots: list = self.init_unexplored_spots()

        # unexplored_spots init to everything except the agent the holes.
        pass    
        
    def init_branches(self) -> List[list]:
        agent_spot = list(np.argwhere(
            self.env.grid == self.env.interactables['agent']))
        (branch := []).append(agent_spot)
        (branches := []).append(branch)
        return branches
    
    def init_unexplored_spots(self) -> List[list]:
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

    def explore(self, spot, branch):
        branch.append(spot) # Add spot to the end of the branch
        self.branches.append(branch) # Update tree with branch extension
        self.unexplored_spots.remove(spot) # make spot "explored"

    
    def pathfind(self):
        def generate_shifted_spots(spot) -> List[list]:
            nsew_shifts = [[1, 0], [0, 1], [0, -1], [-1, 0]]
            cross_shifts = [[1, 1], [1, -1], [-1, 1], [-1, -1]]  
            shifts = nsew_shifts + cross_shifts 
            shifted_spots = []
            for shift in shifts:
                dx, dy = shift
                shifted_spot = [spot[0] + dx, spot[1] + dy]
                shifted_spots.append(shifted_spot)
            return shifted_spots

        for branch in self.branches:
            spot: list = branch[-1]
            assert len(spot) == 2
            possible_spots = generate_shifted_spots(spot)
            for spot in possible_spots:
                # TODO 
                
                if spot in self.unexplored_spots:
                    self.explore(spot, branch)
                    
                # elif len(self.unexplored_spots) == 0
                # else branch more
                raise NotImplementedError  
