import numpy as np 

# pop out the state (s_0, a_0)
# Put in on the 0th row of the "graph" 
# State vector, φ(s), is projection onto feature space. 
    # Feature space is the representation of observation space when the 
    # observation space is too large.  

# Use  Q_G(φ(s), a) ← r + γ max_{a'}( Q_G (φ(s'), a')) )

class Graph:
    """Database of all winning trajectories. """
    def __init__():
        self.actions = np.NaN
        self.state_vec = np.NaN
        self.state

    def update():
        # TODO:
        raise NotImplementedError
    
class Memory: 
    """ This is snapshot in time. In the current state, s, the agent takes
    action, a, to (potentially) receive reward, r, and end up in state s'. """

    def __init__(self, state, action, episodic):
        # self.state 
        # self.action = 
        self.reward = self.get_reward() # reward received from moving s → s'
        self.episodic =
        self.associative: bool # a way of organizing episodic episodic memory

    def get_reward(self):
        self.state
        self.next_state
        raise NotImplementedError, 
        reward = None
        return reward   

class Trajectory:
    """A sequence of memories that occured in order.
    A sequence of states and actions that occured in order.
    Note, trajectories are also frequently called episodes or rollouts."""
    pass # TODO

