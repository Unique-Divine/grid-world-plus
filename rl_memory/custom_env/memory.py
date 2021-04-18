#!/usr/bin/env python3
from typing import List
try:
    import rl_memory
except:
    exec(open('__init__.py').read()) 
    import rl_memory
from rl_memory.custom_env import environment
import environment.Observation

class Memory:

    def __init__(self, state, action, reward):
        self.state = state
        self.action = action
        self.reward = reward

class Trajectory:

    def __init__(self, memories: List[Memory] = [], pct_mask: float = 0.35):
        self.memories = memories
        self.pr_mask

    @property
    def rewards(self) -> List[float]:
        return [memory.reward for memory in self.memories]

    @property
    def actions(self) -> list:
        return [memory.action for memory in self.memories]

    @property
    def states(self) -> list: 
        return [memory.state for memory in self.memories]

    def __len__(self) -> int:
        return len(self.memories)

    def __getitem__(self, idx) -> Memory:
        if idx >= len(self):
            raise IndexError
        return self.memories[idx]

    def __repr__(self):
        return f"<{self.__class__.__name__} at {hex(id(self))}>"

    def masked(self) -> List[Memory]: 
        # Mask pct_mask of the memories by converting them to zeros 
        


