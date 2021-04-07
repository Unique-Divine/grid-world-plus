# memory.py
from typing import List

class Memory:

    def __init__(self, state, action, reward):
        self.state = state
        self.action = action
        self.reward = reward

class Trajectory:

    def __init__(self, memories: List[Memory] = []):
        self.memories = memories

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
    

    