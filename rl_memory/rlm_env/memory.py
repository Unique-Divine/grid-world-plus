#!/usr/bin/env python3
from rl_memory import rlm_env
from typing import List, NamedTuple, Optional

class Memory(NamedTuple):
    """NamedTuple representing a memory, or experience.
    
    Args and Attributes:
        obs (rlm_env.Observation)
        action_idx (int)
        reward (float)
        next_obs (Optional[rlm_env.Observation]): Defaults to None.
    """
    obs: rlm_env.Observation
    action_idx: int
    reward: float
    next_obs: Optional[rlm_env.Observation]


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
        pass 


