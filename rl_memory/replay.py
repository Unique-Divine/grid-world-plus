#!/usr/bin/env python3
"""Module containing all abstractions related to experience replay.

Classes:
    Memory: Dataclass representing a memory, a.k.a. an experience. 
        Contains (obs, action_idx, reward, next_obs).
    Experience: Alias for 'Memory'
    Trajectory: An ordered sequence of memories.
"""
import dataclasses
from rl_memory import rlm_env
import rl_memory as rlm
from typing import Any, Iterable, List, Optional, Tuple


@dataclasses.dataclass
class Memory:
    """Dataclass representing a memory, a.k.a. an experience.
    
    Args and Attributes:
        obs (Observation)
        action_idx (int)
        reward (float)
        next_obs (Optional[Observation]): Defaults to None.
    """
    obs: rlm.Observation
    action_idx: int
    reward: float
    next_obs: Optional[rlm.Observation] = None

    def __iter__(self) -> Iterable[rlm.Observation, int, float, rlm.Observation]:
        return iter([self.obs, self.action_idx, self.reward, self.next_obs])
    
    @property
    def is_terminal(self) -> bool:
        return self.next_obs is None

Experience = Memory

class Trajectory:
    """An ordered sequence of memories.
    
    Attributes:
        states
        actions
        rewards
        next_states
    """

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
            raise IndexError()
        return self.memories[idx]

    def __repr__(self):
        return f"<{self.__class__.__name__} at {hex(id(self))}>"

    def masked(self) -> List[Memory]: 
        # Mask pct_mask of the memories by converting them to zeros 
        pass 


