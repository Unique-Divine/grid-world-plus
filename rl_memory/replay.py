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
    obs: rlm_env.Observation
    action_idx: int
    reward: float
    next_obs: Optional[rlm_env.Observation] = None

    def __iter__(self) -> Iterable[Tuple[
            rlm_env.Observation, int, float, rlm_env.Observation]]:
        return iter([self.obs, self.action_idx, self.reward, self.next_obs])
    
    @property
    def is_terminal(self) -> bool:
        return self.next_obs is None

Experience = Memory

class Trajectory:
    """An ordered sequence of memories.
    
    Args: 
        memories (List[Memory]): Memories that make up the trajectory.

    Attributes:
        memories (List[Memory]): Memories that make up the trajectory.
        obs_seq (List[rlm_env.Observation]): Observations.
        action_idxs (List[int]): Action indices that map to actions.
        rewards (List[float]): Reward for each memory.
        next_obs_seq (List[Optional[rlm_env.Observation]]): The resultant 'next_obs'
            after corresponding to each obs, action_idx, and reward. 
    """

    def __init__(self, memories: List[Memory] = [], pct_mask: float = 0.35):
        self.memories = memories
        self.pct_mask = pct_mask

    @property
    def rewards(self) -> List[float]:
        return [memory.reward for memory in self.memories]

    @property
    def action_idxs(self) -> List[int]:
        return [memory.action_idx for memory in self.memories]

    @property
    def obs_seq(self) -> List[rlm_env.Observation]: 
        return [memory.obs for memory in self.memories]

    @property
    def next_obs_seq(self) -> List[Optional[rlm_env.Observation]]: 
        return [memory.next_obs for memory in self.memories]

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


