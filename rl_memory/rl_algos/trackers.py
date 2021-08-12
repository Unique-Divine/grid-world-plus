import abc
import dataclasses
import numpy as np
from typing import List
Array = np.ndarray

@dataclasses.dataclass
class EpisodeTracker(abc.ABC):
    """Container class for tracking episode-level results.

    Attributes:
        rewards (List[float]): List of total rewards for each episode. Each 
            element of 'rewards' is the total reward for a particular episode.
        returns (List[float]): List of total returns for each episode. Each 
            element of 'returns' is the total discounted reward for a particular 
            episode. Returns refers to the discounted reward. Thus, return is 
            equivalent to reward if the episode has length one. 
    """

    rewards: List[float] = []
    returns: List[float] = []
     
@dataclasses.dataclass
class SceneTracker(abc.ABC):
    """Container class for tracking scene-level results.

    Attributes:
        rewards (List[float]): Scene rewards. Defaults to empty list.
        discounted_rewards (Array): Scene discounted rewards. Defaults to None.
    """
    rewards: List[float] = [] 
    discounted_rewards: Array = None