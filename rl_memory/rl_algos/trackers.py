import abc
import numpy as np
from typing import List
Array = np.ndarray

class EpisodeTracker(abc.ABC):
    """Container class for tracking episode-level results.

    Attributes:
        episode_rewards (List[float]): List of total rewards for each episode. 
            Each element of 'episode_rewards' is the total reward for a 
            particular episode.
        episode_disc_rewards (List[float]): List of total discounted rewards for 
            each episode. Each element of 'episode_disc_rewards' is the total 
            discounted reward for a particular episode. 
            
    Note, "returns" is another term that refers to the discounted reward. 
    Thus, return (discounted reward) is equivalent to reward if the episode 
    has length one. 
    """

    episode_rewards: List[float] 
    episode_returns: List[float] 
    
    def __post_init__(self):
        self._has_required_attributes()

    def _has_required_attributes(self):
        req_attrs: List[str] = ['episode_rewards', 'episode_disc_rewards']
        for attr in req_attrs:
            if not hasattr(self, attr): 
                raise AttributeError(f"Missing attribute: '{attr}'")

class SceneTracker(abc.ABC):
    """Container class for tracking scene-level results.

    Attributes:
        scene_rewards (List[float]): Scene rewards. Defaults to empty list.
        scene_disc_rewards (np.ndarray): An array containing the discounted 
            reward for each scene. Defaults to None.
    """

    scene_rewards: List[float] 
    scene_disc_rewards: np.ndarray = None

    def __post_init__(self):
        self._has_required_attributes()

    def _has_required_attributes(self): 
        req_attrs: List[str] = ['scene_rewards', 'scene_disc_rewards']
        for attr in req_attrs:
            if not hasattr(self, attr): 
                raise AttributeError(f"Missing attribute: '{attr}'")