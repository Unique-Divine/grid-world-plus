import abc

try:
    import grid_world_plus
except:
    exec(open('__init__.py').read()) 
    import grid_world_plus
import grid_world_plus as rlm
from typing import Any, Optional, Tuple

class RLAlgorithm(abc.ABC):
    """Representation of a reinforcement learning algorithm
    
    The class implements the following structure:

    ```python
    def run_algo(self):
        scene_start: Tuple[rlm.Env, rlm.SceneTracker] = self.on_scene_start()
        env, scene_tracker = scene_start
        for episode_idx in range(self.num_episodes):
            self.film_epsiode() 
            self.update_policy_nn()
            self.on_episode_end()
    ```
    """
    num_episodes: int

    @abc.abstractmethod
    def film_scene(self) -> bool:
        """Runs a scene. A scene is one step of an episode.

        Returns:
            done (bool): Whether or not the episode is finished.
        """
    
    @abc.abstractmethod
    def on_episode_start(self) -> Tuple[rlm.Env, rlm.SceneTracker, Optional[Any]]:
        """Called at the beginning of an episode. Initializes the environment 
        and tracks results with the scene tracker."""

    @abc.abstractmethod
    def on_scene_end(self):
        """Called at the end of a scene."""
    
    @staticmethod
    @abc.abstractmethod
    def agent_took_too_long(time: int, max_time: int) -> bool:
        return time == max_time
    
    @abc.abstractmethod
    def film_episode(self, scene_tracker, *args: Any, **kwargs: Any):
        """Runs an episode."""

    @abc.abstractmethod
    def update_policy_nn(self, *args: Any, **kwargs: Any):
        """Updates the weights and biases of the neural network(s)."""

    @abc.abstractmethod
    def on_episode_end(
            self,
            episode_tracker: rlm.EpisodeTracker,
            scene_tracker: rlm.SceneTracker) -> Any:
        """Called at the end of an episode.

        Args:
            episode_tracker (EpisodeTracker): [description]
            scene_tracker (SceneTracker): [description]

        Returns:
            (Any): [description]
        """
    
    @abc.abstractmethod
    def run_algo(self):
        """Executes the reinforcement learning algorithm, i.e. runs 
        'num_episodes' episode iterations.
        
        Recommended format:
        ```python
        def run_algo(self):
            scene_start: Tuple[rlm.Env, rlm.SceneTracker] = self.on_scene_start()
            env, scene_tracker = scene_start
            for episode_idx in range(self.num_episodes):
                self.film_epsiode() 
                self.update_policy_nn()
                self.on_episode_end()
        ```
        """

class TransferLearningManagement(abc.ABC):
    """Abstract object that manages the transfer learning process."""

    @abc.abstractmethod
    def __init__(self, transfer_freq: int):
        self.transfer_freq = transfer_freq

    @abc.abstractmethod
    def transfer(self, ep_idx: int, env: rlm.Env, *args: Any) -> rlm.Env:
        """Transfers the agent to a random environment based on the transfer 
        frequency attribute, 'freq'.
        """

