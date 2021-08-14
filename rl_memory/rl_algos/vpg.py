import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions
import torch.optim
import numpy as np
import dataclasses
import os, sys 
try:
    import rl_memory
except:
    exec(open('__init__.py').read()) 
    import rl_memory
import rl_memory as rlm
import rl_memory.memory
import rl_memory.tools
from rl_memory.rlm_env import representations 
from rl_memory.rlm_env import environment
from rl_memory.rl_algos import base
from rl_memory.rl_algos import trackers

# Type imports
from typing import Dict, List, Iterable, Tuple, Optional, Union
Env = rlm.Env
EnvStep = environment.EnvStep
from torch import Tensor
Array = np.ndarray
Categorical = distributions.Categorical

it = representations.ImgTransforms()

@dataclasses.dataclass
class NNHyperParameters:
    """Hyperparameters for the policy network.
    
    Q: How do hyperparameters differ from the model parameters?
    A: If you have to specify a paramter manually, then it is probably a 
    hyperparameter. For example, the learning rate is a hyperparameter."""

    lr: float
    batch_size: int = 1
    num_filters: int = 16
    filter_size: int = 2
    dropout_pct: float = 0.1
    hidden_dim: int = 5

    def __post_init__(self):
        self._check_valid_batch_size()
        self._check_valid_num_filters()
        self._check_valid_dropout_pct()
        self._check_valid_hidden_dim()

    def _check_valid_batch_size(self):
        batch_size: int = self.batch_size
        assert isinstance(batch_size, int)
        assert batch_size > 0

    def _check_valid_num_filters(self):
        num_filters: int = self.num_filters
        assert isinstance(num_filters, int)
        assert num_filters > 0

    def _check_valid_hidden_dim(self):
        hidden_dim: int = self.hidden_dim
        assert isinstance(hidden_dim, int)
        assert hidden_dim > 0

    def _check_valid_dropout_pct(self):
        dropout_pct: float = self.dropout_pct
        assert isinstance(dropout_pct, (int, float))
        assert (dropout_pct >= 0) and (dropout_pct <= 1), (
            f"'dropout_pct' must be between 0 and 1, not {dropout_pct}")

class VPGPolicyNN(nn.Module):
    """Neural network for vanilla policy gradient. Used in the first experiment
    
    Args:
        action_dim (int): The dimension of the action space, i.e. 
            len(action_space). 
        
    """
    def __init__(self, obs_size: torch.Size, action_dim: int, 
                 h_params: NNHyperParameters):
        super().__init__()
        self.batch_size: int = h_params.batch_size
        num_filters: int = h_params.num_filters
        filter_size: int = h_params.filter_size
        hidden_dim: int = h_params.hidden_dim

        # Model Architecture
        self.convnet_encoder: nn.Module = self._get_convnet_encoder(
            num_filters=num_filters, filter_size=filter_size)

        lin_dim: int = num_filters * (obs_size[0] - filter_size) ** 2

        self.fc_layers = nn.Sequential(
            nn.Linear(lin_dim, hidden_dim),
                nn.Dropout(h_params.dropout_pct),
                nn.ReLU(),
            nn.Linear(hidden_dim, action_dim))

        self.optimizer = torch.optim.Adam(self.parameters(), lr = h_params.lr)
        self.action_dim = action_dim

        self.memory = rl_memory.memory.Memory()

    def forward(self, x: Tensor):
        x: Tensor = x.float()
        x = self.convnet_encoder(x)
        x = self.fc_layers(x.flatten())
        return x

    def _get_convnet_encoder(self, num_filters, filter_size) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = num_filters, 
                        kernel_size = filter_size, stride = 1),
            nn.BatchNorm2d(num_filters),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=filter_size, stride=1),) 

    def action_distribution(self, state) -> Categorical:
        """
        Args:
            state: TODO

        input: state (TODO adjust for when state is sequence of observations??)
        ouput: softmax(nn valuation of each action)

        Returns: 
            action_distribution (Categorical): 
        """
        state_rgb: Tensor = it.grid_to_rgb(state).unsqueeze(0)
        action_logits = self.forward(state_rgb)
        action_probs: Tensor = F.softmax(input = action_logits, dim=-1)
        action_distribution: Categorical = torch.distributions.Categorical(
            probs = action_probs)
        return action_distribution

    def update(self, log_probs, advantages):  # add entropy
        """Update with advantage policy gradient theorem."""
        advantages = torch.FloatTensor(advantages)
        log_probs = torch.cat(tensors = log_probs, dim = 0)
        assert log_probs.requires_grad
        assert not advantages.requires_grad

        loss = - torch.mean(log_probs * advantages.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

@dataclasses.dataclass
class VPGSceneTracker(trackers.SceneTracker):
    """Container class for tracking scene-level results.

    Attributes:
        rewards (List[float]): Scene rewards. Defaults to empty list.
        discounted_rewards (Array): Scene discounted rewards. Defaults to None.
        log_probs (List[float]): Scene log probabilities for the action chosen 
            by the agent. Defaults to empty list.
        env_char_renders (List[Array]): Character renders of the env grid. 
            Used for visualizing how the agent moves.
    """
    rewards: List[float] = dataclasses.field(default_factory = list)
    discounted_rewards: Optional[Array] = None
    log_probs: List[float] = dataclasses.field(default_factory = list)
    env_char_renders: List[Array] = dataclasses.field(default_factory = list)

@dataclasses.dataclass
class VPGEpisodeTracker(trackers.EpisodeTracker):
    """Container class for tracking episode-level results.

    Attributes:
        rewards (List[float]): List of total rewards for each episode. Each 
            element of 'rewards' is the total reward for a particular episode.
        returns (List[float]): List of total returns for each episode. Each 
            element of 'returns' is the total discounted reward for a particular 
            episode. Returns refers to the discounted reward. Thus, return is 
            equivalent to reward if the episode has length one. 
        trajectories (List[]):
        distributions (List[Categorical]): 
    """
    rewards: List[float] = dataclasses.field(default_factory = list)
    returns: List[float] = dataclasses.field(default_factory = list)
    trajectories: List = dataclasses.field(default_factory = list)
    distributions: List[Categorical] = dataclasses.field(default_factory = list)
        # [torch.exp(dist.log_prob(i)) for i in dist.enumerate_support()]

class VPGTransferLearning(base.TransferLearningManagement):
    """Manages the transfer learning process for Vanilla Policy Gradient."""

    def __init__(self, transfer_freq: int):
        self.transfer_freq = transfer_freq

    def transfer(self, ep_idx: int, env: rlm.Env) -> rlm.Env:
        """Transfers the agent to a random environment based on the transfer 
        frequency attribute, 'freq'.
        """

        freq = self.transfer_freq
        if (ep_idx % freq == 0) and (ep_idx > 0):
            env.create_new() # Reset to a random environment (same params)
        else:
            env.reset() # Reset to the same environment
        return env

class VPGAlgo(base.RLAlgorithm):
    """Runs the Vanilla Policy Gradient algorithm.

    Args:
        policy_nn (VPGPolicyNN): [description]
        env (rlm.Env): [description]
        agent (rlm.Agent): [description]
    
    Attributes:
        episode_tracker (trackers.EpisodeTracker): 
        env (rlm.Env): [description]
        agent (rlm.Agent): [description]
    """

    def __init__(
            self, 
            policy_nn: VPGPolicyNN, 
            env_like: rlm.Env, 
            agent: rlm.Agent, 
            transfer_mgmt: Optional[base.TransferLearningManagement] = None,
            discount_factor: float = 0.99
            ):
            
        self.policy_nn = policy_nn
        self.env_like = env_like
        self.agent = agent
        self.transfer_mgmt: base.TransferLearningManagement = transfer_mgmt
        self.discount_factor = discount_factor

        self.episode_tracker = VPGEpisodeTracker()

    def run_algo(
            self, 
            num_episodes: int, 
            max_num_scenes: int,
            training: bool = True):
        """TODO: docs"""
        train_val: str = "train" if training else "val"
        if train_val == "train":
            self.policy_nn.train()
        else:
            self.policy_nn.eval()

        env: rlm.Env = self.env_like
        for episode_idx in range(num_episodes):
            scene_tracker: trackers.SceneTracker   
            env, scene_tracker = self.on_episode_start(
                env = env, episode_idx = episode_idx)
            self.film_episode(env = env, 
                              scene_tracker = scene_tracker, 
                              max_num_scenes = max_num_scenes) 
            if train_val == "train":
                self.update_policy_nn(
                    scene_tracker = scene_tracker)
            self.on_episode_end(
                episode_tracker = self.episode_tracker,
                scene_tracker = scene_tracker)

    def on_episode_start(
            self, 
            env: rlm.Env, 
            episode_idx: int) -> Tuple[rlm.Env, trackers.SceneTracker]:
        if self.transfer_mgmt is not None:
            env = self.transfer_mgmt.transfer(
                ep_idx = episode_idx, env = env)
        else:
            env.reset()
        scene_tracker = VPGSceneTracker()
        return env, scene_tracker

    def film_scene(
        self,
        env: rlm.Env, 
        scene_tracker: VPGSceneTracker) -> bool:
        """Runs a scene. A scene is one step of an episode.

        Args:
            scene_tracker (VPGSceneTracker): Stores scene-level results.

        Returns:
            done (bool): Whether or not the episode is finished.
        """
        # Observe environment
        obs: rlm.Observation = environment.Observation(
            env = env, agent = self.agent)
        action_distribution: Categorical = (
            self.policy_nn.action_distribution(obs))
        action_idx: int = action_distribution.sample()

        # Perform action
        env_step: rlm.EnvStep = env.step(
            action_idx = action_idx, obs = obs)
        next_obs, reward, done, info = env_step
        
        scene_tracker.log_probs.append(action_distribution.log_prob(
            action_idx).unsqueeze(0))
        scene_tracker.rewards.append(reward)
        scene_tracker.env_char_renders.append(env.render_as_char(env.grid))
        return done
    
    def on_scene_end(self, env: rlm.Env, scene_tracker: VPGSceneTracker):
        scene_tracker.env_char_renders.append(
            env.render_as_char(env.grid))
    
    @staticmethod
    def agent_took_too_long(time: int, max_time: int) -> bool:
        return time == max_time
    
    def film_episode(
            self, 
            env: rlm.Env, 
            scene_tracker: VPGSceneTracker, 
            max_num_scenes: int):
        """Runs an episode.

        Args:
            env (Env): [description]
            scene_tracker (VPGSceneTracker): [description]
        """
        scene_idx = 0
        done: bool = False
        while not done:
            done: bool = self.film_scene(
                env = env, 
                scene_tracker = scene_tracker)

            scene_idx += 1
            if done:  
                self.on_scene_end(env = env, scene_tracker = scene_tracker)
                break
            elif self.agent_took_too_long(
                time = scene_idx,
                max_time = max_num_scenes):
                scene_tracker.rewards[-1] = -1  
                done = True
            else:
                continue

    def update_policy_nn(
            self,
            scene_tracker: VPGSceneTracker):
        """Updates the weights and biases of the policy network."""
        discounted_rewards: Array = rl_memory.tools.discount_rewards(
            rewards = scene_tracker.rewards, 
            discount_factor = self.discount_factor)
        scene_tracker.discounted_rewards = discounted_rewards
        baselines = np.zeros(discounted_rewards.shape)
        advantages = discounted_rewards - baselines
        assert len(scene_tracker.log_probs) == len(advantages), "MISMATCH!"
        self.policy_nn.update(scene_tracker.log_probs, advantages)

    def on_episode_end(
            self,
            episode_tracker: VPGEpisodeTracker,
            scene_tracker: VPGSceneTracker):
        """Stores episode results and any other actions at episode end.

        Args:
            episode_tracker (VPGEpisodeTracker): [description]
            scene_tracker (VPGSceneTracker): [description]

        Returns:
            [type]: [description]
        """
        total_reward = np.sum(scene_tracker.rewards)
        total_return = np.sum(scene_tracker.discounted_rewards)
        episode_tracker.rewards.append(total_reward)
        episode_tracker.returns.append(total_return)
        episode_tracker.trajectories.append(
            scene_tracker.env_char_renders)
        episode_tracker.distributions.append(
            scene_tracker.log_probs)
