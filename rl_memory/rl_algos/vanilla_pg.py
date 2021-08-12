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
from rl_memory.custom_env import representations 
from rl_memory.custom_env import agents 
from rl_memory.custom_env import environment
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
class VPGHyperParameters:
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

class VPGNetwork(nn.Module):
    """Neural network for vanilla policy gradient. Used in the first experiment
    
    Args:
        action_dim (int): The dimension of the action space, i.e. 
            len(action_space). 
        
    """
    def __init__(self, obs_size: torch.Size, action_dim: int, 
                 h_params: VPGHyperParameters):
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
    rewards: List[float] = [] 
    discounted_rewards: Optional[Array] = None
    log_probs: List[float] = [] 
    env_char_renders: List[Array] = []

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
    rewards: List[float] = []
    returns: List[float] = []
    trajectories: List = []
    distributions: List[Categorical] = []
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
        policy_network (VPGNetwork): [description]
        env (rlm.Env): [description]
        agent (rlm.Agent): [description]
    
    Attributes:
        episode_tracker (trackers.EpisodeTracker): 
        env (rlm.Env): [description]
        agent (rlm.Agent): [description]
    """

    def __init__(
            self, 
            policy_network: VPGNetwork, 
            env_like: rlm.Env, 
            agent: rlm.Agent, 
            transfer_mgmt: Optional[base.TransferLearningManagement] = None,
            discount_factor: float = 0.99
            ):
            
        self.policy_network = policy_network
        self.env_like = env_like
        self.agent = agent
        self.transfer_mgmt: base.TransferLearningManagement = transfer_mgmt
        self.discount_factor = discount_factor

        self.episode_tracker = VPGEpisodeTracker()

    def run_algo(
            self, 
            num_episodes: int, 
            max_num_scenes: int,):
        """TODO: docs"""

        env: rlm.Env = self.env_like
        for episode_idx in range(num_episodes):
            scene_tracker: trackers.SceneTracker   
            env, scene_tracker = self.on_episode_start(
                env = env, episode_idx = episode_idx)
            self.film_episode(env = env, 
                              policy_network = self.policy_network, 
                              scene_tracker = scene_tracker, 
                              max_num_scenes = max_num_scenes) 
            self.update_policy_network(
                policy_network = self.policy_network,
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
        obs: rlm.Observation = environment.Observation(env, self.agent)
        action_distribution: Categorical = (
            self.policy_network.action_distribution(obs))
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
                policy_network = self.policy_network, 
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

    def update_policy_network(
            self,
            scene_tracker: VPGSceneTracker):
        """Updates the weights and biases of the policy network."""
        discounted_rewards: Array = rl_memory.tools.discount_rewards(
            rewards = scene_tracker.rewards, 
            discount_factor = self.discount_factor)
        scene_tracker.discounted_rewards = discounted_rewards
        baselines = np.zeros(discounted_rewards.shape)
        advantages = discounted_rewards - baselines
        assert len(scene_tracker.log_probs == len(advantages)), "MISMATCH!"
        self.policy_network.update(scene_tracker.log_probs, advantages)

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
            scene_tracker.action_distribution)

class VPGExperiment:
    """formerly pg_cnn.py
    
    Args:
        env (Env): 
        agent (rlm.Agent): 
        episode_tracker (VPGEpisodeTracker):
        num_episodes (int): Number of episodes to evaluate the agent. Defaults 
            to 10000.
        discount_factor (float): Factor used to discount rewards. 
            Denoted γ (gamma) in the RL literature. Defaults to 0.99.
        transfer_freq (int): Th number of episodes the environment will remain 
            constant before randomizing for transfer learning. Defaults to 10.
        lr (float): Policy network learning rate. 
            Denoted α (alpha) in the RL literature. Defaults to 1e-3.
    
    Attributes:
        agent (rlm.Agent): 
        episode_tracker (VPGEpisodeTracker):
        num_episodes (int): Number of episodes to evaluate the agent. Defaults 
            to 10000.
        discount_factor (float): Factor used to discount rewards. A.K.A. gamma
            in the RL literature. Defaults to 0.99.
        transfer_freq (int): Th number of episodes the environment will remain 
            constant before randomizing for transfer learning. Defaults to 10.
        lr (float): Policy network learning rate. 
            Denoted α (alpha) in the RL literature. Defaults to 1e-3.
    """
    def __init__(self, 
                 env: rlm.Env, 
                 agent: rlm.Agent, 
                 episode_tracker: VPGEpisodeTracker, 
                 num_episodes: int = 10000, 
                 discount_factor: float = 0.99, 
                 transfer_freq: int = 10, 
                 lr: float = 1e-3):

        self.agent = agent
        self.episode_tracker = episode_tracker

        # Environment parameters
        self.grid_shape = env.grid_shape
        self.num_goals = env.num_goals
        self.hole_pct = env.hole_pct

        # Experiment hyperparams
        self.num_episodes = num_episodes
        self.discount_factor = discount_factor
        self.lr = lr
        self.transfer_freq = transfer_freq
        self.max_num_scenes: int = 3 * self.grid_shape[0] * self.grid_shape[1]

    def create_policy_network(
            self, 
            env: rlm.Env, 
            obs: rlm.Observation) -> VPGNetwork:
        action_dim: int = len(env.action_space)
        obs_size: torch.Size = obs.size()
        network_h_params = VPGHyperParameters(lr = self.lr)
        policy_network = VPGNetwork(
            obs_size = obs_size, action_dim = action_dim, 
            h_params = network_h_params)
        return policy_network

    @staticmethod
    def easy_env() -> Env:
        """
        Returns:
            (Env): A small, 3 by 3 environment with one goal and one hole.
        """
        easy_env: rlm.Env = environment.Env(
            grid_shape=(3, 3), n_goals=1, hole_pct=.1)
        easy_env.reset()
        return easy_env

    def pretrain_on_easy_env(
            self, 
            policy_network: VPGNetwork):
        """TODO docstring
        Methods come from RLAlgorithm
        """
        easy_env = self.easy_env()
        rl_algo = VPGAlgo(
            policy_network = policy_network,
            env = easy_env,
            agent = self.agent, )
        rl_algo.run_algo(
            num_episodes = self.num_episodes, 
            max_num_scenes = self.max_num_scenes)
        
        return rl_algo.policy_network

    def pretrain_to_threshold(
        self, 
        policy_network: VPGNetwork, 
        ep_len_threshold: float = 3.3, 
        trajectory_lookback_window: int = 500) -> VPGNetwork:
        """Recursively trains the agent (policy network) on an easy environment
        until it can solve it quickly and consistently.

        Args:
            policy_network (VPGNetwork): The network that receives pre-training. 
            ep_len_threshold (float): Defaults to 3.3.
            trajectory_lookback_window (int): Defaults to 500.
        
        Returns
            policy_network (VPGNetwork): Pre-trained network.
        """
        avg_episode_len = np.infty

        while avg_episode_len > ep_len_threshold:
            policy_network = self.pretrain_on_easy_env(
                policy_network = policy_network)
            avg_episode_len = np.mean(
                [len(traj) for traj in self.episode_tracker.trajectories[
                     -trajectory_lookback_window:]])
            print(avg_episode_len)
        return policy_network

    def experiment_vpg_transfer(self) -> VPGNetwork:
        """Runs an experiment to see if the environment is solvable with VPG 
        and pre-training on the easy environment. 

        Experiment steps:
        0. Initialize new env and policy network.
        1. Pretrain a network on the easy environment. 
        2. Transfer learn on the big environment, which has many holes. 
        
        Returns:
            policy_network
        """

        # Step 0: Initialize new env and policy network
        self.env.create_new()
        obs: rlm.Observation = environment.Observation(self.env, self.agent)
        policy_network: VPGNetwork = self.create_policy_network(
            env = self.env, obs = obs)

        # Step 1: Pretrain on the easy environment
        policy_network = self.pretrain_to_threshold(
            policy_network = policy_network)

        # Step 2: Transfer learn on the big environment.
        rl_algo = VPGAlgo(
            policy_network = policy_network, 
            env = self.env,
            agent = self.Agent)
        rl_algo.run_algo(
            num_episodes = self.num_episodes,
            max_num_scenes = self.max_num_scenes)
        return policy_network

# ----------------------------------------------------------------------
#               Begin Experiment
# ----------------------------------------------------------------------

def train(env: rlm.Env, 
          obs: rlm.Observation,
          agent: rlm.Agent, 
          num_episodes = 20, 
          discount_factor: float = .99, 
          lr = 1e-3,  
          transfer_freq = 5):
    """
    TODO: docs, test
    """

    grid_shape: Tuple[int] = env.grid.shape
    max_num_scenes = 3 * grid_shape[0] * grid_shape[1]

    # Specify parameters
    obs_size: torch.Size = obs.observation.size
    action_dim: int = len(env.action_space)
    network_h_params = VPGHyperParameters(lr = lr)
    policy_network = VPGNetwork(
        obs_size = obs_size, action_dim = action_dim, 
        h_params = network_h_params)
    episode_tracker_train = VPGEpisodeTracker()
    transfer_mgmt_train = VPGTransferLearning(transfer_freq = transfer_freq)

    # Run RL algorithm
    training_algo = VPGAlgo(
        policy_network = policy_network, 
        env_like = env, 
        agent = agent, 
        episode_tracker = episode_tracker_train, 
        transfer_mgmt = transfer_mgmt_train, 
        discount_factor = discount_factor)
    training_algo.run_algo(
        num_episodes = num_episodes, max_num_scenes = max_num_scenes)

    return policy_network, episode_tracker_train

def test(env: rlm.Env, agent: rlm.Agent, policy: nn.Module, num_episodes: int = 10):

    max_num_scenes = env.grid_shape[0] * env.grid_shape[1]
    env.create_new()

    episode_trajectories = []
    episode_rewards = []

    for e in range(num_episodes):

        episode_envs = []
        reward_sum = 0
        scene_number = 0
        done: bool = False

        while not done:
            episode_envs.append(env.render_as_char(env.grid))
            state = environment.State(env, agent)
            action_distribution = policy.action_distribution(
                state.observation.flatten())
            a = action_distribution.sample()

            new_state, r, done, info = env.step(action_idx=a, state=state)
            reward_sum += r

            scene_number += 1
            if scene_number > max_num_scenes:
                break

        episode_trajectories.append(episode_envs)
        episode_rewards.append(reward_sum)

    return episode_rewards, episode_trajectories

def main():

    # initialize agent and an environment with no holes
    james_bond: rlm.Agent = agents.Agent(4)
    env: rlm.Env = environment.Env(
        grid_shape = (10, 10), 
        n_goals = 1, 
        hole_pct = 0.4)
    env.create_new()
    obs: rlm.Observation = environment.Observation(env=env, agent=james_bond)

    episode_trajectories: Dict[List] = {}
    episode_rewards: Dict[List[float]] = {}
    policy_network, episode_tracker = train(
        env=env, obs=obs, agent=james_bond, num_episodes = 20)
    episode_trajectories['train'] = episode_tracker.trajectories
    episode_rewards['train'] =  episode_tracker.rewards
    rl_memory.tools.plot_episode_rewards(
        values = episode_rewards['train'], 
        title = "training rewards", 
        reset_frequency = 5)
    
    test_env = environment.Env(
        grid_shape=env.grid_shape, n_goals=env.n_goals, hole_pct=env.hole_pct)

    test_episode_rewards, episode_trajectories['test'] = test(
        env = test_env, 
        agent = james_bond, 
        policy = policy_network, 
        num_episodes = 10)
    rl_memory.tools.plot_episode_rewards(test_episode_rewards, "test rewards", transfer_freq=5)
