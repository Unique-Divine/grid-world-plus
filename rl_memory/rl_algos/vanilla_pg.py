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
from typing import Dict, List, Iterable, Tuple, Optional
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
            env: rlm.Env, 
            agent: rlm.Agent):
        self.policy_network = policy_network
        self.env = env
        self.agent = agent
        self.episode_tracker = VPGEpisodeTracker()

    def run_algo(self, num_episodes: int, max_num_scenes: int):
        env: rlm.Env
        scene_tracker: trackers.SceneTracker
        env, scene_tracker = self.on_scene_start()
        for episode_idx in range(num_episodes):
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

    def on_scene_start(self) -> Tuple[rlm.Env, trackers.SceneTracker]:
        self.env.reset()
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
        agent (Agent):
        episode_tracker (VPGEpisodeTracker):
    """
    def __init__(self, 
                 env: rlm.Env, 
                 agent: Agent, 
                 episode_tracker: VPGEpisodeTracker):

        # env
        self.grid_shape = env.grid_shape
        self.num_goals = env.num_goals
        self.hole_pct = env.hole_pct

        self.agent = agent
        self.episode_tracker = episode_tracker

        # learning hyperparams
        self.num_episodes = 10000
        self.discount_factor = .99 # gamma
        self.lr = 1e-3
        self.reset_frequency = 10
        self.max_num_scenes = 3 * self.grid_shape[0] * self.grid_shape[1]

    def create_policy_network(self, 
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

# env hyperparams
grid_shape = (3, 3)
n_goals = 1
hole_pct = 0

# initialize agent and environment
james_bond: Agent = agents.Agent(4)
env: rlm.Env = environment.Env(
    grid_shape=grid_shape, n_goals=n_goals, hole_pct=hole_pct)
env.create_new()
obs: rlm.Observation = environment.Observation(env, james_bond)

def train(env: rlm.Env = env, agent: rlm.Agent = james_bond, 
          obs: rlm.Observation = obs,
          num_episodes = 20, gamma = .99, lr = 1e-3,  
          reset_frequency = 5):

    max_num_scenes = 3 * grid_shape[0] * grid_shape[1]

    # init model
    obs_size: torch.Size = obs.observation.size
    action_dim: int = len(env.action_space)
    h_params = VPGHyperParameters(lr = lr)
    policy_network = VPGNetwork(
        obs_size = obs_size, action_dim = action_dim, 
        h_params = h_params)

    # tracking important things
    training_tracker = VPGEpisodeTracker()

    for episode_idx in range(num_episodes):
        print(f"episode {episode_idx}")

        # evaluate policy on current init conditions 5 times before switching to new init conditions
        if (episode_idx % reset_frequency == 0) and (episode_idx > 0):
            env.create_new() # Reset to a random environment (same params)
        else:
            env.reset() # Reset to the same environment

        done = False
        log_probs = []  # tracks log prob of each action taken in a scene

        episode_grids = []  # so you can see what the agent did in the episode

        scene_number = 0  # track to be able to terminate episodes that drag on for too long
        scene_rewards = []
        while not done:
            episode_grids.append(env.render_as_char(env.grid))
            obs: rlm.Observation = environment.Observation(env, agent)
            action_distribution = policy_network.action_distribution(
                obs.flatten())
            # [torch.exp(action_dist.log_prob(i)) 
            # for i in action_dist.enumerate_support()] - see probs
            action_idx: int = action_distribution.sample()
            assert action_idx in np.arange(0, action_dim).tolist()

            log_probs.append(action_distribution.log_prob(
                action_idx).unsqueeze(0))
            next_obs, reward, done, info = env.step(
                action_idx = action_idx, obs = obs)  

            scene_number += 1
            if scene_number > max_num_scenes:
                reward = -1
                scene_rewards.append(reward)
                break

            if done:
                episode_grids.append(env.render_as_char(env.grid))

            scene_rewards.append(reward)
        policy_network.update(scene_rewards, gamma, log_probs)

        episode_rewards = np.sum(scene_rewards)
        training_tracker.rewards.append(episode_rewards)
        training_tracker.trajectories.append(episode_grids)

    # return the trained model,
    return policy_network, training_tracker

def test(env: rlm.Env, agent: Agent, policy: nn.Module, num_episodes: int = 10):

    max_num_scenes = grid_shape[0] * grid_shape[1]
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
    episode_trajectories: Dict[List] = {}
    policy_network, training_episode_rewards, episode_trajectories['train'] = train()
    rl_memory.tools.plot_episode_rewards(training_episode_rewards, "training rewards", 5)
    test_env = environment.Env(
        grid_shape=grid_shape, n_goals=n_goals, hole_pct=hole_pct)

    test_episode_rewards, episode_trajectories['test'] = test(
        env = test_env, agent = james_bond, policy = policy_network, num_episodes = 10)
    rl_memory.tools.plot_episode_rewards(test_episode_rewards, "test rewards", reset_frequency=5)
