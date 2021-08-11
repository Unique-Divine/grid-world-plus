import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions
import numpy as np
from dataclasses import dataclass
import os, sys 
try:
    import rl_memory
except:
    exec(open('__init__.py').read()) 
    import rl_memory
import rl_memory.memory
import rl_memory.tools
from rl_memory.custom_env import representations 
from rl_memory.custom_env import agents 
from rl_memory.custom_env import environment
# Type imports
from typing import Dict, List, Iterable
Env = rl_memory.Env
Observation = rl_memory.Observation
State = rl_memory.State
Agent = rl_memory.Agent
from torch import Tensor
Categorical = distributions.Categorical

it = representations.ImageTransforms()

@dataclass
class VPGHyperParameters:
    """Hyperparameters for the policy network.
    
    Q: How do hyperparameters differ from the model parameters?
    A: If you have to specify a paramter manually, then it is probably a 
    hyperparamter. For example, the learning rate is a hyperparameter."""

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
        self.convnet_encoder = self._get_convnet_encoder(
            num_filters=num_filters, filter_size=filter_size)

        lin_dim = num_filters * (obs_size[0] - filter_size) ** 2

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

@dataclass
class VPGEpisodeTracker:
    """Container class for tracking episode results.

    Attributes:
        rewards (List[]): List of episode rewards. Each element of 'rewards' is 
            the total reward for a particular episode.
        returns (List[]): List of episode returns. Each element of 'returns' is
            the total discounted reward for a particular episode. Returns refers 
            to the discounted reward. Thus, return is equivalent to reward if 
            the episode has length one. 
        trajectories (List[]):
        distributions (List[Categorical]): 
    """
    rewards: List[float] = []
    returns: List[float] = []
    trajectories: List = []
    distributions: List[Categorical] = []
        # [torch.exp(dist.log_prob(i)) for i in dist.enumerate_support()]

class VPGExperiment:
    """formerly pg_cnn.py
    
    Args:
        env (Env): 
        agent (Agent):
        episode_tracker (VPGEpisodeTracker):
    """
    def __init__(self, 
                 env: Env, 
                 agent: Agent, 
                 episode_tracker: VPGEpisodeTracker):

        # env
        self.grid_shape = env.grid_shape
        self.num_goals = env.num_goals
        self.hole_pct = env.hole_pct
        self.custom_env = None

        self.agent = agent
        self.episode_tracker = episode_tracker

        # learning hyperparams
        self.num_episodes = 10000
        self.discount_factor = .99 # gamma
        self.lr = 1e-3
        self.reset_frequency = 10
        self.max_num_scenes = 3 * self.grid_shape[0] * self.grid_shape[1]

    def create_policy_network(self, 
                              env: Env, 
                              obs: Observation) -> VPGNetwork:
        action_dim: int = len(env.action_space)
        obs_size: torch.Size = obs.size()
        network_h_params = VPGHyperParameters(lr = self.lr)
        policy_network = VPGNetwork(
            obs_size = obs_size, action_dim = action_dim, 
            h_params = network_h_params)
        return policy_network

    def init_env(self, use_custom=False) -> Env:
        if use_custom:
            env = self.custom_env
        else:
            env = self.env

        env.create_new()
        return env

    @staticmethod
    def easy_env() -> Env:
        """
        Returns:
            (Env): A small, 3 by 3 environment with one goal and one hole.
        """
        easy_env: Env = environment.Env(
            grid_shape=(3, 3), n_goals=1, hole_pct=.1)
        easy_env.reset()
        return easy_env

    def pretrain_on_easy_env(self, policy_network: VPGNetwork):
        for episode_idx in range(self.num_episodes):
            env = self.easy_env() # as opposed to env.reset()
            log_probs = []
            rewards = []
            env_char_renders = [] # visualize agent movement during episode

            t = 0
            done = False

            while not done:
                env_char_renders.append(env.render_as_char(env.grid))

                obs: Observation = environment.Observation(env, self.agent)
                action_distribution: Categorical = (
                    policy_network.action_distribution(obs))
                action_idx: int = action_distribution.sample()

                ns, reward, done, info = env.step(
                    action_idx = action_idx, obs = obs)
                # ns unused b/c env tracks

                t += 1

                if t == self.max_num_scenes:
                    # Agent took too long to solve -> negative reward
                    reward = -1  
                    done = True

                if done:  # get the last env_render
                    env_char_renders.append(env.render_as_char(env.grid))

                log_probs.append(action_distribution.log_prob(
                    action_idx).unsqueeze(0))
                rewards.append(reward)
                self.distributions.append(action_distribution)

            returns = rl_memory.tools.discount_rewards(
                rewards = rewards, discount_factor = self.discount_factor)
            baselines = np.zeros(returns.shape)
            advantages = returns - baselines
            if len(log_probs) != len(advantages):
                print("mismatch")
            policy_network.update(log_probs, advantages)

            total_reward = np.sum(rewards)
            total_return = np.sum(returns)
            if total_reward > 1:
                print("big total")

            self.episode_tracker.rewards.append(total_reward)
            self.episode_tracker.returns.append(total_return)
            self.episode_tracker.trajectories.append(env_char_renders)
        return policy_network

    def pretrain_to_threshold(
        self, 
        policy_network: VPGNetwork, 
        scene_len_threshold: float = 3.3, 
        trajectory_lookback_window: int = 500) -> VPGNetwork:
        """Recursively trains the agent (policy network) on an easy environment
        until it can solve it quickly and consistently.

        Args:
            policy_network (VPGNetwork): The network that receives pre-training. 
            scene_len_threshold (float): Defaults to 3.3.
            trajectory_lookback_window (int): Defaults to 500.
        """
        avg_scene_len = np.infty

        while avg_scene_len > scene_len_threshold:
            policy_network = self.pretrain_on_easy_env(
                policy_network = policy_network)
            avg_scene_len = np.mean(
                [len(traj) for traj in self.episode_tracker.trajectories[
                     -trajectory_lookback_window:]])
            print(avg_scene_len)
        return policy_network

    def pg_cnn_transfer(self):
        """
        run the agent on a big environment with many holes to see if 
        vanilla PG can solve env
        """

        # init new env and get initial state
        self.env.create_new()
        obs: Observation = environment.Observation(self.env, self.agent)

        policy_network: VPGNetwork = self.create_policy_network(
            env = self.env, obs = obs)
        policy_network = self.pretrain_to_threshold(
            policy_network = policy_network)

        for episode_idx in range(self.num_episodes):
            self.env.reset()

            log_probs = []
            rewards = []

            # visualize agent movement during episode
            env_char_renders = []

            t = 0
            done = False

            while not done:
                env_char_renders.append(self.env.render_as_char(self.env.grid))

                obs: Observation = environment.Observation(self.env, self.agent)
                action_distribution = policy_network.action_distribution(obs)
                a = action_distribution.sample()

                new_obs, r, done, info = self.env.step(
                    action_idx=a, obs=obs)  # new_obs unused b/c env tracks
                if done:  # get the last env_render
                    env_char_renders.append(
                        self.env.render_as_char(self.env.grid))
                t += 1
                if t == self.max_num_scenes:
                    # Time limit exceeded -> negative reward
                    r = -1  
                    done = True


                log_probs.append(action_distribution.log_prob(a).unsqueeze(0))
                rewards.append(r)

            returns = rl_memory.tools.discount_rewards(
                rewards, self.discount_factor)
            baselines = np.zeros(returns.shape)
            advantages = returns - baselines
            if len(log_probs) != len(advantages):
                print()
            policy_network.update(log_probs, advantages)

            total_reward = np.sum(rewards)
            total_return = np.sum(returns)
            if total_reward > 1:
                print("big reward")

            self.episode_tracker.rewards.append(total_reward)
            self.episode_tracker.returns.append(total_return)
            self.episode_tracker.trajectories.append(env_char_renders)

        # return the trained model, episode rewards, and env_renders for each trajectory
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
env: Env = environment.Env(
    grid_shape=grid_shape, n_goals=n_goals, hole_pct=hole_pct)
env.create_new()
obs: Observation = environment.Observation(env, james_bond)

def train(env: Env = env, agent: Agent = james_bond, obs: Observation = obs,
          num_episodes = 20, gamma = .99, lr = 1e-3,  
          create_new_counter = 0, reset_frequency = 5):

    max_num_scenes = 3 * grid_shape[0] * grid_shape[1]

    # init model
    obs_size: torch.Size = obs.observation.size
    action_dim: int = len(env.action_space)
    h_params = VPGHyperParameters(lr = lr)
    policy = VPGNetwork(
        obs_size = obs_size, action_dim = action_dim, 
        h_params = h_params)

    # tracking important things
    training_tracker: VPGEpisodeTracker
    training_episode_rewards = []  # records the reward per episode
    episode_trajectories = []

    for episode in range(num_episodes):
        print(f"episode {episode}")

        # evaluate policy on current init conditions 5 times before switching to new init conditions
        if create_new_counter == reset_frequency:
            env.create_new()
            create_new_counter = 0
        else:
            env.reset()

        done = False
        log_probs = []  # tracks log prob of each action taken in a scene
        scene_number = 0  # track to be able to terminate episodes that drag on for too long
        scene_rewards = []

        episode_envs = []  # so you can see what the agent did in the episode

        while not done:
            episode_envs.append(env.render_as_char(env.grid))
            obs: Observation = environment.Observation(env, agent)
            action_distribution = policy.action_distribution(
                obs.flatten())
            # [torch.exp(action_dist.log_prob(i)) for i in action_dist.enumerate_support()] - see probs
            a = action_distribution.sample()
            assert a in np.arange(0, action_dim).tolist()
            log_probs.append(action_distribution.log_prob(a).unsqueeze(0))

            new_state, r, done, info = env.step(action_idx=a, state=state)  # ns is the new state observation bc of vision window

            scene_number += 1
            if scene_number > max_num_scenes:
                r = -1
                scene_rewards.append(r)
                break

            if done:
                episode_envs.append(env.render_as_char(env.grid))

            scene_rewards.append(r)
        episode_rewards = np.sum(scene_rewards)
        create_new_counter += 1
        training_tracker.rewards.append(episode_rewards)
        policy.update(scene_rewards, gamma, log_probs)

        episode_trajectories.append(episode_envs)

    # return the trained model,
    return policy_network, training_episode_rewards, episode_trajectories

def test(env: Env, agent: Agent, policy: nn.Module, num_episodes: int = 10):

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
            action_distribution = policy.action_distribution(state.observation.flatten())
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
