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
State = rl_memory.State
Agent = rl_memory.Agent
from torch import Tensor
Categorical = distributions.Categorical

it = representations.ImageTransforms()

@dataclass
class VanillaPGHyperParameters:
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

class VanillaPGNetwork(nn.Module):
    """Neural network for vanilla policy gradient. Used in the first experiment
    
    Args:
        action_dim (int): The dimension of the action space, i.e. 
            len(action_space). 
        
    """
    def __init__(self, state_size: torch.Size, action_dim: int, 
                 h_params: VanillaPGHyperParameters):
        super().__init__()
        self.batch_size: int = h_params.batch_size
        num_filters: int = h_params.num_filters
        filter_size: int = h_params.filter_size
        hidden_dim: int = h_params.hidden_dim

        # Model Architecture
        self.convnet_encoder = self._get_convnet_encoder(
            num_filters=num_filters, filter_size=filter_size)

        lin_dim = num_filters * (state_size[0] - filter_size) ** 2

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

        input: state (TODO adjust for when state is sqnce of observations??)
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

class VanillaPGExperiment:
    """formerly pg_cnn.py"""
    def __init__(self, grid_shape, num_goals, hole_pct, view_length):

        # env
        self.grid_shape = grid_shape
        self.num_goals = num_goals
        self.hole_pct = hole_pct
        self.env = environment.Env(
            grid_shape = grid_shape, n_goals = num_goals, hole_pct = hole_pct)
        self.custom_env = None

        # agent
        self.agent = agents.Agent(view_length)

        # learning hyperparams
        self.num_episodes = 10000
        self.disc_factor = .99 # gamma
        self.lr = 1e-3
        self.reset_frequency = 10
        self.max_num_scenes = 3 * self.grid_shape[0] * self.grid_shape[1]

        # episode tracking
        self.episode_rewards = []
        self.episode_returns = []
        self.episode_trajectories = []
        self.distributions = []
        # [torch.exp(dist.log_prob(i)) for i in dist.enumerate_support()]

    def setup_pg_cnn_single_env(self, use_custom=False):
        if use_custom:
            env = self.custom_env
        else:
            env = self.env

        # init new env and get initial state
        env.create_new()
        state = environment.Observation(env, self.agent)

        # init actor network
        action_dim: int = len(env.action_space)
        state_size: torch.Size = state.size()
        actor = PolicyNetwork(state_size, action_dim, self.lr)
        return actor, env

    def pretrain_on_transfer_env(self, env, actor):
        for episode_num in range(self.num_episodes):

            # env.reset()  # changed from reset()
            env = self.transfer_env()

            log_probs = []
            rewards = []

            # visualize agent movement during episode
            env_char_renders = []

            t = 0
            done = False

            while not done:

                env_char_renders.append(env.render_as_char(env.grid))

                state = environment.Observation(env, self.agent)
                dist = actor.action_dist(state)
                a = dist.sample()

                ns, reward, done, info = env.step(action_idx=a, obs=state)
                # ns unused b/c env tracks

                t += 1

                if t == self.max_num_scenes:
                    # Agent took too long to solve -> negative reward
                    reward = -1  
                    done = True

                if done:  # get the last env_render
                    env_char_renders.append(env.render_as_char(env.grid))

                log_probs.append(dist.log_prob(a).unsqueeze(0))
                rewards.append(reward)
                self.distributions.append(dist)

            returns = rl_memory.tools.discount_rewards(
                rewards = rewards, discount_factor = self.disc_factor)
            baselines = np.zeros(returns.shape)
            advantages = returns - baselines
            if len(log_probs) != len(advantages):
                print("mismatch")
            actor.update(log_probs, advantages)

            total_reward = np.sum(rewards)
            total_return = np.sum(returns)
            if total_reward > 1:
                print("big total")

            self.episode_rewards.append(total_reward)
            self.episode_returns.append(total_return)
            self.episode_trajectories.append(env_char_renders)

        # return the (trained model, episode rewards, env_char_renders) 
        # for each trajectory
        return actor

    @staticmethod
    def transfer_env() -> Env:
        """Returns:
            (Env): A small, 3 by 3 environment with one goal and one hole.
        """
        tenv: Env = environment.Env(grid_shape=(3, 3), n_goals=1, hole_pct=.1)
        tenv.set_agent_goal()
        tenv.set_holes()
        return tenv

    def pg_cnn_transfer(self):
        """
        run the agent on a big environment with many holes to see if 
        vanilla PG can solve env
        """

        # init new env and get initial state
        self.env.create_new()
        state = environment.Observation(self.env, self.agent)
        initial_grid = self.env.grid

        self.custom_env = self.transfer_env()
        actor, env = self.setup_pg_cnn_single_env(use_custom=True)
        actor = self.pretrain_on_transfer_env(env=env, actor=actor)
        avg_scene_len = np.mean(
            [len(traj) for traj in self.episode_trajectories[-500:]])
        while avg_scene_len > 3.3:
            actor = self.pretrain_on_transfer_env(env=env, actor=actor)
            avg_scene_len = np.mean(
                [len(traj) for traj in self.episode_trajectories[-500:]])
            print(avg_scene_len)

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

                state = environment.Observation(self.env, self.agent)
                dist = actor.action_dist(state)
                a = dist.sample()

                new_state, r, done, info = self.env.step(
                    action_idx=a, obs=state)  # new_state unused b/c env tracks
                if done:  # get the last env_render
                    env_char_renders.append(
                        self.env.render_as_char(self.env.grid))
                t += 1
                if t == self.max_num_scenes:
                    # Time limit exceeded -> negative reward
                    r = -1  
                    done = True


                log_probs.append(dist.log_prob(a).unsqueeze(0))
                rewards.append(r)

            returns = rl_memory.tools.discount_rewards(
                rewards, self.disc_factor)
            baselines = np.zeros(returns.shape)
            advantages = returns - baselines
            if len(log_probs) != len(advantages):
                print()
            actor.update(log_probs, advantages)

            total_reward = np.sum(rewards)
            total_return = np.sum(returns)
            if total_reward > 1:
                print("big reward")

            self.episode_rewards.append(total_reward)
            self.episode_returns.append(total_return)
            self.episode_trajectories.append(env_char_renders)

        # return the trained model, episode rewards, and env_renders for each trajectory
        return actor

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
state: State = environment.State(env, james_bond)

def train(env: Env = env, agent: Agent = james_bond, state: State = state,
          num_episodes = 20, gamma = .99, lr = 1e-3,  
          create_new_counter = 0, reset_frequency = 5):

    max_num_scenes = 3 * grid_shape[0] * grid_shape[1]

    # init model
    state_size: torch.Size = state.observation.size
    action_dim = len(env.action_space)
    h_params = VanillaPGHyperParameters(lr = lr)
    policy = VanillaPGNetwork(
        state_size = state_size, action_dim = action_dim, 
        h_params = h_params)

    # tracking important things
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

        d = False
        log_probs = []  # tracks log prob of each action taken in a scene
        scene_number = 0  # track to be able to terminate episodes that drag on for too long
        scene_rewards = []

        episode_envs = []  # so you can see what the agent did in the episode

        while not d:
            episode_envs.append(env.render_as_char(env.grid))
            state = environment.State(env, agent)
            action_distribution = policy.action_distribution(
                state.observation.flatten())
            # [torch.exp(action_dist.log_prob(i)) for i in action_dist.enumerate_support()] - see probs
            a = action_distribution.sample()
            assert a in np.arange(0, action_dim).tolist()
            log_probs.append(action_distribution.log_prob(a).unsqueeze(0))

            new_state, r, d, info = env.step(action_idx=a, state=state)  # ns is the new state observation bc of vision window

            scene_number += 1
            if scene_number > max_num_scenes:
                r = -1
                scene_rewards.append(r)
                break

            if d:
                episode_envs.append(env.render_as_char(env.grid))

            scene_rewards.append(r)

        create_new_counter += 1
        training_episode_rewards.append(np.sum(scene_rewards))
        policy.update(scene_rewards, gamma, log_probs)

        episode_trajectories.append(episode_envs)

    # return the trained model,
    return actor, training_episode_rewards, episode_trajectories

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
            action_dist = policy.action_dist(state.observation.flatten())
            a = action_dist.sample()

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
    actor, training_episode_rewards, episode_trajectories['train'] = train()
    rl_memory.tools.plot_episode_rewards(training_episode_rewards, "training rewards", 5)
    test_env = environment.Env(
        grid_shape=grid_shape, n_goals=n_goals, hole_pct=hole_pct)

    test_episode_rewards, episode_trajectories['test'] = test(
        env = test_env, agent = james_bond, policy = actor, num_episodes = 10)
    rl_memory.tools.plot_episode_rewards(test_episode_rewards, "test rewards", reset_frequency=5)
