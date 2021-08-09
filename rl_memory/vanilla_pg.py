import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions
import numpy as np
import os, sys 
from rl_memory.models import a2c
import rl_memory.memory
from rl_memory.custom_env import representations 
from rl_memory import vanilla_pg
from rl_memory.models import a2c
from rl_memory.custom_env import agents 
from rl_memory.custom_env import environment
# Type imports
Env = environment.Env
State = environment.State
Agent = agents.Agent
from torch import Tensor
Categorical = distributions.Categorical

it = representations.ImageTransforms()

# nn.module is the base neural network class
class Actor(nn.Module):
    def __init__(self, state_size, action_dim, lr, hidden_dim=5, batch_size=1):
        super().__init__()
        self.batch_size = batch_size
        num_filters: int = 16
        filter_size = 2

        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=num_filters, 
                      kernel_size=filter_size, stride=1),
            nn.BatchNorm2d(num_filters), 
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=filter_size, stride=1),)

        self.fc_layer = nn.Sequential(
            nn.Linear(num_filters * 3 * 3, hidden_dim),
                nn.ReLU(),
            nn.Linear(hidden_dim, action_dim))

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.state_dim = state_size
        self.action_dim = action_dim

        self.memory = rl_memory.memory.Memory()

    def forward(self, x):
        x = x.float()
        x = self.conv_layer(x)
        x = self.fc_layer(x.flatten())
        return x

    def process_state(self, state):
        """
        in future, need to do more processing with state := many obs
        """

        state: Tensor = it.grid_to_rgb(grid = state).unsqueeze(0)

        return state

    def action_dist(self, state) ->  Categorical:
        """
        input: state (TODO adjust for when state is sqnce of observations??)
        ouput: softmax(nn valuation of each action)
        """
        state_rgb = self.process_state(state)
        x = self.forward(state_rgb)

        dist: Categorical = distributions.Categorical(F.softmax(x, dim=-1))
        return dist

    def update(self, scene_rewards, gamma, state_reps, log_probs):
        # find the empirical value of each state in the episode
        discounted_rewards = a2c.tools.discounted_reward(scene_rewards, gamma)  
        baselines = []  # baseline is empirical value of most similar memorized state

        for state_rep, reward in state_reps, discounted_rewards:
            baselines.append(self.memory.val_of_similar(state_rep))  # query memory for baseline
            self.memory.memorize(state_rep, reward)  # memorize the episode you just observed

        # update with advantage policy gradient theorem
        advantages = torch.FloatTensor(discounted_rewards - baselines)
        log_probs = torch.cat(tensors = log_probs, dim = 0)
        assert log_probs.requires_grad
        assert not advantages.requires_grad

        loss = - torch.mean(advantages * log_probs)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

# nn.module is the base neural network class
class ActorNetwork(nn.Module):
    def __init__(self, state_size, action_dim, lr, hidden_dim=5, batch_size=1):
        super().__init__()
        self.batch_size = batch_size
        num_filters = 16
        filter_size = 2

        self.conv1 = nn.Conv2d(in_channels=4, out_channels=num_filters, kernel_size=filter_size, stride=1)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.act1 = nn.LeakyReLU()
        self.pool = nn.MaxPool2d(kernel_size=filter_size, stride=1)

        lin_dim = num_filters * (state_size[0] - filter_size) ** 2

        self.lin = torch.nn.Sequential(
            torch.nn.Linear(lin_dim, hidden_dim),
            torch.nn.Dropout(.1),
            torch.nn.ReLU(),

            torch.nn.Linear(hidden_dim, action_dim)
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.state_dim = state_size
        self.action_dim = action_dim

        self.memory = Memory()

    def forward(self, x):
        x = self.conv1(x.float())
        x = self.bn1(x)
        x = self.act1(x)
        x = self.pool(x)

        x = self.lin(x.flatten())

        return x

    def process_state(self, state):
        """
        in future, need to do more processing with state := many obs
        """
        state = it.grid_to_rgby(state).unsqueeze(0)
        return state

    def action_dist(self, state):
        """
        input: state (TODO adjust for when state is sqnce of observations??)
        ouput: softmax(nn valuation of each action)
        """
        state_rgby = self.process_state(state)
        x = self.forward(state_rgby)

        dist = Categorical(F.softmax(x, dim=-1))
        return dist

    def update(self, log_probs, advantages):  # add entropy

        advantages = torch.FloatTensor(advantages)
        log_probs = torch.cat(log_probs)
        assert log_probs.requires_grad
        assert not advantages.requires_grad

        loss = - (log_probs * advantages.detach()).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

class VanillaPGActor:
    def __init__(self, 
        state_dim: int, 
        action_dim: int, 
        lr: float, 
        hidden_dim: int = 50):

        # What is self.convolution for?
        self.convolution = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=2),
            # , stride=1, padding=1
        )

        self.model: nn.Sequential = self.assemble_model(
            hidden_dim=hidden_dim, state_dim=state_dim, action_dim=action_dim,
            convolve=False)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.memory = rl_memory.memory.Memory()

    def assemble_model(self, hidden_dim: int, state_dim: int, action_dim: int, 
                       convolve: bool = False) -> nn.Sequential:
        model: nn.Sequential
        if convolve:
            model = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=3, kernel_size=2, 
                          stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim))
        else:
            model = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim))
            pass # TODO
        return model
        
    
    def process_state(self, state):
        """
        state is an img representation (L, W, C) Length, Width, Channels
        state has 3 channels: R (holes) , G (goals), B (agent)

        input:
            an RGB image

        """
        one_hots = []
        action_idxs: list[int] = [0, 1, 2, 3, 7]
        for action_idx in action_idxs:
            zeros = np.zeros(len(state))
            ones = state.index(action_idx)
            zeros[ones] = 1
            one_hots.append(zeros)

        return state

    def action_dist(self, state) -> Categorical:
        """
        Args:
            state: TODO

        # input: state (TODO adjust for when state is sqnce of observations??)
        # ouput: softmax(nn valuation of each action)

        Returns: 
            action_dist (Categorical): 
        """
        state = self.process_state(state)
        action_logits: Tensor = self.model(state)
        action_probs: Tensor = F.softmax(input = action_logits, dim=-1)
        action_dist: Categorical = torch.distributions.Categorical(probs = action_probs)
        return action_dist

    def update(self, scene_rewards, gamma, state_reps, log_probs):
        # find the empirical value of each state in the episode
        disc_rewards = a2c.tools.discounted_reward(scene_rewards, gamma)  
        baselines = []  # baseline is empirical value of most similar memorized state

        for state_rep, disc_reward in state_reps, disc_rewards:
            baselines.append(self.memory.val_of_similar(state_rep))  # query memory for baseline
            self.memory.memorize(state_rep, disc_reward)  # memorize the episode you just observed

        # update with advantage policy gradient theorem
        advantages = torch.FloatTensor(disc_rewards - baselines)
        log_probs = torch.cat(log_probs)
        assert log_probs.requires_grad
        assert not advantages.requires_grad

        loss = - torch.mean(advantages * log_probs)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

class ActorNetwork(nn.Module):
    def __init__(self, state_size, action_dim, lr, hidden_dim=5, batch_size=1):
        super().__init__()
        self.batch_size = batch_size
        num_filters = 16
        filter_size = 2

        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels = 4, out_channels = num_filters, 
                      kernel_size = filter_size, stride = 1),
            nn.BatchNorm2d(num_filters),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=filter_size, stride=1),)

        lin_dim = num_filters * (state_size[0] - filter_size) ** 2

        self.fc_layers = nn.Sequential(
            nn.Linear(lin_dim, hidden_dim),
            nn.Dropout(.1),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.state_dim = state_size
        self.action_dim = action_dim

        self.memory = rl_memory.memory.Memory()

    def forward(self, x):
        x = self.conv_layer(x.float())
        x = self.fc_layers(x.flatten())
        return x

    def action_dist(self, state) -> Categorical:
        """
        input: state (TODO adjust for when state is sqnce of observations??)
        ouput: softmax(nn valuation of each action)
        """
        state_rgb = it.grid_to_rgb(state).unsqueeze(0)
        x = self.forward(state_rgb)

        dist: Categorical = distributions.Categorical(F.softmax(x, dim=-1))
        return dist

    def update(self, log_probs, advantages):  # add entropy

        advantages = torch.FloatTensor(advantages)
        log_probs = torch.cat(log_probs)
        assert log_probs.requires_grad
        assert not advantages.requires_grad

        loss = - (log_probs * advantages.detach()).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

# env hyperparams
grid_shape = (3, 3)
n_goals = 1
hole_pct = 0

# initialize agent and environment
james_bond: Agent = agents.Agent(4)
env: Env = environment.Env(grid_shape=grid_shape, n_goals=n_goals, hole_pct=hole_pct)
env.create_new()
state: State = environment.State(env, james_bond)


def train(env: Env = env, agent: Agent = james_bond, state: State = state,
          num_episodes = 20, gamma = .99, lr = 1e-3,  
          create_new_counter = 0, reset_frequency = 5):

    max_num_scenes = 3 * grid_shape[0] * grid_shape[1]

    # init model
    state_dim = state.observation.size
    action_dim = len(env.action_space)
    actor = vanilla_pg.ActorNetwork(state_dim, action_dim, lr)

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
            action_dist = actor.action_dist(state.observation.flatten())
            # [torch.exp(action_dist.log_prob(i)) for i in action_dist.enumerate_support()] - see probs
            a = action_dist.sample()
            assert a in np.arange(0, action_dim).tolist()
            log_probs.append(action_dist.log_prob(a).unsqueeze(0))

            ns, r, d, info = env.step(action_idx=a, state=state)  # ns is the new state observation bc of vision window

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
        actor.update(scene_rewards, gamma, log_probs)

        episode_trajectories.append(episode_envs)

    # return the trained model,
    return actor, training_episode_rewards, episode_trajectories

actor, training_episode_rewards, ept = train()
a2c.tools.plot_episode_rewards(training_episode_rewards, "training rewards", 5)
test_env = environment.Env(
    grid_shape=grid_shape, n_goals=n_goals, hole_pct=hole_pct)

def test(env = test_env, agent = james_bond, policy=actor, num_episodes=10):

    max_num_scenes = grid_shape[0] * grid_shape[1]
    env.create_new()

    episode_trajectories = []
    training_episode_rewards = []

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
        training_episode_rewards.append(reward_sum)

    return training_episode_rewards, episode_trajectories


test_episode_rewards, episode_trajectories = test()
a2c.tools.plot_episode_rewards(test_episode_rewards, "test rewards", reset_frequency=5)
