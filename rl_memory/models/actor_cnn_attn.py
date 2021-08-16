import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from rl_memory.models.a2c.tools import discounted_reward
from rl_memory.memory import Memory
from rl_memory.rlm_env.representations import ImgTransforms
import numpy as np

from torch.nn import Conv2d, LeakyReLU, ReLU, MaxPool2d, BatchNorm2d, Linear

it = ImgTransforms()

# nn.module is the base neural network class
class Actor(nn.Module):
    def __init__(self, state_size, action_dim, lr, hidden_dim=5, batch_size=1):
        super(Actor, self).__init__()
        self.batch_size = batch_size
        num_filters = 16
        filter_size = 2

        self.conv1 = Conv2d(in_channels=4, out_channels=num_filters, kernel_size=filter_size, stride=1)
        self.bn1 = BatchNorm2d(num_filters)
        self.act1 = LeakyReLU()
        self.pool = MaxPool2d(kernel_size=filter_size, stride=1)

        self.lin = torch.nn.Sequential(
            torch.nn.Linear(16*3*3, hidden_dim),
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

    def update(self, scene_rewards, gamma, state_reps, log_probs):
        returns = discounted_reward(scene_rewards, gamma)  # find the empirical value of each state in the episode
        baselines = []  # baseline is empirical value of most similar memorized state

        for sr, dr in state_reps, returns:
            baselines.append(self.memory.val_of_similar(sr))  # query memory for baseline
            self.memory.memorize(sr, dr)  # memorize the episode you just observed

        # update with advantage policy gradient theorem
        advantages = torch.FloatTensor(returns-baselines)
        log_probs = torch.cat(log_probs)
        assert log_probs.requires_grad
        assert not advantages.requires_grad

        loss = - torch.mean(advantages * log_probs)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
