import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from rl_memory.models.a2c.tools import discounted_reward
from rl_memory.memory import Memory
from rl_memory.custom_env.representations import ImageTransforms
import numpy as np

from torch.nn import Conv2d, LeakyReLU, ReLU, MaxPool2d, BatchNorm2d, Linear

it = ImageTransforms()


# nn.module is the base neural network class
class network(nn.Module):
    def __init__(self, state_size, action_dim, lr, hidden_dim=5, batch_size=1):
        super(network, self).__init__()
        self.batch_size = batch_size
        num_filters = 16
        filter_size = 2

        self.conv1 = Conv2d(in_channels=4, out_channels=num_filters, kernel_size=filter_size, stride=1)
        self.bn1 = BatchNorm2d(num_filters)
        self.act1 = LeakyReLU()
        self.pool = MaxPool2d(kernel_size=filter_size, stride=1)

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
