import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from rl_memory.models.a2c.tools import discounted_reward
from rl_memory.memory import Memory
import numpy as np


class Actor(object):
    def __init__(self, state_dim, action_dim, lr, hidden_dim=50):
        self.convolution = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=2),
        )
        # , stride=1, padding=1

        self.model = torch.nn.Sequential(
            # torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=2, stride=1, padding=1),
            # torch.nn.ReLU(),
            # torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Linear(state_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, action_dim)
        )

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.memory = Memory()

    def process_state(self, state):
        """
        state is an img representation (L, W, C) Length, Width, Channels
        state has 3 channels: R (holes) , G (goals), B (agent)

        input:
            an RGB image

        """
        one_hots = []
        for i in [0, 1, 2, 3, 7]:
            zeros = np.zeros(len(state))
            ones = state.index(i)
            zeros[ones] = 1
            one_hots.append(zeros)

        return state

    def action_dist(self, state):
        """
        input: state (TODO adjust for when state is sqnce of observations??)
        ouput: softmax(nn valuation of each action)
        """
        state = self.process_state(state)

        dist = Categorical(F.softmax(self.model(state), dim=-1))
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
