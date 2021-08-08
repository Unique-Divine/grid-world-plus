import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions 
from rl_memory.models import a2c
import rl_memory.memory 
import numpy as np
# Type imports
from torch import Tensor
from torch.distributions.categorical import Categorical

class Actor:
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
            input: state (TODO adjust for when state is sqnce of observations??)
            ouput: softmax(nn valuation of each action)

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
        returns = a2c.tools.discounted_reward(scene_rewards, gamma)  
        baselines = []  # baseline is empirical value of most similar memorized state

        for sr, dr in state_reps, returns:
            baselines.append(self.memory.val_of_similar(sr))  # query memory for baseline
            self.memory.memorize(sr, dr)  # memorize the episode you just observed

        # update with advantage policy gradient theorem
        advantages = torch.FloatTensor(returns - baselines)
        log_probs = torch.cat(log_probs)
        assert log_probs.requires_grad
        assert not advantages.requires_grad

        loss = - torch.mean(advantages * log_probs)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
