import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from grid_world_plus.models.a2c.tools import discounted_reward
from grid_world_plus.memory import Memory


class Actor(object):
    def __init__(self, state_dim, action_dim, lr, hidden_dim=50):
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=2, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Linear(3*state_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, action_dim)
        )

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.memory = Memory()

    def action_dist(self, states):
        """
        input: state
        ouput: softmax(nn valuation of each action)
        """
        states = torch.FloatTensor(states)
        dist = Categorical(F.softmax(self.model(states), dim=-1))
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
