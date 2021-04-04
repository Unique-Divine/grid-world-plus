import torch
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np


class Actor(object):
    def __init__(self, state_dim, action_dim, lr, hidden_dim=50):
        self.model = torch.nn.Sequential(
            torch.nn.Linear(state_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, action_dim),
        )

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.state_dim = state_dim
        self.action_dim = action_dim

    def action_dist(self, states):
        """
        input: state
        ouput: softmax(nn valuation of each action)
        """
        states = torch.FloatTensor(states)
        dist = Categorical(F.softmax(self.model(states), dim=-1))
        return dist

    def train(self, states, actions, advs):
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        advs = torch.FloatTensor(advs[0])

        # COMPUTE probability vector pi(s) for all s in states
        states.requires_grad = True
        dist_over_actions = self.model(states)
        prob = F.softmax(dist_over_actions, dim=-1)
        prob_selected = prob[range(len(prob)), actions]
        log_probs = torch.log(prob_selected)
        log_probs += 1e-8  # apparently needed for robustness

        loss = - torch.mean(advs * log_probs)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
