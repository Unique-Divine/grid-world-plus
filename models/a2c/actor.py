import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from models.a2c.tools import discounted_reward

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

    def update(self, scene_rewards, gamma, log_probs):
        # train
        returns = discounted_reward(scene_rewards, gamma)
        returns = torch.FloatTensor(returns)
        log_probs = torch.cat(log_probs)
        assert log_probs.requires_grad
        assert not returns.requires_grad

        loss = - torch.mean(returns * log_probs)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
