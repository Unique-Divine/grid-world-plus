from typing import Tuple
import torch
from torch import optim
import numpy as np
import torch.nn as nn

QValue = torch.Tensor
"""A torch.Tensor of floats, that's length is the 'action_dim'."""

class Q:
    """
    Attributes:
        model (nn.Module)
        optimizer (Optimizer)
        state_dim (int)
        action_dim (int)
        lr (float)
    """
    def __init__(self, state_dim: int, action_dim: int, lr: float, hidden_dim=15):
        self.model = torch.nn.Sequential(
            torch.nn.Linear(state_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim*2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim*2, action_dim))

        self.optimizer: optim.Optimizer = torch.optim.Adam(
            self.model.parameters(), lr)
        self.loss_fn = torch.nn.MSELoss()

        self.action_dim = action_dim
        self.state_dim = state_dim

    def predict_state_value(self, state) -> QValue:
        """Outputs a Q-value given a state."""
        state = torch.FloatTensor(state)
        with torch.no_grad():
            pred = self.model(state)
        return pred

    def grad_pred(self, state) -> QValue:
        pred = self.model(state)
        return pred

    def train(self, states: np.ndarray, actions: np.ndarray, 
              targets: np.ndarray):
        """
        states (np.ndarray): input to compute loss (s)
        actions (np.ndarray): input to compute loss (a)
        targets (np.ndarray): nput to compute loss (Q targets). 
            Gamma * q_val_next_state + r
        """
        # Convert all input 'np.ndarray's into 'torch.Tensor's
        states = torch.FloatTensor(states)
        states.requires_grad = True
        actions = torch.LongTensor(actions)
        targets = torch.FloatTensor(targets)

        state_vals: QValue = self.grad_pred(states) # Assumed to be 1D Tensor
        q_vals = state_vals[range(state_vals.shape[0]), actions]

        loss = self.loss_fn(q_vals, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()