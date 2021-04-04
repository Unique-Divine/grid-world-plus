import torch


class Q:
    def __init__(self, state_dim, action_dim, lr, hidden_dim=15):
        self.model = torch.nn.Sequential(
            torch.nn.Linear(state_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim*2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim*2, action_dim)
        )

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr)
        self.loss_fn = torch.nn.MSELoss()

        self.action_dim = action_dim
        self.state_dim = state_dim

    def predict_state_value(self, state):
        state = torch.FloatTensor(state)
        with torch.no_grad():
            pred = self.model(state)

        return pred

    def grad_pred(self, state):
        pred = self.model(state)

        return pred

    def train(self, states, actions, targets):
        """
        states: numpy array as input to compute loss (s)
        actions: numpy array as input to compute loss (a)
        targets: numpy array as input to compute loss (Q targets)
        """
        states = torch.FloatTensor(states)
        states.requires_grad = True
        actions = torch.LongTensor(actions)
        targets = torch.FloatTensor(targets)

        state_vals = self.grad_pred(states)
        q_vals = state_vals[range(state_vals.shape[0]), actions]

        loss = self.loss_fn(q_vals, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()