import torch


class Critic(object):
    """
    input: state
    output: value
    """
    def __init__(self, state_dim, lr, hidden_dim=50):
        self.model = torch.nn.Sequential(
            torch.nn.Linear(state_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1)
        )

        # DEFINE THE OPTIMIZER
        self.loss_fn = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        # RECORD HYPER-PARAMS
        self.state_dim = state_dim

    def compute_values(self, states):
        """
        input: vector of states
        output: vector of state values
        """
        states = torch.FloatTensor(states)
        return self.model(states).cpu().data.numpy()

    def train(self, states, mc_vals):
        """
        input: states and targets from an episode
        computes MSE loss between the two and updates the nn weights
        """
        states = torch.FloatTensor(states)
        mc_vals = torch.FloatTensor(mc_vals).view(-1, 1)

        states.requires_grad = True
        mc_vals.requires_grad = False

        critic_vals = self.model(states)
        advantage = mc_vals - critic_vals
        loss = advantage.pow(2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
