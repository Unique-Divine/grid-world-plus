import numpy as np
from IPython.display import clear_output
import matplotlib.pyplot as plt

config = {
    "grid_shape": (5, 5),
    "num_goals": 1,
    "hole_pct": .99,
    "view_length": 2
}

config0 = {
    "grid_shape": (5, 5),
    "num_goals": 1,
    "hole_pct": .2,
    "view_length": 2
}

def epsilon(current_episode, num_episodes):
    """
    epsilon decays as the current episode gets higher because we want the agent to
    explore more in earlier episodes (when it hasn't learned anything)
    explore less in later episodes (when it has learned something)
    i.e. assume that episode number is directly related to learning
    """
    # return 1 - (current_episode/num_episodes)
    return .5 * .9**current_episode


def discount_reward(rewards, gamma):

    dr = np.zeros(len(rewards))
    dr[-1] = rewards[-1]
    for i in range(len(rewards)-2, -1, -1):
        dr[i] = gamma*dr[i+1] + rewards[i]

    return dr



def plot_episode_history(to_graph, title=""):
    """
    example input:
    (list_of_values, comparison/target/baseline value, "title_name"
    to_graph = [(avg_sharpe_hist, 1, "sharpe"),
                (avg_wealth_hist, 10, "avg wealth"),
                (avg_crra_wealth_hist, 2.3, "crra_wealth")]
    """

    clear_output(wait=True)

    # Define the figure
    f, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 7))
    f.suptitle(title)

    for i, (data, comparison, label) in enumerate(to_graph):
        ax[i].plot(data, label=label)
        ax[i].axhline(comparison, c='red', ls='--')
        ax[i].set_xlabel('Trajectories')
        ax[i].set_ylabel(f'{label}')
        ax[i].legend(loc='upper right')

        try:
            x0 = range(len(data))
            z = np.polyfit(x0, data, 1)
            p = np.poly1d(z)
            ax[i].plot(x0, p(x0), "--", label='trend')
        except:
            print('')
