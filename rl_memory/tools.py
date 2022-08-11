import torch
import numpy as np
import matplotlib.pyplot as plt

from typing import Iterable, List, Optional
Array = np.ndarray
Tensor = torch.Tensor

def discount_rewards(rewards: List[float], discount_factor: float):
    """Computes discounted rewards.

    Args:
        rewards (List[float]): A list of reward values from a full trajectory.
        discount_factor (float): [description] Typically denoted by gamma.

    Returns:
        discounted_rewards (Array): [description]
    """

    discounted_rewards = np.zeros(len(rewards))
    discounted_rewards[-1] = rewards[-1]
    for i in range(len(rewards)-2, -1, -1):
        discounted_rewards[i] = (discount_factor * discounted_rewards[i+1]
                                 + rewards[i])
    return discounted_rewards

def epsilon(episode_idx: int, 
            decay_type: str = "exponential",
            num_episodes: Optional[int] = None) -> float:
    """Computes epsilon, which is responsible for tuning how much the agent 
    will "explore". 
    
    Here, we assume the episode number correlates with how much has been 
    learned. Epsilon decays as the current episode gets higher because we want 
    the agent to explore more in earlier episodes (when it is less likely to 
    learn predictive features) and less in later episodes (when the agent has 
    learned relevant features).

    Args:
        episode_idx (int): The current episode number.
        decay_type (str): Determines the type of epsilon decay.
            Defaults to "exponential". 
            Options include ["linear", exponential",]
        num_episodes(Optional[int]): The total number of episodes.
    
    Returns:
        epsilon (float): A value between 0 and 1 that dictates how often a 
            random action will be taken. Epsilon is the probability of taking 
            a random action. Thus, an 'espilon' of 0 corresponds to a purely 
            greedy method where the optimal (according to the policy) action is 
            taken every time. 
    """
    epsilon: float
    decay_types: List[str] = ["linear", "exponential"]
    if not isinstance(decay_type, str):
        raise TypeError(
            f"'decay_type' must be a string, not '{type(decay_type)}'.")
    if decay_type not in decay_types:
        raise ValueError(f"Invalid 'decay_type'. Options include {decay_types}")
    
    if decay_type == "linear":
        assert num_episodes is not None
        assert episode_idx <= num_episodes
        epsilon = 1 - (episode_idx/num_episodes)
    if decay_type == "exponential":
        epsilon = .5 * .9**episode_idx
    return epsilon


def plot_episode_rewards(episode_rewards: List[float], title: str):
    """ Plot the reward curve and histogram of results over time.
    
    Formerly a part of rl_memory.models.a2c.tools
    """
    # Update the window after each episode
    x = range(len(episode_rewards))

    # Define the figure
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12,5))
    fig.suptitle(title)
    ax[0].plot(episode_rewards, label='score per run')
    ax[0].axhline(1, c='red', ls='--', label='goal')
    ax[0].set(xlabel = 'Episodes', ylabel = 'Reward')
    ax[0].legend()

    # xcoords = [i for i in x if i%reset_frequency==0]
    # for xc in xcoords:
    #     plt.axvline(x=xc, color='k', linestyle='--')

    # Calculate the trend
    try:
        z = np.polyfit(x, episode_rewards, 1)
        p = np.poly1d(z)
        ax[0].plot(x, p(x), "--", label='trend')
    except:
        print('')



    # Plot the histogram of results
    ax[1].hist(episode_rewards[-50:])
    ax[1].axvline(1, c='red', label='goal')
    ax[1].set_xlabel('Scores per Last 50 Episodes')
    ax[1].set_ylabel('Frequency')
    ax[1].legend()
    plt.show()
