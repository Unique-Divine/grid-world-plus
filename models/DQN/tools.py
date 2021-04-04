import random
from collections import deque
from IPython.display import clear_output
import matplotlib.pyplot as plt
import numpy as np


def epsilon(current_episode, num_episodes):
    """
    epsilon decays as the current episode gets higher because we want the agent to
    explore more in earlier episodes (when it hasn't learned anything)
    explore less in later episodes (when it has learned something)
    i.e. assume that episode number is directly related to learning
    """
    # return 1 - (current_episode/num_episodes)
    return .5 * .9**current_episode


def run_target_update(Qprincipal, Qtarget):
    for v, v_ in zip(Qprincipal.model.parameters(), Qtarget.model.parameters()):
        v_.data.copy_(v.data)


def plot_episode_rewards(values, title=''):
    """ Plot the reward curve and histogram of results over time."""
    # Update the window after each episode
    clear_output(wait=True)

    # Define the figure
    f, ax = plt.subplots(nrows=1, ncols=2, figsize=(12,5))
    f.suptitle(title)
    ax[0].plot(values, label='score per run')
    ax[0].axhline(195, c='red', ls='--', label='goal')
    ax[0].set_xlabel('Episodes')
    ax[0].set_ylabel('Reward')
    x = range(len(values))
    ax[0].legend()
    # Calculate the trend
    try:
        z = np.polyfit(x, values, 1)
        p = np.poly1d(z)
        ax[0].plot(x, p(x), "--", label='trend')
    except:
        print('')

    # Plot the histogram of results
    ax[1].hist(values[-50:])
    ax[1].axvline(195, c='red', label='goal')
    ax[1].set_xlabel('Scores per Last 50 Episodes')
    ax[1].set_ylabel('Frequency')
    ax[1].legend()
    plt.show()


class ReplayBuffer(object):

    def __init__(self, maxlength):
        """
        maxlength: max number of tuples to store in the buffer
        if there are more tuples than maxlength, pop out the oldest tuples
        """
        self.buffer = deque()
        self.number = 0
        self.maxlength = maxlength

    def append(self, experience):
        """
        this function implements appending new experience tuple
        experience: a tuple of the form (s,a,r,s^\prime)
        """
        self.buffer.append(experience)
        self.number += 1

    def pop(self):
        """
        pop out the oldest tuples if self.number > self.maxlength
        """
        while self.number > self.maxlength:
            self.buffer.popleft()
            self.number -= 1

    def sample(self, batchsize):
        """
        this function samples 'batchsize' experience tuples
        batchsize: size of the minibatch to be sampled
        return: a list of tuples of form (s,a,r,sprime)
        """
        minibatch = random.sample(self.buffer, batchsize)
        return minibatch
