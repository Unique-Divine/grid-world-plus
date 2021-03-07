import numpy as np

def discount_rate(episode_idx, num_episodes,  discount=.95):
    """[summary]

    Args:
        episode_idx (int): The integer encoding for the episode.
        num_episodes (int): The total number of episodes.
        discount (float, optional): Discount factor. Defaults to .95.
            https://tinyurl.com/discount-stack-exchange

    Returns:
        [type]: [description]
    """
    assert discount <= 1, "Discount must be between 0 and 1"
    assert discount >= 0, "Discount must be between 0 and 1"
    return discount


def lr(episode_idx, num_episodes):
    # 1, .99, .98, .97 ,...., 0
    lr = 1 - episode_idx/num_episodes
    assert lr <= 1, "'lr' must be between 0 and 1"
    assert lr >= 0, "'lr' must be between 0 and 1"
    return lr


# returns an integer in 0,...,NUM_ACTIONS
def decision(environment, Q_of_state, ep_num, num_eps, decision_type):

    if decision_type == "epsilon greedy":
        epsilon = 1 - ep_num/num_eps

        prob = np.random.rand()
        if prob < epsilon:
            return environment.action_space.sample()
        else:
            return np.argmax(Q_of_state)