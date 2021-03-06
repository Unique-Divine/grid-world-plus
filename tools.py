import numpy as np

def discount_rate(ep_num, g=.95):
    return g


def lr(ep_num, num_eps):
    # 1, .99, .98, .97 ,...., 0
    return 1 - ep_num/num_eps


# returns an integer in 0,...,NUM_ACTIONS
def decision(environment, Q_of_state, ep_num, num_eps, decision_type):

    if decision_type == "epsilon greedy":
        epsilon = 1 - ep_num/num_eps

        prob = np.random.rand()
        if prob < epsilon:
            return environment.action_space.sample()
        else:
            return np.argmax(Q_of_state)