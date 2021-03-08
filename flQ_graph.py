from tools import discount_rate, lr, decision
import gym
import numpy as np

import networkx as nx
"""
add node: G.add_node(node_name, value = node_value)
add edge: G.add_edge(node_from, node_to, weight = weight_value)
"""
import random
import matplotlib.pyplot as pp

def phi(s):
    return s

def update_Q(old_Q, gamma, )

E = 10

D = []  # replay buffer
G = nx.Graph()  # associative memory graph
Te = 30  # length of eth episode
K = 5  # frequency (in episodes) with which we update Q values in the graph using Alg 1
update_freq = 10  # frequency of theta update

env = gym.make('FrozenLake-v0')

Q = np.zeros([env.observation_space.n, env.action_space.n])



for e in range(E):
    state = env.reset()
    alpha = lr(e, E)
    gamma = discount_rate(e, E)
    for t in range(Te):
        action = decision(env, Q[s, :], e, E, "epsilon greedy")

        perform_step: tuple = env.step(action)
        next_state: int
        done: bool  # 'done': True if in terminal state
        next_state, reward, done, info = perform_step

        D.append((state, action, reward, next_state))

        if t % update_freq == 0:
            memory = random.choice(D)










