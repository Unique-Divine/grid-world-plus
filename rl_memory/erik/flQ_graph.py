from tools import discount_rate, lr, decision
import gym
import numpy as np

import networkx as nx
import matplotlib.pyplot as pp

import random

def phi(s):
    return s


LAMBDA = .5

E = 10
D = []  # replay buffer
G = nx.DiGraph()  # associative memory graph
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
            # likely this should be a for loop with many samples from D
            memory = random.choice(D)
            s = memory[0]
            a = memory[1]
            r = memory[2]
            sp = memory[3]

            nodes = dict(G.nodes.data())
            Qg = nodes[s]['value']

            Q[s, a] = r + np.max(Q[sp, :]) - Q[s, a] + LAMBDA*(Qg - Q[s, a])

    # add states from episode to memory
    # record the sequential ID step
    # update the max value of the state
    for t in range(Te-1, -1, -1):
        memory = D[t]
        s = memory[0]
        a = memory[1]
        r = memory[2]
        sp = memory[3]

        if t == Te-1:
            R = r
        else:
            R = r + gamma*D[t+1][2]

        # assume deterministic env, and s to sp possible through exactly 1 action
        nodes = dict(G.nodes())
        edges = dict(G.edges())
        G.add_node(s)  # if s is already a node, there is no change

        # each node in the graph keeps the earliest time it was reached in a trajectory
        # in alg 1, they do not address this
        #
        try:
            current_time = nodes[s]['time']
            if t < current_time:
                nodes[s]['time'] = t
        except:
            nodes[s]['time'] = t

        try:
            current_value = nodes[s]['value']
            nodes[s]['value'] = np.max(current_value, R)
        except:
            nodes[s]['value'] = R

        # in the stochastic env, (s,sp) could be in edges, but with a different action i.e. diff weight
        # in this case we need to check if (s, sp, weight) in edges
        if (s, sp) not in edges:
            G.add_edge(s, sp, weight=action)

    if e % K == 0:
        pass
        # run algorithm 1














