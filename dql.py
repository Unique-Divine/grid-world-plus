import torch
import numpy as np
import gym
import random
import sys
import os
import collections
Experience = collections.namedtuple(
    "Experience",
    ["obs", "a", "r", "next_obs", "d"])

from models.DQN.q_network import Q
from models.DQN.tools import epsilon, run_target_update, plot_episode_rewards, ReplayBuffer

from custom_env.agent import Agent
from custom_env.environment import Env, State, PathMaker



# buffer hyperparameters
batchsize = 2  # batchsize for buffer sampling
buffer_maxlength = 1000  # max number of tuples held by buffer
episodes_til_buffer_sample = 2
buffer = ReplayBuffer(buffer_maxlength)  # buffer holds the memories of the exp replay

# DQL hyperparameters
steps_til_target_update = 50  # time steps for target update
num_episodes = 500  # number of episodes to run
gamma = .99  # discount

# tracking important things
list_of_episode_rewards = []  # records the reward per episode
Qtarget_update_counter = 0  # count the number of steps taken before updating Qtarget

# initialize environment and agent
env = Env(grid_shape=(5, 5), n_goals=3, hole_pct=0.1)
james_bond = Agent(4)
env.create()

# initialize the principal and the target Q nets
state = State(env, james_bond)
state_dim = state.observation.size
action_dim = len(env.action_space)
lr = 1e-3
QP = Q(state_dim, action_dim, lr)
QT = Q(state_dim, action_dim, lr)

for episode in range(num_episodes):
    env.reset()
    d = False
    reward_sum = 0

    scene_number = 0
    while not d:
        state = State(env, james_bond)
        q_vals = QP.predict_state_value(state.observation.flatten())

        # Execute actions
        if np.random.rand() < epsilon(episode, num_episodes):
            a = np.random.randint(0, action_dim)
        else:
            a = torch.argmax(q_vals)
        a = int(a)
        assert a in np.arange(0, action_dim).tolist()

        """ better to not save ns, but implement the alg to look ahead """
        ns, r, d, info = env.step(action_idx=a, state=state)  # ns is the new state observation bc of vision window
        e = Experience(state.observation, a, r, ns, d)
        # e = (state.observation, a, r, ns, d)  # memory tuple
        buffer.append(e)  # buffer is the experience replay
        while buffer.number > buffer_maxlength:
            buffer.pop()

        if scene_number > 25:
            break

        # Training theta by gradients

    if episode % episodes_til_buffer_sample == 0 and buffer.number > batchsize:
        batch = buffer.sample(batchsize)  # list of memory tuples (current_state, action, r, done_, new_state)
        batch = list(map(list, zip(*batch)))
        states = [s.flatten() for s in batch[0]]
        actions = batch[1]
        rewards = batch[2]
        next_states = [ns.flatten() for ns in batch[3]]
        dones = batch[4]

        q_vals_ns = QT.predict_state_value(next_states)
        max_vals = torch.max(q_vals_ns, dim=1).values  # take the max along the columns
        targets = torch.tensor(rewards) + torch.tensor(gamma)*max_vals

        done_indices = [i for i, d in enumerate(dones) if d==1]
        for idx in done_indices:
            targets[idx] = rewards[idx]

        QP.train(states, actions, targets)

        # 5)
        if Qtarget_update_counter % steps_til_target_update == 0:
            run_target_update(QP, QT)

        # 6)
        Qtarget_update_counter += 1
        reward_sum += r
        print(f"episode {episode}")


    list_of_episode_rewards.append(reward_sum)

plot_episode_rewards(list_of_episode_rewards, "episode_rewards")
