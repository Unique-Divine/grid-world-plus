import torch
import numpy as np
import gym
import sys
import os

from models.DQN.q_network import Q
from models.DQN.tools import epsilon, run_target_update, plot_episode_rewards, ReplayBuffer
from custom_env import environment

breakpoint()

# buffer hyperparameters
batchsize = 200  # batchsize for buffer sampling
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

# initialize environment
envname = "CartPole-v0"
env = gym.make(envname)
"""
obssize
Num	Observation	           Min	    Max
0	Cart Position	      -2.4	    2.4
1	Cart Velocity	      -Inf	    Inf
2	Pole Angle	          -41.8°	41.8°
3	Pole Velocity At Tip  -Inf	    Inf
"""

# initialize the principal and the target Q nets
state_dim = env.observation_space.low.size
action_dim = env.action_space.n
lr = 1e-3
QP = Q(state_dim, action_dim, lr)
QT = Q(state_dim, action_dim, lr)

for episode in range(num_episodes):
    s = env.reset()
    d = False
    reward_sum = 0

    while not d:
        q_vals = QP.predict_state_value(s)

        # Execute actions
        if np.random.rand() < epsilon(episode, num_episodes):
            a = torch.tensor(env.action_space.sample())
        else:
            a = torch.argmax(q_vals)
        assert a in [0, 1]

        ns, r, d, _ = env.step(int(a))
        d_ = 1 if d else 0
        e = (s, a, r, d_, ns)  # memory tuple
        buffer.append(e)  # buffer is the experience replay
        while buffer.number > buffer_maxlength:
            buffer.pop()

        # Training theta by gradients

        if episode % episodes_til_buffer_sample == 0 and buffer.number > batchsize:
            batch = buffer.sample(batchsize)  # list of memory tuples (current_state, action, r, done_, new_state)
            batch = list(map(list, zip(*batch)))
            states = batch[0]
            actions = batch[1]
            rewards = batch[2]
            dones = batch[3]
            next_states = batch[4]

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
        s = ns

    list_of_episode_rewards.append(reward_sum)

plot_episode_rewards(list_of_episode_rewards, "episode_rewards")
