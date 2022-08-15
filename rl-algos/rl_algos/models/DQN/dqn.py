from typing import List, Tuple

import q_network
from q_network import Q
import numpy as np
# import gym
import tools
import torch

# buffer hyperparameters
batchsize = 200  # batchsize for buffer sampling
buffer_maxlength = 1000  # max number of tuples held by buffer
episodes_til_buffer_sample = 2
buffer = tools.ReplayBuffer(buffer_maxlength)  # buffer holds the memories of the exp replay

# DQL hyperparameters
steps_til_target_update = 50  # time steps for target update
num_episodes = 500  # number of episodes to run
# initialsize = 500  # initial time steps before start training - unused
gamma = .99  # discount

# tracking important things
list_of_episode_rewards = []  # records the reward per episode
q_prime_update_counter = 0  # count the number of steps taken before updating q_prime

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
q_greedy: Q = Q(state_dim, action_dim, lr)
q_prime: Q = Q(state_dim, action_dim, lr)

for episode in range(num_episodes):
    # Initialize and reset environment.
    s = env.reset()
    d = False
    reward_sum = 0

    while not d:

        q_vals: q_network.QValue = q_greedy.predict_state_value(state=s)

        # Choose action w/ epsilon greedy approach
        if np.random.rand() < tools.epsilon(episode, num_episodes):
            a = torch.tensor(env.action_space.sample())
        else:
            a = torch.argmax(q_vals)
        assert a in [0, 1]

        ns, r, d, _ = env.step(int(a)) # Perform action in the env.
        d_ = 1 if d else 0
        experience = (s, a, r, d_, ns)  # experience/memory tuple
        # Append experience to the replay buffer.
        buffer.append(experience)  
        # Shorten buffer if it's too long.
        while buffer.number > buffer_maxlength:
            buffer.pop()

        # Training theta by gradients # train_from_buffer_sample()
        if (episode % episodes_til_buffer_sample == 0 
            and buffer.number > batchsize):

            experience_batch: List[tools.Experience] = buffer.sample(batchsize)
            experience_batch: Tuple[List[torch.Tensor], List[int], List[float], 
                                    List[bool], List[torch.Tensor]
                                    ] = list(map(list, zip(*experience_batch)))
            states, actions, rewards, dones, next_states = experience_batch

            q_vals_ns: q_network.QValue = q_prime.predict_state_value(next_states)
            max_vals = torch.max(q_vals_ns, dim=1).values  # take the max along the columns
            targets = torch.tensor(rewards) + torch.tensor(gamma)*max_vals

            done_indices = [i for i, d in enumerate(dones) if d]
            for idx in done_indices:
                targets[idx] = rewards[idx]

            q_greedy.train(states, actions, targets) # update_dqn_greedy()

        # 5)
        if q_prime_update_counter % steps_til_target_update == 0:
            tools.update_q_prime(q_greedy, q_prime)

        # 6)
        q_prime_update_counter += 1
        reward_sum += r
        s = ns

    list_of_episode_rewards.append(reward_sum)

tools.plot_episode_rewards(list_of_episode_rewards, "episode_rewards")