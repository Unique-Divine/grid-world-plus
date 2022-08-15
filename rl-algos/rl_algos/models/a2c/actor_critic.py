import gym
import numpy as np
from grid_world_plus.erik.network_cnn import network
from critic import Critic
from tools import discounted_reward, plot_episode_rewards

# initialize environment
envname = "CartPole-v0"  # environment name
env = gym.make(envname)
state_dim = env.observation_space.low.size
action_dim = env.action_space.n

# init networks
actor = network(state_dim, action_dim, 1e-3)  # policy initialization: policy evaluated on gradewise
critic = Critic(state_dim, 1e-3)  # critic initialization

# hyperparams
num_episodes = 1000  # total num of policy evaluations and updates to perform
gamma = .99  # discount

# record training reward for algorithm evaluation
rrecord = []

for ep in range(num_episodes):

    states = []
    actions = []
    rewards = []

    s = env.reset()  # initial state
    done = False

    while not done:
        # sample action from pi and act in the environment
        action_dist = actor.compute_prob(s)
        a = np.random.choice(env.action_space.n, p=action_dist.flatten(), size=1).item()
        ns, r, done, _ = env.step(a)

        # record each memory in the trajectory
        """
        more efficient to track
        actor prob computed on line 35 and take the log prob
        and compute the critic value in the loop
        """
        states.append(s)
        actions.append(a)
        rewards.append(r)

        s = ns

    mc_vals = discounted_reward(rewards, gamma)
    critic_vals = critic.compute_values(states)
    advantage = mc_vals - critic_vals.T

    actor.train(states, actions, advantage)
    critic.train(states, mc_vals)

    # record the reward for tracking agent's progress
    rrecord.append(np.sum(rewards))
    if ep % 50 == 0:
        print(np.mean(rrecord))

plot_episode_rewards(rrecord, "training rewards")
