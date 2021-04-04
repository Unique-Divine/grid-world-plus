import torch
import numpy as np
import collections

from models.a2c.actor import Actor
from models.a2c.tools import discounted_reward, plot_episode_rewards

from custom_env.agent import Agent
from custom_env.environment import Env, State, PathMaker


# env hyperparams
grid_shape = (3, 3)
n_goals = 2
hold_pct = 0


"""
action 0 is top_right then it goes counter-clockwise

sometimes agent spawns on top of the goal and there is no done trigger
wait there is just no goal at all actually
"""
# initialize environment and agent
env = Env(grid_shape=grid_shape, n_goals=n_goals, hole_pct=hold_pct)
james_bond = Agent(4)
env.create()
state = State(env, james_bond)

# learning hyperparams
num_episodes = 100  # number of episodes to run
gamma = .99  # discount
lr = 1e-3
max_num_scenes = 5 * grid_shape[0] * grid_shape[1]

# init model
state_dim = state.observation.size
action_dim = len(env.action_space)
policy = Actor(state_dim, action_dim, lr)

# tracking important things
episode_rewards = []  # records the reward per episode

for episode in range(num_episodes):
    print(f"episode {episode}")
    env.reset()  # generates a random environment !! needs to have argument rand=Bool
    d = False
    reward_sum = 0  # track rewards in an episode, append to episode_rewards []
    scene_number = 0  # track to be able to terminate episodes that drag on for too long
    scene_rewards = []  # tracks reward in each scene
    log_probs = []  # tracks log prob of each action taken in a scene

    episode_envs = []


    while not d:
        state = State(env, james_bond)
        action_dist = policy.action_dist(state.observation.flatten())
        # [torch.exp(action_dist.log_prob(i)) for i in action_dist.enumerate_support()]
        a = action_dist.sample()
        assert a in np.arange(0, action_dim).tolist()
        log_probs.append(action_dist.log_prob(a).unsqueeze(0))

        ns, r, d, info = env.step(action_idx=a, state=state)  # ns is the new state observation bc of vision window

        episode_envs.append(env.render_as_char(env.grid))
        scene_rewards.append(r)
        reward_sum += r

        scene_number += 1
        if scene_number > max_num_scenes:
            break

    # train
    returns = discounted_reward(scene_rewards, gamma)
    returns = torch.FloatTensor(returns)
    log_probs = torch.cat(log_probs)
    assert log_probs.requires_grad
    assert not returns.requires_grad

    loss = - torch.mean(returns * log_probs)
    policy.optimizer.zero_grad()
    loss.backward()
    policy.optimizer.step()

    episode_rewards.append(reward_sum)

plot_episode_rewards(episode_rewards, "episode_rewards")
