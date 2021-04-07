import torch
import numpy as np

from models.a2c.actor import Actor
from models.a2c.tools import plot_episode_rewards

from rl_memory.custom_env.agent import Agent
from rl_memory.custom_env.environment import Env, State, PathMaker

# env hyperparams
grid_shape = (3, 3)
n_goals = 1
hole_pct = 0

# initialize agent and environment
james_bond = Agent(4)
env = Env(grid_shape=grid_shape, n_goals=n_goals, hole_pct=hole_pct)
env.create_new()
state = State(env, james_bond)


def train(
        env=env, james_bond=james_bond, state=state,
        num_episodes=20, gamma=.99, lr=1e-3,
        create_new_counter=0, reset_frequency=5
):

    max_num_scenes = 5 * grid_shape[0] * grid_shape[1]

    # init model
    state_dim = state.observation.size
    action_dim = len(env.action_space)
    actor = Actor(state_dim, action_dim, lr)

    # tracking important things
    training_episode_rewards = []  # records the reward per episode
    episode_trajectories = []

    for episode in range(num_episodes):
        print(f"episode {episode}")

        # evaluate policy on current init conditions 5 times before switching to new init conditions

        if create_new_counter == reset_frequency:
            env.create_new()
            create_new_counter = 0
        else:
            env.reset()

        d = False
        log_probs = []  # tracks log prob of each action taken in a scene
        scene_number = 0  # track to be able to terminate episodes that drag on for too long
        scene_rewards = []

        episode_envs = []  # so you can see what the agent did in the episode

        while not d:
            episode_envs.append(env.render_as_char(env.grid))
            state = State(env, james_bond)
            action_dist = actor.action_dist(state.observation.flatten())
            # [torch.exp(action_dist.log_prob(i)) for i in action_dist.enumerate_support()]
            a = action_dist.sample()
            assert a in np.arange(0, action_dim).tolist()
            log_probs.append(action_dist.log_prob(a).unsqueeze(0))

            ns, r, d, info = env.step(action_idx=a, state=state)  # ns is the new state observation bc of vision window
            scene_rewards.append(r)

            scene_number += 1
            if scene_number > max_num_scenes:
                break

            if d:
                episode_envs.append(env.render_as_char(env.grid))

        create_new_counter += 1
        training_episode_rewards.append(np.sum(scene_rewards))
        actor.update(scene_rewards, gamma, log_probs)

        episode_trajectories.append(episode_envs)

    # return the trained model,
    return actor, training_episode_rewards, episode_trajectories


actor, training_episode_rewards, ept = train()
plot_episode_rewards(training_episode_rewards, "training rewards", 5)
test_env = Env(grid_shape=grid_shape, n_goals=n_goals, hole_pct=hole_pct)


def test(env=test_env, james_bond=james_bond, policy=actor, num_episodes=10):

    max_num_scenes = grid_shape[0] * grid_shape[1]
    env.create_new()

    episode_trajectories = []
    training_episode_rewards = []

    for e in range(num_episodes):

        episode_envs = []
        reward_sum = 0
        scene_number = 0
        d = False

        while not d:
            episode_envs.append(env.render_as_char(env.grid))
            state = State(env, james_bond)
            action_dist = policy.action_dist(state.observation.flatten())
            a = action_dist.sample()

            ns, r, d, info = env.step(action_idx=a, state=state)  # ns is the new state observation bc of vision window
            reward_sum += r

            scene_number += 1
            if scene_number > max_num_scenes:
                break

        episode_trajectories.append(episode_envs)
        training_episode_rewards.append(reward_sum)

    return training_episode_rewards, episode_trajectories


test_episode_rewards, episode_trajectories = test()
plot_episode_rewards(test_episode_rewards, "test rewards")
