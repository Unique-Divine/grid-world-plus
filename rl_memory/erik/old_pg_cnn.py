import numpy as np

from rl_memory.erik.network_cnn import network
from rl_memory.models.a2c.tools import plot_episode_rewards

from rl_memory.custom_env.agents import Agent
from rl_memory.custom_env.environment import Env, Observation
from rl_memory.custom_env.representations import ImgTransforms

it = ImgTransforms()

# env hyperparams
grid_shape = (5, 5)
n_goals = 1
hole_pct = 0.2  # note that the env screws up if it find an env that happens to not have any holes

# initialize agent and environment
view_length = 3
james_bond = Agent(view_length)
env = Env(grid_shape=grid_shape, n_goals=n_goals, hole_pct=hole_pct)
env.create_new()
state = Observation(env, james_bond)  # in the future, we need to change to sqnce of obs


def pg_cnn_single_env(self, use_custom=False):
    """
    run the agent on a big environment with many holes to see if vanilla PG can solve env
    """
    if use_custom:
        env = self.custom_env
    else:
        env = self.env

    # init new env and get initial state
    env.create_new()
    state = Observation(env, self.james_bond)

    # init actor network
    action_dim = len(env.action_space)
    state_size = state.size()
    actor = network(state_size, action_dim, self.lr)

    for episode in range(self.num_episodes):

        env.reset()

        log_probs = []
        rewards = []

        # visualize agent movement during episode
        env_renders = []

        t = 0
        done = False

        while not done:

            env_renders.append(env.render_as_char(env.grid))

            state = Observation(env, self.james_bond)
            dist = actor.action_dist(state)
            a = dist.sample()
            assert a in np.arange(0, action_dim).tolist()

            ns, r, done, info = env.step(action_idx=a, obs=state)  # ns unused b/c env tracks

            t += 1
            if t == self.max_num_scenes:
                r = -1  # if exceeds alotted time w/o finding the goal give it negative reward
                done = True

            if done:  # get the last env_render
                env_renders.append(env.render_as_char(env.grid))

            log_probs.append(dist.log_prob(a).unsqueeze(0))
            rewards.append(r)
            self.dists.append(dist)

        returns = discount_reward(rewards, self.gamma)
        baselines = np.zeros(returns.shape)
        advantages = returns - baselines
        if len(log_probs) != len(advantages):
            print()
        actor.update(log_probs, advantages)

        total_reward = np.sum(rewards)
        total_return = np.sum(returns)
        if abs(total_reward) > 1:
            print()

        self.episode_rewards.append(total_reward)
        self.episode_returns.append(total_return)
        self.episode_trajectories.append(env_renders)

    # return the trained model, episode rewards, and env_renders for each trajectory
    return actor

num_episodes = 300000
def train(
        env=env, james_bond=james_bond, state=state,
        num_episodes=num_episodes, gamma=.99, lr=1e-3,
        create_new_counter=0, reset_frequency=10
):
    """ in future, change env once it seems to learn that environment
    that way we test whether it can transfer learn well
    the specific env shouldn't matter too much because all it sees is """

    max_num_scenes = 3 * grid_shape[0] * grid_shape[1]

    # init model
    state_size = state.size()
    action_dim = len(env.action_space)
    actor = network(state_size, action_dim, lr)

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
            state = Observation(env, james_bond)
            dist = actor.action_dist(state)
            # [torch.exp(action_dist.log_prob(i)) for i in action_dist.enumerate_support()] - see probs
            a = dist.sample()
            assert a in np.arange(0, action_dim).tolist()
            log_probs.append(dist.log_prob(a).unsqueeze(0))

            ns, r, d, info = env.step(action_idx=a, obs=state)  # ns is the new state observation bc of vision window

            scene_number += 1
            if scene_number > max_num_scenes:
                r = -1
                scene_rewards.append(r)
                break

            if d:
                episode_envs.append(env.render_as_char(env.grid))

            scene_rewards.append(r)

        create_new_counter += 1
        training_episode_rewards.append(np.sum(scene_rewards))

        baseline = np.mean(scene_rewards)
        actor.update(scene_rewards, gamma, baseline, log_probs)

        episode_trajectories.append(episode_envs)

    # return the trained model,
    return actor, training_episode_rewards, episode_trajectories


actor, training_episode_rewards, ept = train()
avg_rewards = [np.mean(training_episode_rewards[:i]) for i in range(1, num_episodes, 50)]
plot_episode_rewards(avg_rewards, "training rewards", 5)

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
            state = Observation(env, james_bond)
            dist = policy.action_dist(state)
            a = dist.sample()

            ns, r, d, info = env.step(action_idx=a, obs=state)  # ns is the new state observation bc of vision window
            reward_sum += r

            scene_number += 1
            if scene_number > max_num_scenes:
                break

        episode_trajectories.append(episode_envs)
        training_episode_rewards.append(reward_sum)

    return training_episode_rewards, episode_trajectories


test_episode_rewards, episode_trajectories = test()
avg_rewards = [np.mean(test_episode_rewards[:i]) for i in range(1, num_episodes, 1)]
plot_episode_rewards(test_episode_rewards, "test rewards", reset_frequency=5)
