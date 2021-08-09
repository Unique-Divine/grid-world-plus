import numpy as np

from rl_memory.erik.network_cnn import network
from tools import discount_reward

from rl_memory.custom_env import agents
from rl_memory.custom_env import environment
from rl_memory.custom_env import representations

it = representations.ImageTransforms()


class RLAlgorithms:
    def __init__(self, grid_shape, num_goals, hole_pct, view_length):

        # env
        self.grid_shape = grid_shape
        self.num_goals = num_goals
        self.hole_pct = hole_pct
        self.env = environment.Env(
            grid_shape = grid_shape, n_goals = num_goals, hole_pct = hole_pct)
        self.custom_env = None

        # agent
        self.view_length = view_length
        self.james_bond = agents.Agent(view_length)

        # learning hyperparams
        self.num_episodes = 10000
        self.gamma = .99
        self.lr = 1e-3
        self.reset_frequency = 10
        self.max_num_scenes = 3 * self.grid_shape[0] * self.grid_shape[1]

        # episode tracking
        self.episode_rewards = []
        self.episode_returns = []
        self.episode_trajectories = []
        self.dists = []  # [torch.exp(dist.log_prob(i)) for i in dist.enumerate_support()]

    def setup_pg_cnn_single_env(self, use_custom=False):
        if use_custom:
            env = self.custom_env
        else:
            env = self.env

        # init new env and get initial state
        env.create_new()
        state = environment.Observation(env, self.james_bond)

        # init actor network
        action_dim = len(env.action_space)
        state_size = state.size()
        actor = network(state_size, action_dim, self.lr)
        return actor, env

    def train_pg_cnn_single_env(self, env, actor):
        for episode in range(self.num_episodes):

            # env.reset()  # changed from reset()
            env = self.transfer_env()

            log_probs = []
            rewards = []

            # visualize agent movement during episode
            env_renders = []

            t = 0
            done = False

            while not done:

                env_renders.append(env.render_as_char(env.grid))

                state = environment.Observation(env, self.james_bond)
                dist = actor.action_dist(state)
                a = dist.sample()

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
                print("mismatch")
            actor.update(log_probs, advantages)

            total_reward = np.sum(rewards)
            total_return = np.sum(returns)
            if total_reward > 1:
                print("big total")

            self.episode_rewards.append(total_reward)
            self.episode_returns.append(total_return)
            self.episode_trajectories.append(env_renders)

        # return the trained model, episode rewards, and env_renders for each trajectory
        return actor

    @staticmethod
    def transfer_env():
        tenv = environment.Env(grid_shape=(3, 3), n_goals=1, hole_pct=.1)
        tenv.set_agent_goal()
        tenv.set_holes()
        return tenv

    def pg_cnn_transfer(self):
        """
        run the agent on a big environment with many holes to see if vanilla PG can solve env
        """

        # init new env and get initial state
        self.env.create_new()
        state = environment.Observation(self.env, self.james_bond)
        initial_grid = self.env.grid

        self.custom_env = self.transfer_env()
        actor, env = self.setup_pg_cnn_single_env(use_custom=True)
        actor = self.train_pg_cnn_single_env(env=env, actor=actor)
        avg_scene_len = np.mean([len(traj) for traj in self.episode_trajectories[-500:]])
        while avg_scene_len > 3.3:
            actor = self.train_pg_cnn_single_env(env=env, actor=actor)
            avg_scene_len = np.mean([len(traj) for traj in self.episode_trajectories[-500:]])
            print(avg_scene_len)

        for episode in range(self.num_episodes):

            self.env.reset()

            log_probs = []
            rewards = []

            # visualize agent movement during episode
            env_renders = []

            t = 0
            done = False

            while not done:

                env_renders.append(self.env.render_as_char(self.env.grid))

                state = environment.Observation(self.env, self.james_bond)
                dist = actor.action_dist(state)
                a = dist.sample()

                ns, r, done, info = self.env.step(action_idx=a, obs=state)  # ns unused b/c env tracks

                t += 1
                if t == self.max_num_scenes:
                    r = -1  # if exceeds alotted time w/o finding the goal give it negative reward
                    done = True

                if done:  # get the last env_render
                    env_renders.append(self.env.render_as_char(self.env.grid))

                log_probs.append(dist.log_prob(a).unsqueeze(0))
                rewards.append(r)

            returns = discount_reward(rewards, self.gamma)
            baselines = np.zeros(returns.shape)
            advantages = returns - baselines
            if len(log_probs) != len(advantages):
                print()
            actor.update(log_probs, advantages)

            total_reward = np.sum(rewards)
            total_return = np.sum(returns)
            if total_reward > 1:
                print("big reward")

            self.episode_rewards.append(total_reward)
            self.episode_returns.append(total_return)
            self.episode_trajectories.append(env_renders)

        # return the trained model, episode rewards, and env_renders for each trajectory
        return actor