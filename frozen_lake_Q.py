from tools import discount_rate, lr, decision

import numpy as np
import gym

class FrozenLakeQ():
    def __init__(self, env_seed: int = 0 ) -> None:

        self.env = gym.make('FrozenLake-v0')
        self.env.seed(env_seed)

        self.reward_episode = 0
        self.reward_list = []
        self.moving_average = 0
        self.num_episodes = 1000  # init 1000
        self.max_num_scenes = 100  # init 100

        self.Q = np.zeros([env.observation_space.n, env.action_space.n])
        self.n = np.ones([env.observation_space.n, env.action_space.n])  # record  number of visits to each state

    def train(self):
        num_episodes = self.num_episodes
        max_num_scenes = self.max_num_scenes

        # Start of training
        # we are training the Q-values

        episode_rewards = []
        episode_trajectories = []

        for episode in range(num_episodes):
            s = self.env.reset()  # reset the environment at the beginning of an episode
            done = False  # done := env returns done i.e. terminal state reached

            alpha = lr(episode, num_episodes)
            gamma = discount_rate(episode, num_episodes)
            scene_rewards = []
            trajectory = []

            for scene in range(max_num_scenes):

                a = decision(env, self.Q[s, :], episode, "epsilon greedy")

                # get new state, reward, done
                # next_state := next state (integer)
                # reward := reward
                # done := boolean true when terminal state is reached
                # info := stuff about the environment
                next_state, reward, done, info = env.step(a)

                scene_rewards.append(reward)
                trajectory.append(next_state)

                # penalize holes
                if done and next_state != 15:
                    reward = - 1

                # update Q
                delta = reward + gamma * np.max(self.Q[next_state, :]) - self.Q[s, a]
                Q_update = self.Q[s, a] + alpha * delta
                self.Q[s, a] = Q_update

                # break if reached terminal state
                if done:
                    break
                else:
                    s = next_state

    def test(self, 
        episode_count: int = 1000,
        episode_max_length: int = 100):

        scores = []
        self.env.reset()
        for i in range(episode_count):
            s = env.reset()
            done: bool = False
            for t in range(episode_max_length):
                action = np.argmax(self.Q[s, :])
                s, episode_reward, done, _ = env.step(action)
                if done:
                    break
            scores.append(episode_reward)
            print("evaluation reward", episode_reward)

        return scores