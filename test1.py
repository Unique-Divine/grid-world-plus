from tools import discount_rate, lr, decision

import numpy as np
import gym
env = gym.make('FrozenLake-v0')
env.seed(0)


"""
def reset_decorate(func):
    def func_wrapper():
        global reward_list
        global moving_avg
        global episode_reward
        global fixed_window
        # when I reset environment at episode end (e.g. when d==True)
        # I want to keep track of the reward from the last episode
        reward_list.append(episode_reward)
        # when number of rewards edceeds fixed window, take the average of your rewards
        # i.e. every 100 episodes, moving average becomes the average of last fixed_window rewards
        # not necessary
        if len(reward_list) >= fixed_window:
            moving_avg = np.mean(reward_list[len(reward_list) - fixed_window:len(reward_list) - 1])
        # reset episode reward
        episode_reward = 0
        return func()
    return func_wrapper


env.reset = reset_decorate(env.reset)


def step_decorate(func):
    def func_wrapper(action):
        global episode_reward
        s1, r, d, other = func(action)
        episode_reward+=r
        return(s1, r, d, other)
    return func_wrapper


env.step = step_decorate(env.step)

"""

# wrapper for accounting rewards
episode_reward = 0
reward_list = []
fixed_window = 100
moving_avg = 0

def init():
    reward_episode = 0
    reward_list = []
    moving_average = 0
    return


init()
self.NUM_EPISODES = 1000  # init 1000
self.max_episode_length = 100  # init 100


Q = np.zeros([env.observation_space.n, env.action_space.n])
n = np.ones([env.observation_space.n, env.action_space.n])  # record  number of visits to each state


def train():
    # Start of training
    # we are training the Q-values

    episode_rewards = []
    episode_trajectories = []

    for episode in range(NUM_EPISODES):
        s = env.reset()  # reset the environment at the beginning of an episode
        done = False  # done := env returns done i.e. terminal state reached

        alpha = lr(episode, NUM_EPISODES)
        gamma = discount_rate(episode, NUM_EPISODES)
        scene_rewards = []
        trajectory = []

        for scene in range(episode_max_length):

            a = decision(env, Q[s, :], episode, "epsilon greedy")

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
            delta = reward + gamma * max(Q[next_state, :]) - Q[s, a]
            Q_update = Q[s, a] + alpha * delta
            Q[s, a] = Q_update

            # break if reached terminal state
            if done:
                break
            else:
                s = next_state





def test():
    init()
    num_episodes = 1000  # number of episodes for evaluation
    episode_max_length = 100
    movingAverageArray = []
    score = 0
    env.reset()
    for i in range(num_episodes):
        s = env.reset()
        d = False
        for t in range(episode_max_length):
            a = np.argmax(Q[s, :])
            s, r, d, _ = env.step(a)
            if d:
                break
        print("evaluation reward", episode_reward)
        print("evaluation reward moving average", moving_avg)
        print("evaluation episode", i)
        movingAverageArray.append(moving_avg)

        # score is x if there is a window of 100 consecutive episodes where moving average was at least x
        if i > 100:
            score = max(score, min(movingAverageArray[i-100:i-1]))