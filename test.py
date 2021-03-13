import numpy as np
import gym
env = gym.make('FrozenLake-v0')
env.seed(0)

# wrapper for accounting rewards
episode_reward = 0
reward_list = []
fixed_window = 100
moving_avg = 0


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

def init():
    rEpisode=0
    rList=[]
    movingAverage=0
    return


def init():
    reward_episode = 0
    reward_list = []
    moving_average = 0
    return


init()
NUM_EPISODES = 1000  # init 1000
episode_max_length = 100  # init 100


def gamma(ep_num, g=.95):
    return g


def alpha(ep_num):
    # 1, .99, .98, .97 ,...., 0
    return 1 - ep_num/NUM_EPISODES


def decision(environment, Q_of_state, ep_num, decision_type):

    if decision_type == "epsilon greedy":
        epsilon = 1 - ep_num/NUM_EPISODES

        prob = np.random.rand()
        if prob < epsilon:
            return environment.action_space.sample()
        else:
            return np.argmax(Q_of_state)


Q = np.zeros([env.observation_space.n, env.action_space.n])
n = np.ones([env.observation_space.n, env.action_space.n])  # record  number of visits to each state


# Start of training
# we are training the Q-values
for episode in range(NUM_EPISODES):
    s = env.reset()  # reset the environment at the beginning of an episode
    d = False  # d := env returns done i.e. terminal state reached

    learning_rate = alpha(episode)
    discount = gamma(episode)

    for scene in range(episode_max_length):

        # p = np.random.rand()
        # if p < exploration:
        #     a = env.action_space.sample()
        # else:
        #     a = np.argmax(Q[s,:])

        a = decision(env, Q[s, :], episode, "epsilon greedy")

        # get new state, reward, done
        # s1 := next state (integer)
        # r := reward
        # d := done
        # info := stuff about the environment
        s1, r, d, info = env.step(a)
        # print(info)

        # lets penalize the holes
        if d and s1 != 15:
            r = - 1

        # update Q
        delta = r + discount*max(Q[s1, :]) - Q[s, a]
        Q_update = Q[s, a] + learning_rate*delta
        Q[s, a] = Q_update

        # break if reached terminal state
        if d:
            break
        else:
            s = s1

# policy evaluation
init()
num_episodes = 1000  # number of episodes for evaluation
episode_max_length = 100
movingAverageArray = []

shaka = []
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
    shaka.append(r)

    # score is x if there is a window of 100 consecutive episodes where moving average was at least x
    if i > 100:
        score = max(score, min(movingAverageArray[i-100:i-1]))

a=5