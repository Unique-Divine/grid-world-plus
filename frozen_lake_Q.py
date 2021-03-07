# %%
from tools import discount_rate, lr, decision
import numpy as np
import gym

# %%
class FrozenLakeQ():
    def __init__(self, env = gym.make('FrozenLake-v0'), 
        env_seed: int = 0, num_episodes: int = 1000, 
        max_num_scenes: int = 100) -> None:

        self.env = env
        self.env.seed(env_seed)

        self.episode_reward = 0
        self.reward_list = []
        # self.moving_average = 0
        self.num_episodes = num_episodes
        self.max_num_scenes = max_num_scenes

        self.Q = np.zeros([env.observation_space.n, env.action_space.n])
        self.n = np.ones([env.observation_space.n, env.action_space.n])  # record  number of visits to each state

    def train(self):
        """Tabular Q-learning training method."""
        num_episodes = self.num_episodes
        max_num_scenes = self.max_num_scenes

        episode_rewards = []
        episode_trajectories = []

        for episode in range(num_episodes):
            state = self.env.reset() # Reset environment 

            alpha = lr(episode, num_episodes)
            gamma = discount_rate(episode, num_episodes)
            scene_rewards = []
            trajectory = []

            for scene in range(max_num_scenes):
                action = decision(self.env, self.Q[state, :], episode, 
                                  num_episodes, "epsilon greedy")

                perform_step: tuple = self.env.step(action)
                next_state: int
                done: bool  # 'done': True if in terminal state
                next_state, reward, done, info = perform_step                
                
                # Save the rewards and trajectory for the scene
                scene_rewards.append(reward)
                trajectory.append(next_state)

                # Penalize agent for falling into the holes.
                if done and next_state != 15:
                    reward = - 1

                # Update Q-fn
                delta = reward + gamma*np.max(self.Q[next_state, :])
                delta = delta - self.Q[state, action]
                Q_update = self.Q[state, action] + alpha*delta
                self.Q[state, action] = Q_update

                if done:
                    break  # Terminal state reached.
                else:
                    state = next_state  # Advance to next state

            
            episode_rewards.append(scene_rewards)
            episode_trajectories.append(trajectory)

    def test(self,
        episode_count: int = 1000,
        episode_max_length: int = 100) -> list:

        self.episode_reward = 0
        self.reward_list = []
        self.moving_average = 0

        self.env.reset()
        for i in range(episode_count):
            state = self.env.reset()
            done: bool = False
            for t in range(episode_max_length):
                action = np.argmax(self.Q[state, :])
                state, episode_reward, done, _ = self.env.step(action)
                if done:
                    break

            self.reward_list.append(episode_reward)
            print("evaluation reward", episode_reward)

        return self.reward_list
# %%
def toy():
    fl = FrozenLakeQ()
    fl.train()
    test_rewards = fl.test()
    a=5

toy()
# %%
