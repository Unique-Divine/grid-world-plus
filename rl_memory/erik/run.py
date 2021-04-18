from rl_memory.erik.pg_cnn import RLAlgorithms
from rl_memory.erik.tools import plot_episode_history, config
import matplotlib.pyplot as plt
import numpy as np

a = RLAlgorithms(**config)
m = a.pg_cnn_transfer()

returns = np.array(a.episode_returns)
rewards = np.array(a.episode_rewards)

avg_returns = [np.mean(returns[:i+1]) for i in range(0, len(returns), 1)]
avg_rewards = [np.mean(rewards[:i+1]) for i in range(0, len(rewards), 1)]

to_graph0 = [(returns, .9, "agent returns"), (avg_returns, .9, "moving average returns")]
to_graph1 = [(rewards, .9, "agent rewards"), (avg_rewards, .9, "moving average rewards")]

plot_episode_history(to_graph0)
plot_episode_history(to_graph1)

plt.show()
