import torch
import torch.nn as nn
import numpy as np
import os, sys 
try:
    import rl_memory
except:
    exec(open('__init__.py').read()) 
    import rl_memory
import rl_memory as rlm
import rl_memory.tools
from rl_memory import rlm_env
from rl_memory.rl_algos import vpg

from typing import Optional, Tuple, Dict, List

class PretrainingExperiment:
    """Experimentation with pretraining on a small environment.
    
    Args:
        env (Env): 
        agent (rlm.Agent): 
        episode_tracker (vpg.VPGEpisodeTracker):
        num_episodes (int): Number of episodes to evaluate the agent. Defaults 
            to 10000.
        discount_factor (float): Factor used to discount rewards. 
            Denoted γ (gamma) in the RL literature. Defaults to 0.99.
        transfer_freq (int): Th number of episodes the environment will remain 
            constant before randomizing for transfer learning. Defaults to 10.
        lr (float): Policy network learning rate. 
            Denoted α (alpha) in the RL literature. Defaults to 1e-3.
    
    Attributes:
        agent (rlm.Agent): 
        episode_tracker (vpg.VPGEpisodeTracker):
        num_episodes (int): Number of episodes to evaluate the agent. Defaults 
            to 10000.
        discount_factor (float): Factor used to discount rewards. A.K.A. gamma
            in the RL literature. Defaults to 0.99.
        transfer_freq (int): Th number of episodes the environment will remain 
            constant before randomizing for transfer learning. Defaults to 10.
        lr (float): Policy network learning rate. 
            Denoted α (alpha) in the RL literature. Defaults to 1e-3.
    """
    def __init__(
        self, 
        env: rlm.Env, 
        agent: rlm.Agent, 
        episode_tracker: vpg.VPGEpisodeTracker = vpg.VPGEpisodeTracker(), 
        num_episodes: int = 10000, 
        discount_factor: float = 0.99, 
        transfer_freq: int = 10, 
        lr: float = 1e-3):

        self.agent = agent
        self.episode_tracker = episode_tracker

        # Environment parameters
        self.grid_shape = env.grid.shape
        self.num_goals = env.n_goals
        self.hole_pct = env.hole_pct

        # Experiment hyperparams
        self.num_episodes = num_episodes
        self.discount_factor = discount_factor
        self.lr = lr
        self.transfer_freq = transfer_freq
        self.max_num_scenes: int = 3 * self.grid_shape[0] * self.grid_shape[1]

    def create_policy_nn(
            self, 
            env: rlm.Env, 
            obs: rlm.Observation) -> vpg.VPGPolicyNN:
        action_dim: int = len(env.action_space)
        obs_size: torch.Size = obs.size()
        nn_hparams = vpg.NNHyperParameters(lr = self.lr)
        policy_nn = vpg.VPGPolicyNN(
            obs_size = obs_size, action_dim = action_dim, 
            h_params = nn_hparams)
        return policy_nn

    @staticmethod
    def easy_env() -> rlm.Env:
        """
        Returns:
            (Env): A small, 3 by 3 environment with one goal and one hole.
        """
        easy_env: rlm.Env = rlm_env.Env(
            grid_shape=(3, 3), n_goals=1, hole_pct=.1)
        easy_env.reset()
        return easy_env

    def pretrain_on_easy_env(
            self, 
            policy_nn: vpg.VPGPolicyNN):
        """TODO docstring
        Methods come from RLAlgorithm
        """
        easy_env = self.easy_env()
        rl_algo = vpg.VPGAlgo(
            policy_nn = policy_nn,
            env_like = easy_env,
            agent = self.agent, )
        rl_algo.run_algo(
            num_episodes = self.num_episodes, 
            max_num_scenes = self.max_num_scenes)
        
        return rl_algo.policy_nn

    def pretrain_to_threshold(
        self, 
        policy_nn: vpg.VPGPolicyNN, 
        ep_len_threshold: float = 3.3, 
        trajectory_lookback_window: int = 500) -> vpg.VPGPolicyNN:
        """Recursively trains the agent (policy network) on an easy environment
        until it can solve it quickly and consistently.

        Args:
            policy_nn (vpg.VPGPolicyNN): The network that receives pre-training. 
            ep_len_threshold (float): Defaults to 3.3.
            trajectory_lookback_window (int): Defaults to 500.
        
        Returns
            policy_nn (vpg.VPGPolicyNN): Pre-trained network.
        """

        avg_episode_len = np.infty

        while avg_episode_len > ep_len_threshold:
            policy_nn = self.pretrain_on_easy_env(
                policy_nn = policy_nn)
            avg_episode_len = np.mean(
                [len(traj) for traj in self.episode_tracker.trajectories[
                     -trajectory_lookback_window:]])
            print(avg_episode_len)
        return policy_nn

    def experiment_vpg_transfer(
            self, 
            policy_nn: Optional[vpg.VPGPolicyNN] = None) -> vpg.VPGPolicyNN:
        """Runs an experiment to see if the environment is solvable with VPG 
        and pre-training on the easy environment. 

        Experiment steps:
        0. Initialize new env and policy network.
        1. Pretrain a network on the easy environment. 
        2. Transfer learn on the big environment, which has many holes. 
        
        Returns:
            policy_nn
        """

        # Step 0: Initialize new env and policy network
        self.env.create_new()
        obs: rlm.Observation = rlm_env.Observation(self.env, self.agent)
        if not policy_nn:
            policy_nn: vpg.VPGPolicyNN = self.create_policy_nn(
                env = self.env, obs = obs)

        # Step 1: Pretrain on the easy environment
        policy_nn = self.pretrain_to_threshold(
            policy_nn = policy_nn)

        # Step 2: Transfer learn on the big environment.
        rl_algo = vpg.VPGAlgo(
            policy_nn = policy_nn, 
            env = self.env,
            agent = self.Agent)
        rl_algo.run_algo(
            num_episodes = self.num_episodes,
            max_num_scenes = self.max_num_scenes)
        return policy_nn

# ----------------------------------------------------------------------
#               Begin Experiment
# ----------------------------------------------------------------------
class UnnamedExperiment:
    """
    TODO: 
        docs
        test
    """

    def train(env: rlm.Env, 
            obs: rlm.Observation,
            agent: rlm.Agent, 
            num_episodes = 20, 
            discount_factor: float = .99, 
            lr = 1e-3,  
            transfer_freq = 5):
        """
        TODO: 
            docs
            test
        """

        grid_shape: Tuple[int] = env.grid.shape
        max_num_scenes = 3 * grid_shape[0] * grid_shape[1]

        # Specify parameters
        obs_size: torch.Size = obs.observation.size
        action_dim: int = len(env.action_space)
        nn_hparams = vpg.NNHyperParameters(lr = lr)
        policy_nn = vpg.VPGPolicyNN(
            obs_size = obs_size, action_dim = action_dim, 
            h_params = nn_hparams)
        episode_tracker_train = vpg.VPGEpisodeTracker()
        transfer_mgmt_train = vpg.VPGTransferLearning(transfer_freq = transfer_freq)

        # Run RL algorithm
        training_algo = vpg.VPGAlgo(
            policy_nn = policy_nn, 
            env_like = env, 
            agent = agent, 
            episode_tracker = episode_tracker_train, 
            transfer_mgmt = transfer_mgmt_train, 
            discount_factor = discount_factor)
        training_algo.run_algo(
            num_episodes = num_episodes, max_num_scenes = max_num_scenes)

        return policy_nn, episode_tracker_train

    def test(self, 
             env: rlm.Env, 
             agent: rlm.Agent, 
             policy: nn.Module, 
             num_episodes: int = 10):
        """[summary]
        TODO: 
            docs
            test

        Args:
            env (rlm.Env): [description]
            agent (rlm.Agent): [description]
            policy (nn.Module): [description]
            num_episodes (int, optional): [description]. Defaults to 10.

        Returns:
            [type]: [description]
        """

        max_num_scenes = env.grid_shape[0] * env.grid_shape[1]
        env.create_new()

        episode_trajectories = []
        episode_rewards = []

        for episode_idx in range(num_episodes):

            episode_envs = []
            reward_sum = 0
            scene_number = 0
            done: bool = False

            while not done:
                episode_envs.append(env.render_as_char(env.grid))
                state = environment.State(env, agent)
                action_distribution = policy.action_distribution(
                    state.observation.flatten())
                a = action_distribution.sample()

                new_state, r, done, info = env.step(action_idx=a, state=state)
                reward_sum += r

                scene_number += 1
                if scene_number > max_num_scenes:
                    break

            episode_trajectories.append(episode_envs)
            episode_rewards.append(reward_sum)

        return episode_rewards, episode_trajectories

    def main(self):

        # initialize agent and an environment with no holes
        james_bond: rlm.Agent = rlm_env.Agent(4)
        env: rlm.Env = rlm_env.Env(
            grid_shape = (10, 10), 
            n_goals = 1, 
            hole_pct = 0.4)
        env.create_new()
        obs: rlm.Observation = rlm_env.Observation(env=env, agent=james_bond)

        episode_trajectories: Dict[List] = {}
        episode_rewards: Dict[List[float]] = {}

        dataset = "train"
        policy_nn, episode_tracker = self.train(
            env=env, obs=obs, agent=james_bond, num_episodes = 20)
        episode_trajectories[dataset] = episode_tracker.trajectories
        episode_rewards[dataset] =  episode_tracker.rewards
        rlm.tools.plot_episode_rewards(
            values = episode_rewards[dataset], 
            title = f"{dataset} rewards", 
            reset_frequency = 5)

        dataset = "test"
        test_env = rlm_env.Env(
            grid_shape=env.grid_shape, n_goals=env.n_goals, hole_pct=env.hole_pct)
        test_episode_rewards, episode_trajectories['test'] = self.test(
            env = test_env, 
            agent = james_bond, 
            policy = policy_nn, 
            num_episodes = 10)
        rlm.tools.plot_episode_rewards(
            values = episode_rewards[dataset], 
            title = f"{dataset} rewards", 
            reset_frequency = 5)