import torch
import torch.nn as nn
import numpy as np
import os, sys 
try:
    import grid_world_plus
except:
    exec(open('__init__.py').read()) 
    import grid_world_plus
import grid_world_plus as rlm
import grid_world_plus.tools
from grid_world_plus.rlm_env import environment
from grid_world_plus.rl_algos import vpg

from typing import Optional, Tuple, Dict, List

class PretrainingExperiment:
    """Experimentation with pretraining on a small environment.
    
    Args:
        env (Env): 
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
        episode_tracker: vpg.VPGEpisodeTracker = vpg.VPGEpisodeTracker(), 
        num_episodes: int = 10000, 
        discount_factor: float = 0.99, 
        transfer_freq: int = 10, 
        lr: float = 1e-3):

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
        easy_env: rlm.Env = environment.Env(
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
            env_like = easy_env,)
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
        obs: rlm.Observation = environment.Observation(env = self.env)
        if not policy_nn:
            policy_nn: vpg.VPGPolicyNN = self.create_policy_nn(
                env = self.env, obs = obs)

        # Step 1: Pretrain on the easy environment
        policy_nn = self.pretrain_to_threshold(
            policy_nn = policy_nn)

        # Step 2: Transfer learn on the big environment.
        rl_algo = vpg.VPGAlgo(
            policy_nn = policy_nn, 
            env = self.env,)
        rl_algo.run_algo(
            num_episodes = self.num_episodes,
            max_num_scenes = self.max_num_scenes)
        return policy_nn

class VPGEvalExperiment:
    """
    TODO: 
        docs
        test
    """

    def train(
        self,
        env: rlm.Env, 
        num_episodes = 20, 
        discount_factor: float = .99, 
        lr: int = 1e-3,  
        transfer_freq: int = 5):
        """
        TODO: 
            docs
            test
        """

        # Specify parameters
        max_num_scenes: int = env.grid.shape[0] * env.grid.shape[1]
        obs: rlm.Observation = environment.Observation(env = env)
        obs_size: torch.Size = obs.size()
        action_dim: int = len(env.action_space)
        nn_hparams = vpg.NNHyperParameters(lr = lr)
        policy_nn = vpg.VPGPolicyNN(
            obs_size = obs_size, action_dim = action_dim, 
            h_params = nn_hparams)
        transfer_mgmt_train = vpg.VPGTransferLearning(
            transfer_freq = transfer_freq)

        # Run RL algorithm
        training_algo = vpg.VPGAlgo(policy_nn = policy_nn, 
                                    env_like = env, 
                                    transfer_mgmt = transfer_mgmt_train, 
                                    discount_factor = discount_factor)
        training_algo.run_algo(num_episodes = num_episodes, 
                               max_num_scenes = max_num_scenes,
                               training = True)

        return training_algo

    def test(self, 
             rl_algo: vpg.VPGAlgo, 
             env: rlm.Env, 
             num_episodes: int = 10) -> vpg.VPGAlgo:
        """[summary]
        TODO: 
            docs
            test

        Args:
            env (rlm.Env): [description]
            policy_nn (nn.Module): [description]
            num_episodes (int, optional): [description]. Defaults to 10.

        Returns:
            rl_algo (vpg.VPGAlgo): [description]
        """

        max_num_scenes: int = env.grid.shape[0] * env.grid.shape[1]

        rl_algo.run_algo(
            num_episodes = num_episodes, 
            max_num_scenes = max_num_scenes,
            training = False)
        return rl_algo

    def main(self, n_episodes_train: int = 2000, n_episodes_test: int = 100):
        """Trains, test, and then plots the results.
        
        Args:
            n_episodes_train (int): Number of training episodes
            n_episodes_test (int): Number of testing episode
        """

        # initialize agent and an environment with no holes
        train_env: environment.Env = environment.Env(
            grid_shape = (10, 10), n_goals = 1, hole_pct = 0.4)
        train_env.create_new()

        train_algo: vpg.VPGAlgo = self.train(
            env = train_env, num_episodes = n_episodes_train)
        self.plot_results(rl_algo = train_algo, dataset = "train")

        test_env = environment.Env(
            grid_shape = train_env.grid.shape, 
            n_goals = train_env.n_goals, 
            hole_pct = train_env.hole_pct)
        test_algo: vpg.VPGAlgo = self.test(
            rl_algo = train_algo,
            env = test_env, 
            num_episodes = n_episodes_test)
        self.plot_results(rl_algo = test_algo, dataset = "test")
    
    def plot_results(self, rl_algo: vpg.VPGAlgo, dataset: str = "train"):
        """Plots rewards"""
        assert dataset in ["train", "test"]
        episode_trajectories = rl_algo.episode_tracker.trajectories
        episode_rewards =  rl_algo.episode_tracker.episode_rewards
        grid_world_plus.tools.plot_episode_rewards(
            episode_rewards = episode_rewards, 
            title = f"{dataset} rewards")

