import torch
import numpy as np
import os, sys 
try:
    import rl_memory
except:
    exec(open('__init__.py').read()) 
    import rl_memory
import rl_memory as rlm
from rl_memory.custom_env import environment
from rl_memory.rl_algos import vpg

from typing import Optional

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
    def __init__(self, 
                 env: rlm.Env, 
                 agent: rlm.Agent, 
                 episode_tracker: vpg.VPGEpisodeTracker, 
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

    def create_policy_network(
            self, 
            env: rlm.Env, 
            obs: rlm.Observation) -> vpg.VPGNetwork:
        action_dim: int = len(env.action_space)
        obs_size: torch.Size = obs.size()
        network_h_params = vpg.NetworkHyperParameters(lr = self.lr)
        policy_network = vpg.VPGNetwork(
            obs_size = obs_size, action_dim = action_dim, 
            h_params = network_h_params)
        return policy_network

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
            policy_network: vpg.VPGNetwork):
        """TODO docstring
        Methods come from RLAlgorithm
        """
        easy_env = self.easy_env()
        rl_algo = vpg.VPGAlgo(
            policy_network = policy_network,
            env_like = easy_env,
            agent = self.agent, )
        rl_algo.run_algo(
            num_episodes = self.num_episodes, 
            max_num_scenes = self.max_num_scenes)
        
        return rl_algo.policy_network

    def pretrain_to_threshold(
        self, 
        policy_network: vpg.VPGNetwork, 
        ep_len_threshold: float = 3.3, 
        trajectory_lookback_window: int = 500) -> vpg.VPGNetwork:
        """Recursively trains the agent (policy network) on an easy environment
        until it can solve it quickly and consistently.

        Args:
            policy_network (vpg.VPGNetwork): The network that receives pre-training. 
            ep_len_threshold (float): Defaults to 3.3.
            trajectory_lookback_window (int): Defaults to 500.
        
        Returns
            policy_network (vpg.VPGNetwork): Pre-trained network.
        """

        avg_episode_len = np.infty

        while avg_episode_len > ep_len_threshold:
            policy_network = self.pretrain_on_easy_env(
                policy_network = policy_network)
            avg_episode_len = np.mean(
                [len(traj) for traj in self.episode_tracker.trajectories[
                     -trajectory_lookback_window:]])
            print(avg_episode_len)
        return policy_network

    def experiment_vpg_transfer(
            self, 
            policy_network: Optional[vpg.VPGNetwork] = None) -> vpg.VPGNetwork:
        """Runs an experiment to see if the environment is solvable with VPG 
        and pre-training on the easy environment. 

        Experiment steps:
        0. Initialize new env and policy network.
        1. Pretrain a network on the easy environment. 
        2. Transfer learn on the big environment, which has many holes. 
        
        Returns:
            policy_network
        """

        # Step 0: Initialize new env and policy network
        self.env.create_new()
        obs: rlm.Observation = environment.Observation(self.env, self.agent)
        if not policy_network:
            policy_network: vpg.VPGNetwork = self.create_policy_network(
                env = self.env, obs = obs)

        # Step 1: Pretrain on the easy environment
        policy_network = self.pretrain_to_threshold(
            policy_network = policy_network)

        # Step 2: Transfer learn on the big environment.
        rl_algo = vpg.VPGAlgo(
            policy_network = policy_network, 
            env = self.env,
            agent = self.Agent)
        rl_algo.run_algo(
            num_episodes = self.num_episodes,
            max_num_scenes = self.max_num_scenes)
        return policy_network
