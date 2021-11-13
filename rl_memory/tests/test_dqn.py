#!/usr/bin/env python3
"""Test module for all classes related to Deep Q-Learning, Deep Q-Network (DQN),
and Double DQN. See module `rl_memory.rl_algos.dqn`.
"""

import os
import sys
import torch
import random
import pytest
import warnings; warnings.filterwarnings("ignore")

from rl_memory.rlm_env import environment
from rl_memory.rl_algos import dqn
# TODO
# from rl_memory.experiments import dqn_experiments 

import numpy as np
import rl_memory as rlm
from typing import List, Tuple, Optional
from torch import Tensor

Array = np.ndarray


class TestDQNInits:
    """Verifies that all of the abstract classes and concrete classes of 
    the vanilla policy gradient instantiate correctly.
    """

    @staticmethod
    def init_env() -> rlm.Env:
        env: rlm.Env = environment.Env(
            grid_shape=(15,15), n_goals=4, hole_pct = 0.3)
        env.reset()
        return env
    
    def default_experiment_setup(self) \
                                -> Tuple[rlm.Env, dqn.DQN]:
        env: rlm.Env = self.init_env()
        obs: rlm.Observation = environment.Observation(env = env)
        obs_size = obs.size()
        network_h_params = dqn.NNHyperParameters(lr = 1e-3)
        dqn_nn = dqn.DQN(
            obs_size = obs_size, action_dim = len(env.action_space), 
            h_params = network_h_params)
        return env, dqn_nn

    """
    def test_placeholder(self):
        return 'yuh'
        raise NotImplementedError
    """

    def test_init_NNHyperParameters(self):
        network_hparams = dqn.NNHyperParameters(lr = 1e-3)
        assert network_hparams
    
    def test_init_DQNTransferLearning(self):
        transfer_mgmt = dqn.DQNTransferLearning(transfer_freq = 2)
        assert transfer_mgmt
    
    def test_DQNAlgo(self):
        env, dqn_nn = self.default_experiment_setup()
        rl_algo = dqn.DQNAlgo(
            dqn=dqn_nn, 
            env_like = env,)
        rl_algo.run_algo(num_episodes = 5, max_num_scenes = 3)

    def test_DQNAlgo_w_transfer(self):
        env, dqn_nn = self.default_experiment_setup()
        
        transfer_freqs: List[int] = [1, 2, 3]
        for transfer_freq in transfer_freqs:
            rl_algo = dqn.DQNAlgo(
                dqn=dqn_nn, 
                env_like = env, 
                transfer_mgmt = dqn.DQNTransferLearning(
                    transfer_freq = transfer_freq))
            rl_algo.run_algo(num_episodes = 5, max_num_scenes = 3)

class TestPretrainingExperiment:  
    
    @staticmethod
    def init_env() -> rlm.Env:
        env: rlm.Env = environment.Env(
            grid_shape=(15,15), n_goals=4, hole_pct = 0.3)
        env.reset()
        return env

    def default_experiment_setup(self) \
                                -> Tuple[rlm.Env, dqn.DQN]:
        env: rlm.Env = self.init_env()
        obs: rlm.Observation = environment.Observation(
            env = env)
        obs_size = obs.size()
        network_h_params = vpg.NNHyperParameters(lr = 1e-3)
        dqn_nn = dqn.DQN(
            obs_size = obs_size, action_dim = len(env.action_space), 
            h_params = network_h_params)
        return env, dqn_nn

    def test_init_PretrainingExperiment(self):
        env, dqn_nn = self.default_experiment_setup()
        experiment = dqn_experiments.PretrainingExperiment(
            env = env, 
            num_episodes = 3, transfer_freq = 1 )
        assert experiment

    def test_easy_env(self):
        env, dqn_nn = self.default_experiment_setup()
        experiment = dqn_experiments.PretrainingExperiment(
            env = env, 
            num_episodes = 3, transfer_freq = 1 )
        easy_env: rlm.Env = experiment.easy_env()
        assert isinstance(easy_env, environment.Env)

    def test_pretrain_on_easy_env(self):
        env, dqn_nn = self.default_experiment_setup()
        experiment = dqn_experiments.PretrainingExperiment(
            env = env, 
            num_episodes = 3, transfer_freq = 1 )
        experiment.pretrain_on_easy_env(dqn = dqn_nn)

    def test_pretrain_to_threshold(self):
        env, dqn_nn = self.default_experiment_setup()
        experiment = dqn_experiments.PretrainingExperiment(
            env = env, 
            num_episodes = 100, transfer_freq = 1 )
        dqn_nn = experiment.pretrain_to_threshold(
            dqn = dqn_nn)
        return dqn_nn
        
    def test_experiment_vpg_transfer(self):
        return 'yuh' # TODO
        # Check both with default policy net 
        # and with a custom one
        raise NotImplementedError
    
class TestEvaluateDQN:

    def test_train(self):
        """Integration test on whether DQNAlgo runs for training."""
        env = environment.Env()
        env.reset()
        experiment = dqn_experiments.DQNEvalExperiment()
        experiment.train(env = env, num_episodes = 2)

    def test_test(self):
        """Integration test on whether DQNAlgo runs for validation."""
        env = environment.Env()
        env.reset()
        experiment = dqn_experiments.DQNEvalExperiment()
        train_algo: vpg.DQNAlgo = experiment.train(env = env, num_episodes = 1)
        experiment.test(rl_algo = train_algo, env = env, num_episodes = 1)

    def test_plot_results(self):
        env = environment.Env()
        env.reset()
        experiment = dqn_experiments.DQNEvalExperiment()
        experiment.main(n_episodes_train = 500, n_episodes_test = 20)