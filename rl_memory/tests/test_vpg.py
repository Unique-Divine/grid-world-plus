import os, sys
try:
    import rl_memory
except:
    exec(open('__init__.py').read()) 
    import rl_memory
import rl_memory as rlm
from rl_memory import rlm_env
from rl_memory.rl_algos import vpg
from rl_memory.experiments import vpg_experiments
import numpy as np
import torch
import random
import warnings; warnings.filterwarnings("ignore")
# Type imports
from typing import List, Tuple, Optional
from torch import Tensor
Array = np.ndarray

import pytest

class TestVPGInits:
    """Verifies that all of the abstract classes and concrete classes of 
    the vanilla policy gradient instantiate correctly.
    """

    @staticmethod
    def init_env() -> rlm.Env:
        env: rlm.Env = rlm_env.Env(
            grid_shape=(15,15), n_goals=4, hole_pct = 0.3)
        env.reset()
        return env
    
    def default_experiment_setup(self) \
                                -> Tuple[rlm.Env, vpg.VPGPolicyNN]:
        env: rlm.Env = self.init_env()
        obs: rlm.Observation = rlm_env.Observation(env = env)
        obs_size = obs.size()
        network_h_params = vpg.NNHyperParameters(lr = 1e-3)
        policy_nn = vpg.VPGPolicyNN(
            obs_size = obs_size, action_dim = len(env.action_space), 
            h_params = network_h_params)
        return env, policy_nn

    def test_placeholder(self):
        return 'yuh'
        raise NotImplementedError

    def test_init_NNHyperParameters(self):
        network_hparams = vpg.NNHyperParameters(lr = 1e-3)
        assert network_hparams
    
    def test_init_VPGTransferLearning(self):
        transfer_mgmt = vpg.VPGTransferLearning(transfer_freq = 2)
        assert transfer_mgmt
    
    def test_VPGAlgo(self):
        env, policy_nn = self.default_experiment_setup()
        rl_algo = vpg.VPGAlgo(
            policy_nn = policy_nn, 
            env_like = env,)
        rl_algo.run_algo(num_episodes = 5, max_num_scenes = 3)

    def test_VPGAlgo_w_transfer(self):
        env, policy_nn = self.default_experiment_setup()
        
        transfer_freqs: List[int] = [1, 2, 3]
        for transfer_freq in transfer_freqs:
            rl_algo = vpg.VPGAlgo(
                policy_nn = policy_nn, 
                env_like = env, 
                transfer_mgmt = vpg.VPGTransferLearning(
                    transfer_freq = transfer_freq))
            rl_algo.run_algo(num_episodes = 5, max_num_scenes = 3)

class TestPretrainingExperiment:  
    
    @staticmethod
    def init_env() -> rlm.Env:
        env: rlm.Env = rlm_env.Env(
            grid_shape=(15,15), n_goals=4, hole_pct = 0.3)
        env.reset()
        return env

    def default_experiment_setup(self) \
                                -> Tuple[rlm.Env, vpg.VPGPolicyNN]:
        env: rlm.Env = self.init_env()
        obs: rlm.Observation = rlm_env.Observation(
            env = env)
        obs_size = obs.size()
        network_h_params = vpg.NNHyperParameters(lr = 1e-3)
        policy_nn = vpg.VPGPolicyNN(
            obs_size = obs_size, action_dim = len(env.action_space), 
            h_params = network_h_params)
        return env, policy_nn

    def test_init_PretrainingExperiment(self):
        env, policy_nn = self.default_experiment_setup()
        experiment = vpg_experiments.PretrainingExperiment(
            env = env, 
            num_episodes = 3, transfer_freq = 1 )
        assert experiment

    def test_easy_env(self):
        env, policy_nn = self.default_experiment_setup()
        experiment = vpg_experiments.PretrainingExperiment(
            env = env, 
            num_episodes = 3, transfer_freq = 1 )
        easy_env: rlm.Env = experiment.easy_env()
        assert isinstance(easy_env, rlm_env.Env)

    def test_pretrain_on_easy_env(self):
        env, policy_nn = self.default_experiment_setup()
        experiment = vpg_experiments.PretrainingExperiment(
            env = env, 
            num_episodes = 3, transfer_freq = 1 )
        experiment.pretrain_on_easy_env(policy_nn = policy_nn)

    def test_pretrain_to_threshold(self):
        return 'yuh' # TODO
        raise NotImplementedError
        
    def test_experiment_vpg_transfer(self):
        return 'yuh' # TODO
        # Check both with default policy net 
        # and with a custom one
        raise NotImplementedError
    
