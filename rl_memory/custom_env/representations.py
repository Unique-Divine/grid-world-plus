# %%
import numpy as np
import torch
import matplotlib.pyplot as plt
from numpy import ndarray
from torch import Tensor
from pprint import pprint

exec(open('__init__.py').read()); import rl_memory
from rl_memory.custom_env import environment
from rl_memory.tests import test_environment

# pop out the state (s_0, a_0)
# Put in on the 0th row of the "graph" 
# State vector, φ(s), is projection onto feature space. 
    # Feature space is the representation of observation space when the 
    # observation space is too large.  

# Use  Q_G(φ(s), a) ← r + γ max_{a'}( Q_G (φ(s'), a')) )

class ImageTransforms:
    rgb_codes = {
        'red': (255, 0, 0), 'green': (0, 255, 0), 'blue': (0, 0, 255), 
        'white': (255, 255, 255), 'black': (0, 0, 0)}
    interactables = {
        'frozen': 0, 'hole': 1, 'goal': 2, 'agent': 7, 'blocked': 3}
    NUM_CHANNELs = 4
    
    def __init__(self):
        pass

    def grid_to_rgby(self, grid) -> Tensor:
        assert isinstance(grid, (ndarray, Tensor))
        is_frozen = grid == self.interactables['frozen']
        is_hole = grid == self.interactables['hole']
        is_goal = grid == self.interactables['goal']
        is_agent = grid == self.interactables['agent']
        is_blocked = grid == self.interactables['blocked']
        # Make holes red, goals green, and blocked pts blue.
        r, g, b = [np.where(condition, 1, 0) \
            for condition in [is_hole, is_goal, is_agent]]
        # Make blocked spots black. 
        r, g, b = [np.where(is_blocked, 0, channel) for channel in [r, g, b]]
        # Make frozen spots white.
        r, g, b = [np.where(is_frozen, 1, channel ) for channel in [r, g, b]]
        y = 0.2126*r + 0.7152*g + 0.0722*b
        
        img: torch.Tensor = torch.from_numpy(np.stack([r, g, b, y], axis=0))
        return img 

    @staticmethod
    def show_rgby(rgby: Tensor):
        rgb: ndarray = rgby[:3].numpy()
        r, g, b  = rgb
        rgb = np.stack([r, g, b], axis=-1)
        # mode = 'RGB'
        plt.imshow(rgb)
        plt.show()

def demo_show_plots():
    it = ImageTransforms()
    env = test_environment.init_env()[0]
    env.create_new()
    img_env = it.grid_to_rgby(grid = env.grid)
    obs = environment.Observation(env=env, agent=environment.Agent(3))
    img_obs = it.grid_to_rgby(grid = obs)

    for an_img in [img_env, img_obs]:
        it.show_rgby(an_img)

class TestImageTransfroms:
    @staticmethod
    def test_interactables_match():
        env, pm = test_environment.init_env()
        img_trans: ImageTransforms = ImageTransforms()
        assert env.interactables == img_trans.interactables, (
            f"The 'interactables' dictionaries must match."
            + f"env.interactables: {env.interactables}\n"
            + f"img_trans.interactables: {img_trans.interactables}")
    
    def test_grid_to_rgby(self):
        env, _ = test_environment.init_env()
        it = ImageTransforms()
        env.create_new()
        env_img = it.grid_to_rgby(grid = env.grid)
        obs = environment.Observation(env=env, agent=environment.Agent(4))
        obs_img = it.grid_to_rgby(grid = obs)
        for img_tensor in [env_img, obs_img]:
            assert isinstance(img_tensor, Tensor)
            assert img_tensor.shape[0] == it.NUM_CHANNELs, (
                "'img_tensor' has the wrong number of channels" 
                + f"({img_tensor.shape[-1]}), when it should have 4.")



