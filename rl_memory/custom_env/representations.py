# %%
import numpy as np
import torch
from torch._C import Value
import torch.nn as nn
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

class ImgTransforms:
    rgb_codes = {
        'red': (255, 0, 0), 'green': (0, 255, 0), 'blue': (0, 0, 255), 
        'white': (255, 255, 255), 'black': (0, 0, 0)}
    interactables = {
        'frozen': 0, 'hole': 1, 'goal': 2, 'agent': 7, 'blocked': 3}
    
    def __init__(self, num_channels: int = 3):
        self.num_channels = num_channels

    def grid_to_rgb(self, grid, use_y: bool = False) -> Tensor:
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

        img: torch.Tensor
        if use_y:
            y = 0.2126*r + 0.7152*g + 0.0722*b
            img = torch.from_numpy(np.stack([r, g, b, y], axis=0))
        else:
            img = torch.from_numpy(np.stack([r, g, b], axis=0))
        return img 

    @staticmethod
    def show_rgb(rgb: Tensor):
        # Format rgb for plt.imshow()
        rgb: ndarray = rgb.float().numpy()
        r, g, b  = rgb
        rgb = np.stack([r, g, b], axis=-1) # mode = 'RGB'
        # Show the image 
        plt.imshow(rgb)
        plt.show()

def demo_show_plots():
    it = ImgTransforms()
    env = test_environment.init_env()[0]
    env.create_new()
    img_env = it.grid_to_rgb(grid = env.grid)
    obs = environment.Observation(env=env, agent=environment.Agent(4))
    img_obs = it.grid_to_rgb(grid = obs)
    for an_img in [img_env, img_obs]:
        it.show_rgb(an_img)

class ImgEncoder(nn.Module):
    """Convolutional pixel encoder f_θ

    input_img
    Conv2d input size is (N, C, H, W), where H is the img height, W the img 
        width, C the number of channels (3 for RGB), and N is the batch size
    ouput_channels is the number of filters, a.k.a. kernels.
    """
    def __init__(self, obs_shape, batch_size = 1, num_img_channels=3):
        super().__init__()
        self.obs_shape = obs_shape
        self.batch_size = batch_size

        self.num_img_channels = num_img_channels
        num_filters = 20
        img_len = obs_shape[-1]
        filter_len = 3
        padding = 0
        stride = 1
        output_len = ((img_len - filter_len + 2*padding) / stride) + 1
        print(output_len)
        # Architecture
        self.conv_l0 = nn.Conv2d(
            in_channels=3, out_channels=num_filters, kernel_size=filter_len, 
            padding=padding)
        
        self.img_len = img_len

    def forward(self, x):
        correct_shape = torch.Size((self.batch_size, self.num_img_channels, 
                                    self.img_len, self.img_len))
        assert x.shape == correct_shape, (
            f"Expected input of shape (N, C, H, W), which is {correct_shape},"
            + f" but got input with {x.shape} instead")

        return self.conv_l0(x)

# ---------------------------------------------------------
# Test Classes
# ---------------------------------------------------------

class TestImgEncoder:
    @staticmethod
    def test_forward():
        SIGHT_DISTANCE: int = 4
        
        # Create and initialize environment
        env, pm = test_environment.init_env()
        env.create_new()

        # Get observation as an image
        obs = environment.Observation(env=env, agent=environment.Agent(
            sight_distance = SIGHT_DISTANCE))
        img_transforms = ImgTransforms()
        img_len: int = 2*SIGHT_DISTANCE + 1
        obs_img = img_transforms.grid_to_rgb(obs).view(
            1, img_transforms.num_channels, img_len, img_len).float()

        # Pass observation image through the image encoder
        rl_encoder = ImgEncoder(obs_shape = obs.shape, batch_size=1)
        out = rl_encoder(obs_img)
        return out

class TestImgTransfroms:
    @staticmethod
    def test_interactables_match():
        env, pm = test_environment.init_env()
        img_trans: ImgTransforms = ImgTransforms()
        assert env.interactables == img_trans.interactables, (
            f"The 'interactables' dictionaries must match."
            + f"env.interactables: {env.interactables}\n"
            + f"img_trans.interactables: {img_trans.interactables}")
    
    def test_grid_to_rgb(self):
        env, _ = test_environment.init_env()
        it = ImgTransforms()
        env.create_new()
        env_img = it.grid_to_rgb(grid = env.grid)
        obs = environment.Observation(env=env, agent=environment.Agent(4))
        obs_img = it.grid_to_rgb(grid = obs)
        for img_tensor in [env_img, obs_img]:
            assert isinstance(img_tensor, Tensor)
            assert img_tensor.shape[0] == it.num_channels, (
                "'img_tensor' has the wrong number of channels" 
                + f"({img_tensor.shape[-1]}), when it should have 4.")

#%%

out = TestImgEncoder.test_forward()
print(out.shape)