# %%
import numpy as np
import torch
from torch._C import Value
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from numpy import ndarray
from torch import Tensor
from pprint import pprint

from torch.nn.modules.normalization import LayerNorm

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
    """Convolutional pixel encoder for observations

    input_img
    Conv2d input size is (N, C, H, W), where H is the img height, W the img 
        width, C the number of channels (3 for RGB), and N is the batch size
    ouput_channels is the number of filters, a.k.a. kernels.
    """
    def __init__(self, obs_shape, rep_dim: int = 128, num_img_channels=3, 
                 activate_output: bool = True):
        super().__init__()
        self.obs_shape = obs_shape; assert len(obs_shape) == 3
        batch_size = obs_shape[0]
        self.batch_size = batch_size
        self.activate_output = activate_output

        # Convolutional layer parameters
        self.num_img_channels = num_img_channels
        num_filters = 24
        img_width = obs_shape[-1]
        self.img_width = img_width
        filter_width = 3
        l0_padding = 0
        l0_stride = 1
        l0_output_width = (
            (img_width - filter_width + 2*l0_padding) / l0_stride) + 1
        l1_output_width = (l0_output_width - filter_width) + 1

        # --------------------------------------------
        # Architecture
        # --------------------------------------------
        self.conv_l0 = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=num_filters, 
                kernel_size=filter_width, stride = l0_stride, 
                padding=l0_padding, ),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=num_filters),
            nn.Dropout(p=0.1))
        self.conv_l1 = nn.Sequential(
            nn.Conv2d(
                in_channels=num_filters, out_channels=num_filters, 
                kernel_size=filter_width, stride = 1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=num_filters))
 
        self.conv_layers = [self.conv_l0, self.conv_l1]
        fc_in_dim = int(num_filters*l1_output_width*l1_output_width)
        self.fc = nn.Sequential(
            nn.Linear(in_features=fc_in_dim, 
                      out_features=rep_dim),
            nn.LayerNorm(rep_dim))

        self.outputs: dict = {}

        # __init__ mini tests
        assert obs_shape[-1] == obs_shape[-2], (
            "The input observation should be a square image.")        

    def forward_convolutions(self, obs: Tensor) -> Tensor:
        """[summary]

        Args:
            obs (Tensor): [description]
        """
        self.outputs['obs'] = obs
        batch_size = obs.size(0)
        # Pass observation through conv_layers
        x = obs
        for idx, layer in enumerate(self.conv_layers):
            x = layer(x)
            self.outputs[f'conv_l{idx}'] = x
        
        embedding = x.view(batch_size, -1) # store as column vector
        return embedding

    def forward(self, x: Tensor, detach_grad=False) -> Tensor:
        batch_size = x.size(0)
        correct_shape = torch.Size([batch_size, self.num_img_channels, 
                                    self.img_width, self.img_width])
        assert x.shape == correct_shape, (
            f"Expected 'x' of shape (N, C, H, W), which is {correct_shape},"
            + f" but got 'x' with {x.shape} instead")

        embedding: Tensor = self.forward_convolutions(x)
        if detach_grad:
            embedding = embedding.detach()
        unactivated_output = self.fc(embedding)
        if self.activate_output:
            output = unactivated_output
        else:
            output = torch.tanh(unactivated_output)
            self.outputs['tanh_out'] = output

        return output

# ---------------------------------------------------------
# Test Classes
# ---------------------------------------------------------

class TestImgEncoder:
    @staticmethod
    def test_forward():
        BATCH_SIZE = 10
        SIGHT_DISTANCE: int = 4
        
        # Create and initialize environment
        env, pm = test_environment.init_env()
        env.create_new()

        # Get observation as an image
        obs = environment.Observation(env=env, agent=environment.Agent(
            sight_distance = SIGHT_DISTANCE))
        img_transforms = ImgTransforms()
        img_width: int = 2*SIGHT_DISTANCE + 1
        obs_img = img_transforms.grid_to_rgb(obs).float()
        assert obs_img.size() == torch.Size(
            (img_transforms.num_channels, img_width, img_width))
        
        # Create batch of observations to test the forward pass
        obs_imgs = [obs_img] * BATCH_SIZE
        batch = torch.stack(tensors=obs_imgs, dim=0)

        # Pass batch of images through image encoder
        rl_encoder = ImgEncoder(obs_shape = obs_img.shape)
        breakpoint()
        out: Tensor = rl_encoder(batch)
        assert out.shape[0] == BATCH_SIZE

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

# TestImgEncoder.test_forward()
