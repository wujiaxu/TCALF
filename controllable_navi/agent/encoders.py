import torch
from torch import nn
import torch.nn.functional as F
from hydra.core.config_store import ConfigStore
from dataclasses import dataclass
import omegaconf
import hydra
import typing as tp
from typing import Any, Tuple
from controllable_navi import utils

@dataclass
class StateEncoderConfig:
    # _target_: str = "controllable_navi.agent.encoders.StateEncoder"
    # obs_shape: tp.Tuple[int, ...] = omegaconf.MISSING  # to be specified later
    feature_dim:int = 32

@dataclass
class ScanEncoderConfig:
    # _target_: str = "controllable_navi.agent.encoders.ScanEncoder"
    # obs_shape: tp.Tuple[int, ...] = omegaconf.MISSING  # to be specified later
    feature_dim:int = 128
    out_channels_1:int =32
    kernel_size_1:int =5
    stride_1:int =2
    out_channels_2:int =32
    kernel_size_2:int =3
    stride_2:int =2
    liner_layer_1:int = 256

@dataclass
class ImageEncoderConfig:
    # _target_: str = "controllable_navi.agent.encoders.ImageEncoder"
    feature_dim:int = 128
    # obs_shape: tp.Tuple[int, ...] = omegaconf.MISSING  # to be specified later
    
@dataclass
class EncoderConfig:
    state_encoder:StateEncoderConfig=StateEncoderConfig()
    scan_encoder:ScanEncoderConfig=ScanEncoderConfig()
    image_encoder:ImageEncoderConfig=ImageEncoderConfig()

# cs = ConfigStore.instance()
# cs.store(name="state_encoder", node=StateEncoderConfig)
# cs.store(name="scan_encoder", node=ScanEncoderConfig)
# cs.store(name="image_encoder", node=ImageEncoderConfig)

class ImageEncoder(nn.Module):
    def __init__(self, input_shape,cfg:ImageEncoderConfig) -> None:
        super().__init__()

        self.cfg = cfg

        assert len(input_shape) == 3
        self.repr_dim = 32 * 10 * 10

        self.convnet = nn.Sequential(nn.Conv2d(input_shape[0], 32, 3, stride=2),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU())

        self.apply(utils.weight_init)

    def forward(self, obs) -> Any:
        obs = obs / 255.0 - 0.5
        h = self.convnet(obs)
        h = h.view(h.shape[0], -1)
        return h
    
class StateEncoder(nn.Module):
    def __init__(self, input_shape,cfg:StateEncoderConfig) -> None:
        super().__init__()
        self.cfg = cfg
        assert len(input_shape) == 1
        self.net = nn.Sequential(nn.Linear(input_shape[0], cfg.feature_dim),
                                   nn.LayerNorm(cfg.feature_dim))
        self.repr_dim = cfg.feature_dim

    def forward(self,x):
        return self.net(x)

class ScanEncoder(nn.Module):
    def __init__(self, input_shape,cfg:ScanEncoderConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.input_shape = input_shape
        assert len(input_shape) == 1
        
        input_channels = 1
        # First convolutional layer
        self.conv1 = nn.Conv1d(in_channels=input_channels, 
                               out_channels=cfg.out_channels_1, 
                               kernel_size=cfg.kernel_size_1, 
                               stride=cfg.stride_1)
        # Second convolutional layer
        self.conv2 = nn.Conv1d(in_channels=cfg.out_channels_1, 
                               out_channels=cfg.out_channels_2, 
                               kernel_size=cfg.kernel_size_2, 
                               stride=cfg.stride_2)
        # Fully connected layer
        self.fc1 = nn.Linear(in_features=self._get_conv_output_size(input_channels), 
                             out_features=cfg.liner_layer_1)
        # Output layer (if needed, depends on the task)
        self.fc2 = nn.Linear(in_features=cfg.liner_layer_1, 
                             out_features=cfg.feature_dim)
        
        self.repr_dim = cfg.feature_dim

    def _get_conv_output_size(self, input_channels):
        # Function to compute the size of the output from the conv layers
        # Assuming input size is (N, input_channels, L) where L is the length of the sequence
        dummy_input = torch.zeros(1, input_channels, self.input_shape[0])  # Replace 100 with an appropriate sequence length
        dummy_output = self._forward_conv_layers(dummy_input)
        return int(torch.flatten(dummy_output, 1).size(1))
    
    def _forward_conv_layers(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x
    
    def forward(self, x):
        x = x.unsqueeze(-1).transpose(-2, -1)
        if len(x.shape)==4:
            N,L,C,S = x.shape
            x = x.view(-1,C,S)
        else:
            L = None
            N,C,S = x.shape
        x = self._forward_conv_layers(x)
        x = torch.flatten(x, 1)  # Flatten the tensor except for the batch dimension
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        if L is not None:
            x = x.view(N,L,-1)
        return x
    
class MultiModalEncoder(nn.Module):
    def __init__(self, obs_shape,cfg:EncoderConfig) -> None:
        super().__init__()
        self.sub_encoders = {}
        self.obs_shape_dict = obs_shape
        self.repr_dim = 0
        for name in self.obs_shape_dict.keys():
            input_shape,input_size = self.obs_shape_dict[name]
            if 'scan' in name:
                self.sub_encoders[name]=ScanEncoder(input_shape,cfg.scan_encoder)
            else:
                self.sub_encoders[name]=StateEncoder(input_shape,cfg.state_encoder)
            self.repr_dim += self.sub_encoders[name].repr_dim
    
    def parameters(self, recurse: bool = True) -> tp.Iterator[nn.Parameter]:
        for name, subnet in self.sub_encoders.items():
            for item in subnet.parameters():
                yield item
    
    def to_device(self,device):
        for name in self.obs_shape_dict.keys():
            self.sub_encoders[name].to(device)

    def forward(self,x):
        start = 0
        hs = []
        for name in self.obs_shape_dict.keys():
            input_shape,input_size = self.obs_shape_dict[name]
            in_ = x[...,start:start+input_size]
            hs.append(self.sub_encoders[name](in_))
            start = input_size
        h = torch.cat(hs, dim=-1)
        return h