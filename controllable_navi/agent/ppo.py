import typing as tp
import dataclasses
from typing import Any, Tuple
from collections import OrderedDict

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from hydra.core.config_store import ConfigStore
import omegaconf

from controllable_navi.dmc import TimeStep
from controllable_navi.in_memory_replay_buffer import ReplayBuffer
from controllable_navi import utils
from .fb_modules import mlp
from .ddpg import Actor, Critic
from controllable_navi.agent.encoders import MultiModalEncoder,EncoderConfig
from controllable_navi.agent.sequence_encoders import SequenceEncoder,SequenceEncoderConfig
MetaDict = tp.Mapping[str, np.ndarray]

@dataclasses.dataclass
class PPOAgentConfig:
    _target_: str = "controllable_navi.agent.ppo.PPOAgent"
    name: str = "ddpg"
    reward_free: bool = omegaconf.II("reward_free")
    obs_type: str = omegaconf.MISSING  # to be specified later
    obs_shape: dict = omegaconf.MISSING  # to be specified later
    action_shape: tp.Tuple[int, ...] = omegaconf.MISSING  # to be specified later
    encoder_config:EncoderConfig=EncoderConfig()
    sequence_encoder_config:SequenceEncoderConfig=SequenceEncoderConfig()
    device: str = omegaconf.II("device")
    lr: float = 1e-5

cs = ConfigStore.instance()
cs.store(group="agent", name="ppo", node=PPOAgentConfig)
   

class PPOAgent:
    encoder: tp.Union[SequenceEncoder, nn.Identity, MultiModalEncoder]
    aug: tp.Union[utils.RandomShiftsAug, nn.Identity]
    # pylint: disable=unused-argument
    def __init__(self, meta_dim: int = 0, **kwargs: tp.Any) -> None:
        if self.__class__.__name__.startswith(("DIAYN", "APS", "RND", "Proto", "ICMAPT", "MaxEnt")):  # HACK
            cfg_fields = {field.name for field in dataclasses.fields(PPOAgentConfig)}
            # those have their own config, so lets curate the fields
            # others will need to be ported in time
            kwargs = {x: y for x, y in kwargs.items() if x in cfg_fields}

        cfg = PPOAgentConfig(**kwargs)
        self.cfg = cfg
        self.action_dim = cfg.action_shape[0]
        self.solved_meta = None

        # self.update_encoder = update_encoder  # used in subclasses
        _shape_total = 0
        for compo_name in cfg.obs_shape.keys():
            shape,dim = cfg.obs_shape[compo_name]
            _shape_total+=dim
        if 'toy' in cfg.obs_type and not cfg.use_sequence:
            self.aug = nn.Identity()
            self.encoder = nn.Identity()
            self.obs_dim = _shape_total + meta_dim
        else:
            self.aug = nn.Identity()
            self.encoder = MultiModalEncoder(cfg.obs_shape,cfg.encoder_config) 
            #TODO get parameter ValueError('optimizer got an empty parameter list')
            self.encoder.to_device(cfg.device)
            self.obs_dim = self.encoder.repr_dim + meta_dim
        # for sequence input case
        if cfg.use_sequence:
            self.encoder = SequenceEncoder(_shape_total, self.encoder,cfg.sequence_encoder_config).to(cfg.device)
            
        self.actor = Actor(cfg.obs_type, self.obs_dim, self.action_dim,
                           cfg.feature_dim, cfg.hidden_dim).to(cfg.device)

        self.critic: nn.Module = Critic(cfg.obs_type, self.obs_dim, self.action_dim,
                                        cfg.feature_dim, cfg.hidden_dim).to(cfg.device)
        self.critic_target: nn.Module = Critic(cfg.obs_type, self.obs_dim, self.action_dim,
                                               cfg.feature_dim, cfg.hidden_dim).to(cfg.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        # optimizers

        self.encoder_opt: tp.Optional[torch.optim.Adam] = None

        if 'toy' not in cfg.obs_type or cfg.use_sequence:
            self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=cfg.lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg.lr)

        self.reward_model: tp.Optional[torch.nn.Module] = None
        self.reward_opt: tp.Optional[torch.optim.Adam] = None
        if self.reward_free:
            self.reward_model = mlp(self.obs_dim, cfg.hidden_dim, "ntanh", cfg.hidden_dim,  # type: ignore
                                    "relu", cfg.hidden_dim, "relu", 1).to(cfg.device)  # type: ignore
            self.reward_opt = torch.optim.Adam(self.reward_model.parameters(), lr=1e-3)

        self.train()
        self.critic_target.train()

    def __getattr__(self, name: str) -> tp.Any:
        # LEGACY: allow accessing the config directly as attribute
        # to avoid having to rewrite everything at once
        # cost: less type safety
        if "cfg" in self.__dict__:
            return getattr(self.cfg, name)
        raise AttributeError
    
    def train(self, training: bool = True) -> None:
        self.training = training
        if training:
            self.encoder.train()
            self.actor.train()
            self.critic.train()
        else:
            self.encoder.eval()
            self.actor.eval()
            self.critic.eval()

    def act(self, obs):


    def update(self, replay_loader: ReplayBuffer, step: int) -> tp.Dict[str, float]:
        metrics: tp.Dict[str, float] = {}

        return metrics