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

from TCALF.robust_navi.common.storage import DiscrimRolloutStorage
from controllable_navi import utils
from .crowd_aps import APS
from controllable_navi.agent.encoders import MultiModalEncoder,EncoderConfig
from controllable_navi.agent.sequence_encoders import SequenceEncoder,SequenceEncoderConfig,build_rnn_state_encoder
from controllable_navi.agent.distributions import DiagGaussian

MetaDict = tp.Mapping[str, np.ndarray]

@dataclasses.dataclass
class PPOAgentConfig:
    _target_: str = "controllable_navi.agent.ppo.PPOAgent"
    name: str = "ddpg"
    sf_dim: int = 5
    hidden_dim: int = 1024
    reward_free: bool = omegaconf.II("reward_free")
    intrinsic_reward_weight: float = 0.1
    obs_type: str = omegaconf.MISSING  # to be specified later
    obs_shape: dict = omegaconf.MISSING  # to be specified later
    action_shape: tp.Tuple[int, ...] = omegaconf.MISSING  # to be specified later
    encoder_config:EncoderConfig=EncoderConfig()
    sequence_encoder_config:SequenceEncoderConfig=SequenceEncoderConfig()
    device: str = omegaconf.II("device")
    lr: float = 1e-5
    use_gae: bool = True
    gae_lambda: float = 0.95
    ppo_epoch:int = 5
    use_clipped_value_loss: bool = True
    value_loss_coef: float = 0.5
    max_grad_norm: float = 0.5
    num_mini_batch: float = 10
    clip_param: float = 0.2

cs = ConfigStore.instance()
cs.store(group="agent", name="ppo", node=PPOAgentConfig)
   
class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim) -> None:
        super().__init__()

        self.trunk = nn.Sequential(nn.Linear(obs_dim, hidden_dim),
                                   nn.LayerNorm(hidden_dim), nn.Tanh())

        policy_layers = []
        policy_layers += [
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True)
        ]

        self.policy = nn.Sequential(*policy_layers)

        self.dist = DiagGaussian(hidden_dim, action_dim)

        self.apply(utils.weight_init)

    def evaluate_action(self,obs,action):

        h = self.trunk(obs)

        actor_features = self.policy(h)
        
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()
        return action_log_probs,dist_entropy


    def forward(self, obs,deterministic=False):
        h = self.trunk(obs)

        actor_features = self.policy(h)
        
        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)

        return action, action_log_probs

class Critic(nn.Module):
    def __init__(self, obs_dim, hidden_dim) -> None:
        super().__init__()

        # for states actions come in the beginning
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.Tanh())
        trunk_dim = hidden_dim

        def make_v():
            v_layers = []
            v_layers += [
                nn.Linear(trunk_dim, hidden_dim),
                nn.ReLU(inplace=True)
            ]
            
            v_layers += [nn.Linear(hidden_dim, 1)]
            return nn.Sequential(*v_layers)

        self.V = make_v()

        self.apply(utils.weight_init)

    def forward(self, obs) -> Tuple[Any, Any]:
        inpt = obs 
        h = self.trunk(inpt)

        v = self.V(h)

        return v
    
class PPOAgent:
    encoder: tp.Union[SequenceEncoder, nn.Identity, MultiModalEncoder]
    aug: tp.Union[utils.RandomShiftsAug, nn.Identity]
    # pylint: disable=unused-argument
    def __init__(self,**kwargs: tp.Any) -> None:

        cfg = PPOAgentConfig(**kwargs)
        self.cfg = cfg
        self.action_dim = cfg.action_shape[0]
        self.sf_dim = cfg.sf_dim
        self.solved_meta = None
        self.use_gae = cfg.use_gae
        self.gae_lambda = cfg.gae_lambda
        self.ppo_epoch = cfg.ppo_epoch
        self.entropy_coef = cfg.intrinsic_reward_weight
        self.use_clipped_value_loss = cfg.use_clipped_value_loss
        self.value_loss_coef = cfg.value_loss_coef 
        self.max_grad_norm = cfg.max_grad_norm
        self.num_mini_batch = cfg.num_mini_batch
        self.clip_param = cfg.clip_param
        
        self.aug = nn.Identity()
        self.encoder = MultiModalEncoder(cfg.obs_shape,cfg.encoder_config) 
        self.rnn = build_rnn_state_encoder(self.encoder.repr_dim,
                                           cfg.sequence_encoder_config.rnn.hidden_dim,
                                           cfg.sequence_encoder_config.rnn.rnn_type,
                                           cfg.sequence_encoder_config.rnn.num_layers)

        self.encoder.to_device(cfg.device)
        self.rnn.to_device(cfg.device)
        
        self.obs_dim = cfg.sequence_encoder_config.rnn.hidden_dim + self.sf_dim
            
        self.actor = Actor(self.obs_dim, self.action_dim,
                           cfg.hidden_dim).to(cfg.device)

        self.critic = Critic(self.obs_dim,cfg.hidden_dim)

        self.aps = APS(cfg.sequence_encoder_config.rnn.hidden_dim, self.sf_dim,
                       cfg.hidden_dim).to(cfg.device)
        

        # optimizers
        self.optimizer = torch.optim.Adam(list(self.encoder.parameters())+list(self.rnn.parameters())+list(self.actor.parameters())+list(self.critic.parameters()), lr=cfg.lr)
        self.aps_opt = torch.optim.Adam(self.aps.parameters(), lr=cfg.lr)

        self.train()
        self.aps.train()

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

    def init_meta(self) -> tp.Dict[str, np.ndarray]:
        if self.solved_meta is not None:
            return self.solved_meta
        task = torch.randn(self.sf_dim)
        task = task / torch.norm(task)
        task_array = task.cpu().numpy()
        meta = OrderedDict()
        meta['task'] = task_array
        return meta
    
    def get_value(self, obs, hidden_states,masks,meta):

        obs = obs.unsqueeze(0)
        meta = meta.unsqueeze(0)
        h = self.encoder(obs)
        x, h = self.rnn(h,hidden_states,masks)
        
        inputs = [x,meta]
        
        inpt = torch.cat(inputs, dim=-1)

        value = self.critic(inpt)

        return value
    
    def evaluate_actions(self, obs, hidden_states, masks,meta,actions):

        h = self.encoder(obs)
        x, h = self.rnn(h,hidden_states,masks)
       
        inputs = [x,meta]
        inpt = torch.cat(inputs, dim=-1)

        value = self.critic(inpt)
        action_log_probs,dist_entropy = self.actor.evaluate_action(inpt,actions)

        return value, action_log_probs, dist_entropy, h

    def act(self, obs, hidden_states,masks,meta,eval_mode=True):

        obs = obs.unsqueeze(0)
        meta = meta.unsqueeze(0)
        h = self.encoder(obs)
        x, h = self.rnn(h,hidden_states,masks)
       
        inputs = [x,meta]
        inpt = torch.cat(inputs, dim=-1)

        value = self.critic(inpt)
        action, action_log_probs = self.actor(inpt,eval_mode)

        return value, action, action_log_probs, h
    
    def compute_aps_loss(self, obs, task) -> Any:
        """MLE loss"""
        loss = -torch.einsum("bi,bi->b", task, self.aps(obs)).mean()
        return loss
    
    def update_aps(self, task, obs, step) -> tp.Dict[str, Any]:
        metrics: tp.Dict[str, float] = {}

        loss = self.compute_aps_loss(obs, task)

        self.aps_opt.zero_grad(set_to_none=True)
        
        loss.backward()
        self.aps_opt.step()

        if self.use_tb or self.use_wandb:
            metrics['aps_loss'] = loss.item()

        return metrics

    def update(self, rollouts:DiscrimRolloutStorage, step: int) -> tp.Dict[str, float]:
        metrics: tp.Dict[str, float] = {}

        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-5)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0

        for e in range(self.ppo_epoch):
            
            data_generator = rollouts.recurrent_generator(
                advantages, self.num_mini_batch)

            for sample in data_generator:
                obs_batch, recurrent_hidden_states_batch, actions_batch, metas_batch,\
                   value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, \
                        adv_targ = sample
                
                metrics.update(self.update_aps(metas_batch,recurrent_hidden_states_batch.detach(),step))

                # Reshape to do in a single forward pass for all steps
                values, action_log_probs, dist_entropy, _ = self.evaluate_actions(
                    obs_batch, recurrent_hidden_states_batch, masks_batch,metas_batch,
                    actions_batch)

                ratio = torch.exp(action_log_probs -
                                  old_action_log_probs_batch)
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                    1.0 + self.clip_param) * adv_targ
                action_loss = -torch.min(surr1, surr2).mean()

                if self.use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch + \
                        (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (
                        value_pred_clipped - return_batch).pow(2)
                    value_loss = 0.5 * torch.max(value_losses,
                                                 value_losses_clipped).mean()
                else:
                    value_loss = 0.5 * (return_batch - values).pow(2).mean()

                self.optimizer.zero_grad()
                total_loss=value_loss * self.value_loss_coef + action_loss - dist_entropy * self.entropy_coef
                total_loss.backward()
                nn.utils.clip_grad_norm_(list(self.encoder.parameters())+list(self.rnn.parameters())+list(self.actor.parameters())+list(self.critic.parameters()),
                                         self.max_grad_norm)
                self.optimizer.step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates

        if self.use_tb or self.use_wandb:
            metrics['value_loss_epoch'] = value_loss_epoch
            metrics['action_loss_epoch'] = action_loss_epoch
            metrics['dist_entropy_epoch'] = dist_entropy_epoch            

        return metrics
    
