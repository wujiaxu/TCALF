import typing as tp
import dataclasses
from hydra.core.config_store import ConfigStore
import omegaconf
import numpy as np
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F

from robust_navi.rl.ppo import PPO,PPOAgentConfig
from robust_navi.rl.networks.encoders import EncoderConfig,MultiModalEncoder
from robust_navi.rl.networks.actor_critic import RecurrentActorCriticPolicy
from robust_navi.rl.networks.network_utils import init,reshapeT,zip_strict

@dataclasses.dataclass
class ScanPPOConfig(PPOAgentConfig):
    name: str = "diverse_recurrent_ppo"
    feature_dim: int = 256
    hidden_dim: int = 256
    encoder_config:EncoderConfig=EncoderConfig()
    seq_length: int = omegaconf.II("num_steps")


cs = ConfigStore.instance()
cs.store(group="agent", name="scan_ppo", node=ScanPPOConfig)

class ScanActorCritic(nn.Module):
    def __init__(self,obs_shape,cfg:ScanPPOConfig):
        super(ScanActorCritic, self).__init__()
        self.is_recurrent = True
        self.recurrent_hidden_state_size = cfg.feature_dim
        self.output_size = cfg.hidden_dim
        self.nenv = cfg.num_processes
        self.seq_length = cfg.seq_length
        self.nminibatch = cfg.num_mini_batch

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))
        
        self.encoder = MultiModalEncoder(obs_shape,cfg.encoder_config) 

        self.gru = nn.GRU(self.encoder.repr_dim,self.recurrent_hidden_state_size)

        self.actor = nn.Sequential(
            init_(nn.Linear(self.recurrent_hidden_state_size, cfg.hidden_dim)), nn.Tanh(),
            init_(nn.Linear(cfg.hidden_dim, cfg.hidden_dim)), nn.Tanh())

        self.critic = nn.Sequential(
            init_(nn.Linear(self.recurrent_hidden_state_size, cfg.hidden_dim)), nn.Tanh(),
            init_(nn.Linear(cfg.hidden_dim, cfg.hidden_dim)), nn.Tanh())


        self.critic_linear = init_(nn.Linear(cfg.hidden_dim, 1))

    def eval_mode(self):
        self._mode="eval"
        self.encoder.eval()
        self.gru.eval()
        self.actor.eval()
        self.critic.eval()
        self.critic_linear.eval()

    def train_mode(self):
        self._mode="train"
        self.encoder.train()
        self.gru.train()
        self.actor.train()
        self.critic.train()
        self.critic_linear.train()

    def forward(self,inputs:tp.Dict[str,torch.Tensor], rnn_hxs:torch.Tensor, masks:torch.Tensor, infer=False):
        """
        inputs: 
            -train_mode:
                -act:(1 X nenv X input_shape)                
                -update:(seq_len X batch_size X input_shape)
            -eval_mode:(1 X 1 X input_shape)
        rnn_hxs: (1,nenv,h)
        masks:(seq_length,nenv)
        """

        if infer:
            # Test/rollout time
            seq_length = 1
            if self._mode=="eval":
                nenv = 1
            else:
                nenv = self.nenv

        else:
            # Training time
            seq_length = self.seq_length
            nenv = self.nenv // self.nminibatch
        
        states = self.encoder(inputs)
        states = reshapeT(states,seq_length,nenv) # (seq_length,nenv,input_shape)
        masks = masks.view(seq_length,nenv,1)
        rnn_hxs = reshapeT(rnn_hxs,1,nenv)


        gru_output = []
        # Iterate over the sequence
        for state, episode_start_mask in zip_strict(states, masks):
            rnn_hxs, gru_state = self.gru(
                state.unsqueeze(dim=0),
                # Reset the states at the beginning of a new episode
                episode_start_mask.unsqueeze(dim=0) * rnn_hxs,

            )
            gru_output += [rnn_hxs]
        gru_output = torch.cat(gru_output) #(seq_length,nenv,hidden_shape)
        
        hidden_critic = self.critic(gru_output)
        hidden_actor = self.actor(gru_output)
        if infer:
            return (
                    self.critic_linear(hidden_critic).squeeze(0), 
                    hidden_actor.squeeze(0), gru_state.squeeze(0)
                    )
        else:
            return (
                    self.critic_linear(hidden_critic).view(-1, 1), 
                    hidden_actor.view(-1, self.output_size), 
                    gru_output.view(seq_length*nenv,-1)
                    )
        

class ScanPPO(PPO):

    def __init__(self, obs_shape, 
                 output_dim,
                 cfg:ScanPPOConfig):
        base = ScanActorCritic(obs_shape,cfg)
        _actor_critic = RecurrentActorCriticPolicy(base,output_dim)
        super().__init__(_actor_critic,cfg)