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
from robust_navi.rl.networks.actor_critic import RecurrentActorCriticPolicy
from robust_navi.rl.networks.encoders import make_mlp
from robust_navi.rl.networks.network_utils import init,reshapeT,zip_strict

@dataclasses.dataclass
class CrowdAgentConfig(PPOAgentConfig):
    name: str = "diverse_recurrent_ppo"
    feature_dim: int = 256
    hidden_dim: int = 256
    seq_length: int = omegaconf.II("num_steps")
    intrinsic_reward_weight: float = 0.1
    crowd_meta_dim: int = omegaconf.II("crowd_meta_dim")


cs = ConfigStore.instance()
cs.store(group="crowd_policy", name="diverse_recurrent_ppo", node=CrowdAgentConfig)


class CrowdActorCritic(nn.Module):
    def __init__(self,input_dim,cfg:CrowdAgentConfig):
        super(CrowdActorCritic, self).__init__()
        self.is_recurrent = True
        self.crowd_meta_dim = cfg.crowd_meta_dim
        self.recurrent_hidden_state_size = cfg.feature_dim
        self.output_size = cfg.hidden_dim
        self.nenv = cfg.num_processes
        self.seq_length = cfg.seq_length
        self.nminibatch = cfg.num_mini_batch

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        self.gru = nn.GRU(input_dim,self.recurrent_hidden_state_size)

        self.actor = nn.Sequential(
            init_(nn.Linear(self.recurrent_hidden_state_size+self.crowd_meta_dim, cfg.hidden_dim)), nn.Tanh(),
            init_(nn.Linear(cfg.hidden_dim, cfg.hidden_dim)), nn.Tanh())

        self.critic = nn.Sequential(
            init_(nn.Linear(self.recurrent_hidden_state_size+self.crowd_meta_dim, cfg.hidden_dim)), nn.Tanh(),
            init_(nn.Linear(cfg.hidden_dim, cfg.hidden_dim)), nn.Tanh())


        self.critic_linear = init_(nn.Linear(cfg.hidden_dim, 1))

    def eval_mode(self):
        self._mode="eval"
        self.gru.eval()
        self.actor.eval()
        self.critic.eval()
        self.critic_linear.eval()

    def train_mode(self):
        self._mode="train"
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

        metas = inputs["crowd_preference"]
        states = inputs["full_state"]
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
        states = reshapeT(states,seq_length,nenv) # (seq_length,nenv,input_shape)
        metas = reshapeT(metas,seq_length,nenv) # (seq_length,nenv,meta_shape)
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
        
        outputs = torch.cat([gru_output,metas],dim=-1)
        hidden_critic = self.critic(outputs)
        hidden_actor = self.actor(outputs)
        if infer:
            return self.critic_linear(hidden_critic).squeeze(0), hidden_actor.squeeze(0), gru_state.squeeze(0)
        else:
            return self.critic_linear(hidden_critic).view(-1, 1), hidden_actor.view(-1, self.output_size), gru_output.view(seq_length*nenv,-1)
        
class CrowdBehaviorDiscriminator(nn.Module):

    def __init__(self, input_dim, feature_dim, hidden_dim):

        super().__init__()

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))
        
        self.state_feat_net = nn.Sequential(
            init_(nn.Linear(input_dim, hidden_dim)), nn.ReLU(),
            init_(nn.Linear(hidden_dim, hidden_dim)), nn.ReLU(),
            init_(nn.Linear(hidden_dim, feature_dim)))
        
    def forward(self, obs, norm=True):
        state_feat = self.state_feat_net(obs)
        state_feat = F.normalize(state_feat, dim=-1) if norm else state_feat
        return state_feat


class CrowdPPO(PPO):

    def __init__(self, input_dim, 
                 output_dim,
                 cfg:CrowdAgentConfig):
        self.human_num = int(output_dim/2)
        base = CrowdActorCritic(input_dim,cfg)
        _actor_critic = RecurrentActorCriticPolicy(base,output_dim)
        super().__init__(
                                _actor_critic,
                                cfg)
        self.discriminator = CrowdBehaviorDiscriminator(cfg.feature_dim,cfg.crowd_meta_dim,cfg.hidden_dim).to(cfg.device)
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=cfg.lr, eps=cfg.eps)
        
    def _cal_discriminator_loss(self, hidden_state, metas):
        """MLE loss"""
        loss = -torch.einsum("bi,bi->b", metas, self.discriminator(hidden_state)).mean()
        return loss
    
    def train_mode(self):
        self.actor_critic.train_mode()
        self.discriminator.train()
    
    def eval_mode(self):
        self.actor_critic.eval_mode()
        self.discriminator.eval()

    def update(self, rollouts):
        metrics: tp.Dict[str, float] = {}

        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-5)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0
        discriminator_loss_epoch = 0
        value_sum = 0

        for e in range(self.ppo_epoch):
            if self.actor_critic.is_recurrent:
                data_generator = rollouts.recurrent_generator(
                    advantages, self.num_mini_batch)
            else:
                data_generator = rollouts.feed_forward_generator(
                    advantages, self.num_mini_batch)
                
            for sample in data_generator:

                (
                    obs_batch, 
                    recurrent_hidden_states_batch, 
                    actions_batch, 
                    value_preds_batch, 
                    return_batch, 
                    masks_batch, 
                    old_action_log_probs_batch, 
                    adv_targ 
                ) = sample
                metas_batch = obs_batch["crowd_preference"]
                # Reshape to do in a single forward pass for all steps #TODO
                (
                    values, 
                    action_log_probs, 
                    dist_entropy, 
                    hidden_state 
                ) = self.actor_critic.evaluate_actions(
                    obs_batch, 
                    recurrent_hidden_states_batch,
                    actions_batch, 
                    masks_batch)

                value_sum+=values.mean().item()
                action_loss = self._cal_action_loss(action_log_probs,old_action_log_probs_batch,adv_targ)
                value_loss = self._cal_value_loss(return_batch,values,value_preds_batch)
                discriminator_loss = self._cal_discriminator_loss(hidden_state.detach(),metas_batch)
                
                self.optimizer.zero_grad()
                total_loss=value_loss * self.value_loss_coef + action_loss - dist_entropy * self.entropy_coef
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                         self.max_grad_norm)
                self.optimizer.step()

                self.d_optimizer.zero_grad()
                discriminator_loss.backward()
                self.d_optimizer.step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()
                discriminator_loss_epoch += discriminator_loss.item()

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates
        discriminator_loss_epoch /= num_updates

        metrics["{}_human_model_mean_value".format(self.human_num)] = value_sum/num_updates
        metrics["{}_human_model_value_loss".format(self.human_num)] = value_loss_epoch
        metrics["{}_human_model_action_loss".format(self.human_num)] = action_loss_epoch
        metrics["{}_human_model_action_entropy".format(self.human_num)] = dist_entropy_epoch
        metrics["{}_human_model_discriminator_loss".format(self.human_num)] = discriminator_loss_epoch

        return metrics