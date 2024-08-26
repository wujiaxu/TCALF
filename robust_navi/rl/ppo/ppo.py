import torch
import torch.nn as nn
import torch.optim as optim
from hydra.core.config_store import ConfigStore
import dataclasses
import omegaconf
import typing as tp

from robust_navi.common.utils import hard_update_params
from robust_navi.rl.networks.actor_critic import RecurrentActorCriticPolicy

@dataclasses.dataclass
class PPOAgentConfig:
    _target_: str = "robust_navi.rl.ppo.PPO"
    name: str = "PPO"
    clip_param: float = 0.2
    ppo_epoch: int = 5
    num_processes: int = omegaconf.II("num_processes")
    num_mini_batch: int = 2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    lr: float = 1e-5
    eps: float = 1e-5
    max_grad_norm: float = 0.5
    use_clipped_value_loss:bool = True
    use_gae: bool = True
    gae_lambda: bool = True
    discount:float=omegaconf.II("discount")
    device: str = omegaconf.II("device")

cs = ConfigStore.instance()
cs.store(group="agent", name="ppo", node=PPOAgentConfig)


class PPO():
    """ Class for the PPO optimizer """
    def __init__(self,
                 actor_critic:RecurrentActorCriticPolicy,
                 cfg:PPOAgentConfig):

        self.actor_critic:RecurrentActorCriticPolicy = actor_critic
        self.actor_critic.to(cfg.device)
        
        self.clip_param = cfg.clip_param
        self.ppo_epoch = cfg.ppo_epoch
        self.num_mini_batch = cfg.num_mini_batch

        self.value_loss_coef = cfg.value_loss_coef
        self.entropy_coef = cfg.entropy_coef

        self.max_grad_norm = cfg.max_grad_norm
        self.use_clipped_value_loss = cfg.use_clipped_value_loss

        self.optimizer = optim.Adam(actor_critic.parameters(), lr=cfg.lr, eps=cfg.eps)

    def train_mode(self):
        self.actor_critic.train_mode()
    
    def eval_mode(self):
        self.actor_critic.eval_mode()

    def _cal_action_loss(self,action_log_probs,old_action_log_probs_batch,adv_targ):
        ratio = torch.exp(action_log_probs -
                                    old_action_log_probs_batch)
        surr1 = ratio * adv_targ
        surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                            1.0 + self.clip_param) * adv_targ
        action_loss = -torch.min(surr1, surr2).mean()
        
        return action_loss
    
    def _cal_value_loss(self,return_batch,values,value_preds_batch):

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

        return value_loss
    
    def init_from(self, other) -> None:
        # copy parameters over
        hard_update_params(other.actor_critic, self.actor_critic)

    def update(self, rollouts):
        metrics: tp.Dict[str, float] = {}

        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-5)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0


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

                # Reshape to do in a single forward pass for all steps
                values, action_log_probs, dist_entropy, _ = self.actor_critic.evaluate_actions(
                    obs_batch, recurrent_hidden_states_batch, masks_batch,
                    actions_batch)

                action_loss = self._cal_action_loss(action_log_probs,old_action_log_probs_batch,adv_targ)
                value_loss = self._cal_value_loss(return_batch,values,value_preds_batch)

                
                self.optimizer.zero_grad()
                total_loss=value_loss * self.value_loss_coef + action_loss - dist_entropy * self.entropy_coef
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                         self.max_grad_norm)
                self.optimizer.step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()


        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates

        metrics["value_loss"] = value_loss_epoch
        metrics["action_loss"] = action_loss_epoch
        metrics["action_entropy"] = dist_entropy_epoch

        return metrics
