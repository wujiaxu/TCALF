# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from pathlib import Path
import pdb  # pylint: disable=unused-import
import typing as tp
import dataclasses
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict

from url_benchmark import utils
from hydra.core.config_store import ConfigStore
import omegaconf

from url_benchmark.dmc import TimeStep
from .ddpg import DDPGAgent, MetaDict, DDPGAgentConfig,Actor,Critic
from controllable_navi.in_memory_replay_buffer import ReplayBuffer
from typing import Any, Dict, Tuple
from .crowd_aps import CriticSF,APS
from controllable_navi.agent.lagrange import Lagrange
from torch.utils.tensorboard import SummaryWriter

# TODO(HL): how to include GPI for continuous domain?


@dataclasses.dataclass
class GD_APSAgentConfig(DDPGAgentConfig):
    _target_: str = "controllable_navi.agent.GD_aps.APSAgent" 
    name: str = "gd_aps"
    update_encoder: bool = omegaconf.II("update_encoder")
    sf_dim: int = 10
    update_task_every_step: int = 5
    knn_rms: bool = True
    knn_k: int = 12
    knn_avg: bool = True
    knn_clip: float = 0.0001
    num_init_steps: int = 4096  # set to ${num_train_frames} to disable finetune policy parameters
    lstsq_batch_size: int = 4096
    num_inference_steps: int = 10000
    balancing_factor: float = 1.0
    use_constraint: bool = True
    init_constraint_value: float = 10
    max_constraint_value: float = 40
    dynamic_contrain_step: int = 2000010
    constraint_on: str = "Qsf"
    lagrangian_k_p: float = 0.0003
    lagrangian_k_i: float = 0.0003
    lagrangian_k_d: float = 0.0003
    lagrange_multiplier_upper_bound: float = 0.01
    lagrange_update_interval: int = 1
    use_self_supervised_encoder: bool = True
    self_supervised_encoder:str = "IDP"
    sse_dim: int = 128

cs = ConfigStore.instance()
cs.store(group="agent", name="gd_aps", node=GD_APSAgentConfig)

class IDP(nn.Module):
    # inverse dynamic presentation for embedding obs to controllable state
    def __init__(self, obs_dim, action_dim, hidden_dim) -> None:
        super().__init__()

        self.embedding_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim))

        self.backward_net = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim),
                                          nn.ReLU(),
                                          nn.Linear(hidden_dim, action_dim),
                                          nn.Tanh())

        self.apply(utils.weight_init)

    def forward(self, obs, action, next_obs) -> Tuple[Any, Any, Any]:
        assert obs.shape[0] == next_obs.shape[0]
        assert obs.shape[0] == action.shape[0]

        f_obs = self.embedding_net(obs)
        f_next_obs = self.embedding_net(next_obs)
        action_hat = self.backward_net(torch.cat([f_obs, f_next_obs], dim=-1))

        backward_error = torch.norm(action - action_hat,
                                    dim=-1,
                                    p=2,
                                    keepdim=True)

        return f_obs, f_next_obs, backward_error
    
    def embed(self,obs):
        return self.embedding_net(obs)
    
class APSAgent(DDPGAgent):
    def __init__(self, **kwargs: tp.Any) -> None:
        
        cfg = GD_APSAgentConfig(**kwargs)

        # create actor and critic
        # increase obs shape to include task dim (through meta_dim)
        super().__init__(**kwargs, meta_dim=cfg.sf_dim)
        self.cfg: GD_APSAgentConfig = cfg  # override base ddpg cfg type
        
        if self.cfg.use_self_supervised_encoder:
            self.sse_dim = self.cfg.sse_dim
            if self.cfg.self_supervised_encoder=="IDP":
                # inverse dynamic feature embedding
                self.sse = IDP(self.obs_dim - self.sf_dim,self.action_dim,self.sse_dim).to(kwargs['device'])
                self.sse_opt = torch.optim.Adam(self.sse.parameters(), lr=self.lr)
            else:
                raise NotImplementedError
        else:
            self.sse_dim = self.obs_dim - self.sf_dim
            self.sse = nn.Identity()

        self.actor = Actor('states', self.sse_dim + self.sf_dim, self.action_dim,
                           cfg.feature_dim, cfg.hidden_dim).to(cfg.device)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr)
        # overwrite critic with critic sf
        # # input of critic sf is idp feature with dim=cfg.hidden_dim
        self.critic = CriticSF('states', self.sse_dim + self.sf_dim, self.action_dim,
                               self.feature_dim, self.hidden_dim,
                               self.sf_dim).to(self.device)
        self.critic_target = CriticSF('states', self.sse_dim + self.sf_dim,
                                      self.action_dim, self.feature_dim,
                                      self.hidden_dim,
                                      self.sf_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_opt = torch.optim.Adam(self.critic.parameters(),
                                           lr=self.lr)
        
        self.critic_goal = Critic('states', self.sse_dim, self.action_dim,
                               self.feature_dim, self.hidden_dim).to(self.device)
        self.critic_goal_target = Critic('states', self.sse_dim,
                                      self.action_dim, self.feature_dim,
                                      self.hidden_dim).to(self.device)
        self.critic_goal_target.load_state_dict(self.critic_goal.state_dict())
        self.critic_goal_opt = torch.optim.Adam(self.critic_goal.parameters(),
                                           lr=self.lr)

        # aps is denoted as phi in the original paper
        # input of aps is idp feature with dim=cfg.hidden_dim
        self.aps = APS(self.sse_dim*2+self.action_dim, self.sf_dim,
                       kwargs['hidden_dim']).to(kwargs['device'])
        self.aps_opt = torch.optim.Adam(self.aps.parameters(), lr=self.lr)
        
        # particle-based entropy
        rms = utils.RMS(self.device)
        self.pbe = utils.PBE(rms, cfg.knn_clip, cfg.knn_k, cfg.knn_avg, cfg.knn_rms,
                             cfg.device)
        
        self.balancing_factor = cfg.balancing_factor
        self.constrain_value = cfg.init_constraint_value
        self.lagrange = Lagrange(cfg.balancing_factor,cfg.lagrange_multiplier_upper_bound,cfg.lagrangian_k_p,cfg.lagrangian_k_i,cfg.lagrangian_k_d)
        self.update_lagrange_every_steps = self.update_every_steps * cfg.lagrange_update_interval
        
        self.train()
        self.critic_target.train()
        self.aps.train()
        self.critic_goal.train()
        self.critic_goal_target.train()
        if self.cfg.use_self_supervised_encoder:
            self.sse.train()

    #TODO debug
    """
    RuntimeError: Tracer cannot infer type of TruncatedNormal(loc: torch.Size([1, 2]), scale: torch.Size([1, 2]))
    :Only tensors and (possibly nested) tuples of tensors, lists, or dictsare supported as inputs or outputs of traced functions, 
    but instead got value of type TruncatedNormal.
    """
    def add_to_tb(self,writer:SummaryWriter)->None:
        dummy_input = torch.randn(1, self.sse_dim + self.sf_dim).to(self.cfg.device)
        writer.add_graph(self.actor,(dummy_input,dummy_input[:,:1]))
        writer.add_graph(self.critic,(dummy_input,dummy_input[:,:2]))
        writer.add_graph(self.critic_target,(dummy_input,dummy_input[:,:2]))
        dummy_input = torch.randn(1, self.sse_dim).to(self.cfg.device)
        writer.add_graph(self.critic_goal,(dummy_input,dummy_input[:,:2]))
        writer.add_graph(self.critic_goal_target,(dummy_input,dummy_input[:,:2]))
        dummy_input = torch.randn(1,self.sse_dim*2+self.action_dim).to(self.cfg.device)
        writer.add_graph(self.aps,dummy_input)
        if self.cfg.use_self_supervised_encoder:
            dummy_input = torch.randn(1,self.obs_dim - self.sf_dim).to(self.cfg.device)
            writer.add_graph(self.sse,dummy_input)
        return 
    
    def act(self, obs, meta, step, eval_mode) -> np.ndarray:
        if self.cfg.use_sequence:
            seq,mask = obs
            seq = torch.as_tensor(seq, device=self.device).unsqueeze(0)
            mask = torch.as_tensor(mask, device=self.device).unsqueeze(0)
            h = self.encoder(seq,mask)
        else:
            obs = torch.as_tensor(obs, device=self.device).unsqueeze(0)
            h = self.encoder(obs)
        # if self.cfg.use_self_supervised_encoder:
        #     h = self.sse.embed(h) #<--used only for state entropy estimation
        inputs = [h]
        for value in meta.values():
            value = torch.as_tensor(value, device=self.device).unsqueeze(0)
            inputs.append(value)
        inpt = torch.cat(inputs, dim=-1)
        #assert obs.shape[-1] == self.obs_shape[-1]
        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(inpt, stddev)
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample(clip=None)
            if step < self.num_expl_steps:
                action.uniform_(-1.0, 1.0)
        return action.cpu().numpy()[0]

    def init_meta(self) -> tp.Dict[str, np.ndarray]:
        if self.solved_meta is not None:
            return self.solved_meta
        task = torch.randn(self.sf_dim)
        task = task / torch.norm(task)
        task_array = task.cpu().numpy()
        meta = OrderedDict()
        meta['task'] = task_array
        return meta
    
    def update_constraint_value(self,step,new_value=None):
        if new_value:
            self.constrain_value = new_value
        else:
            self.constrain_value = self.cfg.init_constraint_value\
                                    +step*(self.cfg.max_constraint_value-self.cfg.init_constraint_value)/4000010 #TODO 

    # pylint: disable=unused-argument
    def update_meta(
        self,
        meta: MetaDict,
        global_step: int,
        time_step: TimeStep,
        finetune: bool = False,
        replay_loader: tp.Optional[ReplayBuffer] = None
    ) -> MetaDict:
        if global_step % self.update_task_every_step == 0:
            return self.init_meta()
        return meta

    def update_aps(self, task, obs, action, next_obs, step) -> Dict[str, Any]:
        metrics: tp.Dict[str, float] = {}

        loss = self.compute_aps_loss(obs, action, next_obs, task)

        self.aps_opt.zero_grad(set_to_none=True)
        if self.encoder_opt is not None:
            self.encoder_opt.zero_grad(set_to_none=True)
        loss.backward()
        self.aps_opt.step()
        if self.encoder_opt is not None:
            self.encoder_opt.step()

        if self.use_tb or self.use_wandb:
            metrics['aps_loss'] = loss.item()

        return metrics

    def compute_intr_reward(self, task, obs, action, next_obs, step) -> Tuple[Any, Any]:
        # maxent reward
        with torch.no_grad():
            _in = torch.cat([obs, action, next_obs], dim=1)
            rep = self.aps(_in, norm=False)
        
        # Encoding
        if self.cfg.use_self_supervised_encoder:
            f_next_obs = self.sse.embed(next_obs)
        else:
            f_next_obs = next_obs #~5.28 next_obs, but original paper used rep
        reward = self.pbe(f_next_obs) #not encoded? hahahaha
        intr_ent_reward = reward.reshape(-1, 1)

        # successor feature reward
        rep = rep / torch.norm(rep, dim=1, keepdim=True)
        intr_sf_reward = torch.einsum("bi,bi->b", task, rep).reshape(-1, 1)

        return intr_ent_reward, intr_sf_reward

    def compute_aps_loss(self, obs, action, next_obs, task) -> Any:
        """MLE loss"""
        _in = torch.cat([obs, action, next_obs], dim=1)
        loss = -torch.einsum("bi,bi->b", task, self.aps(_in)).mean()
        return loss

    def update(self, replay_loader: ReplayBuffer, step: int) -> tp.Dict[str, float]:
        metrics: tp.Dict[str, float] = {}

        if step % self.update_every_steps != 0:
            return metrics

        if self.cfg.use_sequence:
            batch = replay_loader.sample_sequence(self.cfg.batch_size).to(self.device)
            obs, obs_mask, action, extr_reward, discount, next_obs, next_obs_mask = batch.unpack_with_mask()
            # augment and encode
            obs = self.aug(obs)
            obs = self.encoder(obs,obs_mask)
            next_obs = self.aug(next_obs)
            next_obs = self.encoder(next_obs,next_obs_mask)
        else:
            batch = replay_loader.sample(self.cfg.batch_size).to(self.device)

            obs, action, extr_reward, discount, next_obs = batch.unpack()

            # augment and encode
            obs = self.aug_and_encode(obs)
            next_obs = self.aug_and_encode(next_obs)
        task = batch.meta["task"]

        if self.cfg.use_self_supervised_encoder:
            metrics.update(self.update_sse(obs.detach(),action,next_obs.detach()))

        if self.reward_free:
            # freeze successor features at finetuning phase
            metrics.update(self.update_aps(task, obs,action,next_obs, step))

            with torch.no_grad():
                intr_ent_reward, intr_sf_reward = self.compute_intr_reward(
                    task, obs, action, next_obs, step)
                intr_reward = intr_ent_reward + intr_sf_reward

            if self.use_tb or self.use_wandb:
                metrics['intr_reward'] = intr_reward.mean().item()
                metrics['intr_ent_reward'] = intr_ent_reward.mean().item()
                metrics['intr_sf_reward'] = intr_sf_reward.mean().item()
            
            reward = intr_reward

        else:
            reward = extr_reward

        if self.use_tb or self.use_wandb:
            metrics['extr_reward'] = extr_reward.mean().item()
            metrics['batch_reward'] = reward.mean().item()

        # if not self.update_encoder:
        obs = obs.detach()
        next_obs = next_obs.detach()

        metrics.update(
            self.update_critic_goal(obs, action, extr_reward, discount,
                               next_obs, task, step))

        # extend observations with task
        # f_obs = torch.cat([f_obs, task], dim=1)
        # f_next_obs = torch.cat([f_next_obs, task], dim=1)

        # update critic
        metrics.update(
            self.update_critic(obs, action, intr_reward, discount,
                               next_obs, task, step))

        # update actor
        metrics.update(self.update_actor(obs, task, step))

        # update critic target
        utils.soft_update_params(self.critic, self.critic_target,
                                 self.critic_target_tau)
        utils.soft_update_params(self.critic_goal, self.critic_goal_target,
                                 self.critic_target_tau)

        return metrics

    @torch.no_grad()
    def regress_meta(self, replay_loader, step):
        obs, action, reward,next_obs = [], [],[], []
        batch_size = 0
        while batch_size < self.lstsq_batch_size:
            batch = replay_loader.sample(self.cfg.batch_size)
            #obs, action, extr_reward, discount, next_obs
            batch_obs, batch_action, batch_reward, _, batch_next_obs = utils.to_torch(batch, self.device)
            obs.append(batch_obs)
            action.append(batch_action)
            reward.append(batch_reward)
            next_obs.append(batch_next_obs)
            batch_size += batch_obs.size(0)
        obs, reward = torch.cat(obs, 0), torch.cat(reward, 0)
        action, next_obs = torch.cat(action, 0), torch.cat(next_obs, 0)

        obs = self.aug_and_encode(obs)
        next_obs = self.aug_and_encode(next_obs)

        _in = torch.cat([obs, action, next_obs], dim=1)
        rep = self.aps(_in)
        task = torch.linalg.lstsq(reward, rep)[0][:rep.size(1), :][0]
        task = task / torch.norm(task)
        task = task.cpu().numpy()
        meta = OrderedDict()
        meta['task'] = task

        # save for evaluation
        self.solved_meta = meta
        return meta

    @torch.no_grad()
    def infer_meta(self, replay_loader: ReplayBuffer) -> MetaDict:
        obs_list, reward_list = [], []
        action_list, next_obs_list = [], []
        batch_size = 0
        while batch_size < self.cfg.num_inference_steps:
            batch = replay_loader.sample(self.cfg.batch_size)
            batch = batch.to(self.cfg.device)
            obs_list.append(batch.obs)
            reward_list.append(batch.reward)
            next_obs_list.append(batch.next_obs)
            action_list.append(batch.action)
            batch_size += batch.next_obs.size(0)
        obs, reward = torch.cat(obs_list, 0), torch.cat(reward_list, 0)  # type: ignore
        obs, reward = obs[:self.cfg.num_inference_steps], reward[:self.cfg.num_inference_steps]
        next_obs, action = torch.cat(next_obs_list, 0), torch.cat(action_list, 0)  # type: ignore
        next_obs, action = next_obs[:self.cfg.num_inference_steps], action[:self.cfg.num_inference_steps]
        return self.infer_meta_from_obs_and_rewards(obs, action, reward,next_obs)

    @torch.no_grad()
    def infer_meta_from_obs_and_rewards(self, obs: torch.Tensor, 
                                        action: torch.Tensor, 
                                        reward: torch.Tensor,
                                        next_obs: torch.Tensor,
                                        obs_mask: tp.Optional[torch.Tensor]=None,
                                        next_obs_mask: tp.Optional[torch.Tensor]=None) -> MetaDict:
        print('max reward: ', reward.max().cpu().item())
        print('99 percentile: ', torch.quantile(reward, 0.99).cpu().item())
        print('median reward: ', reward.median().cpu().item())
        print('min reward: ', reward.min().cpu().item())
        print('mean reward: ', reward.mean().cpu().item())
        print('num reward: ', reward.shape[0])

        if obs_mask and next_obs_mask:
            # augment and encode
            obs = self.aug(obs)
            obs = self.encoder(obs,obs_mask)
            next_obs = self.aug(next_obs)
            next_obs = self.encoder(next_obs,next_obs_mask)
        else:
            obs = self.aug_and_encode(obs)
            next_obs = self.aug_and_encode(next_obs)

        _in = torch.cat([obs, action, next_obs], dim=1)

        rep = self.aps(_in)
        # task = torch.linalg.lstsq(reward, rep)[0][:rep.size(1), :][0]
        task = torch.linalg.lstsq(rep, reward)[0].squeeze()
        task = task / torch.norm(task)
        task = task.cpu().numpy()
        meta = OrderedDict()
        meta['task'] = task

        # self.solved_meta = meta
        return meta

    def update_sse(self,obs,action,next_obs):
        metrics: tp.Dict[str, float] = {}

        _,_,sse_loss = self.sse(obs,action,next_obs)
        loss = sse_loss.mean()

        self.sse_opt.zero_grad(set_to_none=True)
        loss.backward()
        self.sse_opt.step()

        if self.use_tb or self.use_wandb:
            metrics['sse_loss'] = loss.item()

        return metrics
    
    def update_critic(self, obs, action, reward, discount, next_obs, task,
                      step) -> Dict[str, Any]:
        """diff is critic takes task as input"""
        metrics: tp.Dict[str, float] = {}
        
        obs = torch.cat([obs, task], dim=1)
        next_obs = torch.cat([next_obs, task], dim=1)

        with torch.no_grad():
            stddev = utils.schedule(self.stddev_schedule, step)
            dist = self.actor(next_obs, stddev)
            next_action = dist.sample(clip=self.stddev_clip)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action,
                                                      task)
            target_V = torch.min(target_Q1, target_Q2)
            target_Q = reward + (discount * target_V)

        # print(action.shape) -> torch.Size([1024, 1])? why?
        Q1, Q2 = self.critic(obs, action, task)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        if self.use_tb or self.use_wandb:
            metrics['critic_target_q'] = target_Q.mean().item()
            metrics['critic_q1'] = Q1.mean().item()
            metrics['critic_q2'] = Q2.mean().item()
            metrics['critic_loss'] = critic_loss.item()

        # update balancing_factor
        if self.cfg.use_constraint:
            self.update_constraint_value(step)
        if step % self.update_lagrange_every_steps ==0 and self.cfg.use_constraint:
            cost_limit = -self.constrain_value
            cost = -target_Q.mean().item() #-intr_sf_reward.mean().cpu().item()
            self.lagrange.update_lagrange_multiplier(cost,cost_limit)
            self.balancing_factor = self.lagrange.lagrangian_multiplier.to(self.cfg.device)
            if self.use_tb or self.use_wandb:
                metrics['lagrange_multiplier'] = self.lagrange.lagrangian_multiplier
                metrics['cost'] = cost
                metrics['cost_limit'] = cost_limit

        # optimize critic
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()

        return metrics
    
    def update_critic_goal(self, obs, action, reward, discount, next_obs, task,
                      step) -> Dict[str, Any]:
        """diff is critic takes task as input"""
        metrics: tp.Dict[str, float] = {}
        # print(obs.shape) -> torch.Size([1024, 4112])

        with torch.no_grad():
            stddev = utils.schedule(self.stddev_schedule, step)
            dist = self.actor(torch.cat([next_obs, task], dim=1), stddev)
            next_action = dist.sample(clip=self.stddev_clip)

            target_goal_Q1, target_goal_Q2 = self.critic_goal_target(next_obs, next_action)
            target_goal_V = torch.min(target_goal_Q1, target_goal_Q2)
            target_goal_Q = reward + (discount * target_goal_V)


        Q_goal_1, Q_goal_2 = self.critic_goal(obs, action)
        critic_goal_loss = F.mse_loss(Q_goal_1, target_goal_Q) + F.mse_loss(Q_goal_2, target_goal_Q)

        if self.use_tb or self.use_wandb:
            metrics['critic_goal_target_q'] = target_goal_Q.mean().item()
            metrics['critic_goal_q1'] = Q_goal_1.mean().item()
            metrics['critic_goal_q2'] = Q_goal_2.mean().item()
            metrics['critic_goal_loss'] = critic_goal_loss.item()

        # optimize critic
        self.critic_goal_opt.zero_grad(set_to_none=True)
        critic_goal_loss.backward()
        self.critic_goal_opt.step()

        return metrics

    def update_actor(self, obs, task, step) -> Dict[str, Any]:
        """diff is critic takes task as input"""
        metrics: tp.Dict[str, float] = {}

        obs_z = torch.cat([obs, task], dim=1)
        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs_z, stddev)
        action = dist.sample(clip=self.stddev_clip)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        Q1, Q2 = self.critic(obs_z, action, task)
        Q = torch.min(Q1, Q2)
        Q_goal_1, Q_goal_2 = self.critic_goal(obs, action)
        Q_goal = torch.min(Q_goal_1, Q_goal_2)

        actor_loss = (-self.balancing_factor*Q.mean()-Q_goal.mean())/(1+self.balancing_factor)

        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        if self.use_tb or self.use_wandb:
            metrics['actor_loss'] = actor_loss.item()
            metrics['actor_logprob'] = log_prob.mean().item()
            metrics['actor_ent'] = dist.entropy().sum(dim=-1).mean().item()

        return metrics
