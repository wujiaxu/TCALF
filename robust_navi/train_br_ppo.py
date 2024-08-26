# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from pathlib import Path
import sys
import faulthandler

faulthandler.enable()

base = Path(__file__).absolute().parents[1]
# we need to add base repo to be able to import controllable_navi
# we need to add controllable_navi to be able to reload legacy checkpoints
for fp in [base, base / "robust_navi"]:
    assert fp.exists()
    if str(fp) not in sys.path:
        sys.path.append(str(fp))

import os
import json
import pdb  # pylint: disable=unused-import
import copy
import logging
import dataclasses
import typing as tp
import warnings
from pathlib import Path

warnings.filterwarnings('ignore', category=DeprecationWarning)


import hydra
from hydra.core.config_store import ConfigStore
import numpy as np
import torch
import omegaconf as omgcf

from robust_navi.common.storage import RolloutStorage, DiscrimRolloutStorage
from robust_navi.common import utils
from robust_navi.common.logger import Logger
from robust_navi.common.video import VideoRecorder
from robust_navi.agent.crowd_ppo import CrowdPPO
from robust_navi.robot_crowd_sim.vec_env.envs import make_vec_envs
from robust_navi.robot_crowd_sim.core.simulator import RobotCrowdSim
from robust_navi.robot_crowd_sim.core.monitor import InfoMonitor


logger = logging.getLogger(__name__)
torch.backends.cudnn.benchmark = True

# # # Config # # #

@dataclasses.dataclass
class Config:
    agent: tp.Any
    crowd_policy: tp.Any #="crowd_agent"
    crowd_sim_env: tp.Any 
    max_human_num: int = 5
    crowd_meta_dim: int = 10
    max_episode_length:int = 50
    num_processes:int=5
    num_steps: int=30

    # misc
    seed: int = 11
    device: str = "cuda"
    save_video: bool = False
    use_tb: bool = True
    discount: float = 0.99
    
    # eval
    num_eval_episodes: int = 10
    final_tests: int = 10
    # checkpoint
    snapshot_at: tp.Tuple[int, ...] = (1000, 2000, 5000, 8000, 10000, 15000)
    checkpoint_every: int = 1000
    load_model: tp.Optional[str] = None


@dataclasses.dataclass
class PretrainConfig(Config):
    # train settings
    num_train_frames: int = 20000
    # snapshot
    eval_every_frames: int = 1000


# loaded as base_pretrain in pretrain.yaml
# we keep the yaml since it's easier to configure plugins from it
ConfigStore.instance().store(name="workspace_config", node=PretrainConfig)
# # # Implem # # #

C = tp.TypeVar("C", bound=Config) # Can be any subtype of Config


class BaseWorkspace(tp.Generic[C]):
    def __init__(self, cfg: C) -> None:
        self.work_dir = Path.cwd()
        print(f'Workspace: {self.work_dir}')
        print(f'Running code in : {Path(__file__).parent.resolve().absolute()}')
        logger.info(f'Workspace: {self.work_dir}')
        logger.info(f'Running code in : {Path(__file__).parent.resolve().absolute()}')

        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        if not torch.cuda.is_available():
            if cfg.device != "cpu":
                logger.warning(f"Falling back to cpu as {cfg.device} is not available")
                cfg.device = "cpu"
                cfg.agent.device = "cpu"
        self.device = torch.device(cfg.device)

        self.human_nums = [] #number of humans for each env
        self.train_env = self._make_vec_envs()
        self.eval_env = self._make_env(phase='val')
        self.phase = 'train'
        
        # create agent
        self.crowd_agent:tp.Dict[int,CrowdPPO] = {}
        for i in range(1,cfg.max_human_num+1):
            cfg.crowd_policy.num_processes = self.human_nums.count(i)
            self.crowd_agent[i] = CrowdPPO(7+8*i,2*i,cfg.crowd_policy)
        self.self_agent = None # TODO

        self.agent_monitor = InfoMonitor(cfg.num_processes,self.human_nums,0.25)
        # create logger
        self.logger = Logger(self.work_dir,
                             use_tb=cfg.use_tb)

        # for train PPO 
        self.rollouts_self_agent = RolloutStorage(
            cfg.num_steps,
            cfg.num_processes,
            self.train_env.observation_space,
            self.train_env.action_space,
            {"observations":["robot_state","robot_scan"],
            "action":"robot_action"},
            self.cfg.crowd_policy.hidden_dim #TODO
        )
        self.rollouts_crowd_agent:tp.Dict[int,DiscrimRolloutStorage] = {}
        for i in range(1,cfg.max_human_num+1):
            nenv_i_human = self.human_nums.count(i)
            observation_space = copy.deepcopy(self.train_env.observation_space)
            observation_space["full_state"] = (7+8*i,)
            action_space = copy.deepcopy(self.train_env.action_space)
            action_space["crowd_action"] = (2*i,)
            self.rollouts_crowd_agent[i] = DiscrimRolloutStorage(
                              self.crowd_agent[i].discriminator,
                              self.cfg.crowd_policy.intrinsic_reward_weight,
                              cfg.num_steps,
							  nenv_i_human, #need get from _make_vec_envs() function
							  observation_space,
							  action_space,
                              {"observations":["full_state","crowd_preference"],
                               "action":"crowd_action"},
                              self.cfg.crowd_policy.hidden_dim
							  )
        self.video_recorder = VideoRecorder(self.work_dir if cfg.save_video else None)

        self.timer = utils.Timer()
        self.global_step = 0
        self.eval_rewards_history: tp.List[float] = []
        self._checkpoint_filepath = self.work_dir / "models" / "latest.pt"
        
        if self._checkpoint_filepath.exists():
            self.load_checkpoint(self._checkpoint_filepath)
        elif cfg.load_model is not None:
            self.load_checkpoint(cfg.load_model, exclude=["replay_loader"])

    def _make_vec_envs(self,phase='train'):
        assert phase=='train'
        def get_random_human_num(max_human_num,nenv):
            return np.random.randint(1,max_human_num+1,nenv)

        self.human_nums = get_random_human_num(self.cfg.max_human_num,self.cfg.num_processes).tolist()
        print(self.human_nums)
        return make_vec_envs(self.cfg.crowd_sim_env,
                                self.cfg.seed,
                                self.cfg.num_processes,
                                self.human_nums,
                                phase,
                                1.0, #discount gamma will be used in calculating return in storage class
                                self.cfg.max_episode_length,
                                self.device)

    def _make_env(self,phase='train'):
        envs = {}
        for human_num in range(1,self.cfg.max_human_num+1):
            envs[human_num] = RobotCrowdSim(self.cfg.crowd_sim_env,phase,
                                            human_num,
                                            1.0,
                                            self.cfg.max_episode_length)
        return envs

    def eval(self) -> None:
        # self.agent.train(False) #TODO
        step, episode = 0, 0
        success_num = 0
        eval_until_episode = utils.Until(self.cfg.num_eval_episodes)
        rewards: tp.List[float] = []

        while eval_until_episode(episode):
            # time_step_multi = self.eval_env.reset() #TODO
            # get meta

            self.video_recorder.init(self.eval_env, enabled=True) #enabled=(episode == 0) force the recorder only save episode 0
            # while not all([ts.last() for ts in time_step_multi]):
                # act
                # self.video_recorder.record(self.eval_env)

                #step
                # step += 1
                # episode_step+=1

            # summarize episode
            # for time_step in time_step_multi: 
            #     # if time_step.last(): continue
            #     success_num = success_num+1 if time_step.info.contain(ReachGoal()) else success_num #this seemly no working!! TODO debug
            # rewards+=total_reward
            episode += 1
            # self.video_recorder.save(f'{self.global_step}_{episode}.mp4')

        # self.agent.train(True) #TODO

        # log
        # self.eval_rewards_history.append(float(np.mean(rewards)))
        # with self.logger.log_and_dump_ctx(self.global_step, ty='eval') as log:
        #     log('episode_reward', self.eval_rewards_history[-1])
        #     if len(rewards) > 1:
        #         log('episode_reward#std', float(np.std(rewards)))
        #     log('episode_length', step * self.cfg.action_repeat / episode)
        #     log('episode', self.global_episode)
        #     log('z_correl', z_correl / episode)
        #     log('step', self.global_step)
        #     log('success rate', float(success_num)/total_robot_number)
        #     if actor_success:
        #         log('actor_sucess', float(np.mean(actor_success)))

    _CHECKPOINTED_KEYS = ('self_agent', 'crowd_agent', 'global_step')

    def save_checkpoint(self, fp: tp.Union[Path, str], exclude: tp.Sequence[str] = ()) -> None:
        logger.info(f"Saving checkpoint to {fp}")
        exclude = list(exclude)
        assert all(x in self._CHECKPOINTED_KEYS for x in exclude)
        fp = Path(fp)
        fp.parent.mkdir(exist_ok=True, parents=True)
        # assert isinstance(self.replay_loader, ReplayBuffer), "Is this buffer designed for checkpointing?"
        # this is just a dumb security check to not forget about it
        payload = {k: self.__dict__[k] for k in self._CHECKPOINTED_KEYS if k not in exclude}
        with fp.open('wb') as f:
            torch.save(payload, f, pickle_protocol=4)

    def load_checkpoint(self, fp: tp.Union[Path, str], only: tp.Optional[tp.Sequence[str]] = None, exclude: tp.Sequence[str] = ()) -> None:
        """Reloads a checkpoint or part of it

        Parameters
        ----------
        only: None or sequence of str
            reloads only a specific subset (defaults to all)
        exclude: sequence of str
            does not reload the provided keys
        """
        print(f"loading checkpoint from {fp}")
        fp = Path(fp)
        with fp.open('rb') as f:
            payload = torch.load(f)
        if only is not None:
            only = list(only)
            assert all(x in self._CHECKPOINTED_KEYS for x in only)
            payload = {x: payload[x] for x in only}
        exclude = list(exclude)
        assert all(x in self._CHECKPOINTED_KEYS for x in exclude)
        for x in exclude:
            payload.pop(x, None)
        for name, val in payload.items():
            logger.info("Reloading %s from %s", name, fp)
            if name == "self_agent":
                self.self_agent.init_from(val) #TODO
            if name == "crowd_agent":
                for i in range(1, self.cfg.max_human_num+1):
                    self.crowd_agent[i].init_from(val[i])
            else:
                assert hasattr(self, name)
                setattr(self, name, val)

    def finalize(self,num_eval_episodes=None,custom_task=None) -> None:
        self.phase = 'test'
        print("Running final test", flush=True)
        self.eval_env = self._make_env(phase='test')
        if num_eval_episodes is not None:
            self.cfg.num_eval_episodes=num_eval_episodes
        else:    self.cfg.num_eval_episodes = 10
        self.eval()
        # with (self.work_dir / "test_rewards.json").open("w") as f:
        #     json.dump(rewards, f)


class Workspace(BaseWorkspace[PretrainConfig]):
    def __init__(self, cfg: PretrainConfig) -> None:
        super().__init__(cfg)

    def train(self) -> None:
        # predicates
        train_until_step = utils.Until(self.cfg.num_train_frames)
        eval_every_step = utils.Every(self.cfg.eval_every_frames)

        #get init obs of all robots in all env
        """
        crowd_obs:
            1 human {full_state, crowd_preference}
            2 human 
            ...
            N human
        robot_obs: {scan, robot_state}
        """
        crowd_obs, robot_obs = self.train_env.reset() 

        # first obs
        for human_num in crowd_obs.keys():
            for key in crowd_obs[human_num].keys():
                self.rollouts_crowd_agent[human_num].obs[key][0].copy_(crowd_obs[human_num][key])
            self.rollouts_crowd_agent[human_num].to(self.device)
        for key in robot_obs.keys():
            self.rollouts_self_agent.obs[key][0].copy_(robot_obs[key])
        self.rollouts_self_agent.to(self.device)


        while train_until_step(self.global_step):
            self.phase = 'train'
            metrics: tp.Dict[str,float] = {}
            for human_num in self.crowd_agent.keys():
                self.crowd_agent[human_num].train_mode()
            # self.self_agent.train_mode() 

            # step the environment for a few times
            for step in range(self.cfg.num_steps):
                
                crowd_action = {}
                crowd_value = {}
                crowd_action_log_probs = {}
                crowd_rnn_hxs = {}
                # sample actions
                with torch.no_grad():

                    robot_action = torch.cat([
                        torch.ones((self.cfg.num_processes,1),device=self.cfg.device),
                        torch.zeros((self.cfg.num_processes,1),device=self.cfg.device)
                     ],dim=-1) #TODO 
                    
                    for human_num in self.crowd_agent.keys():
                        rollouts_obs = {}
                        for key in self.rollouts_crowd_agent[human_num].obs:
                            rollouts_obs[key] = self.rollouts_crowd_agent[human_num].obs[key][step]
                        rollouts_hidden_s = self.rollouts_crowd_agent[human_num].recurrent_hidden_states[step]
                        mask = self.rollouts_crowd_agent[human_num].masks[step]
                        (
                            value, 
                            action, 
                            action_log_probs, 
                            rnn_hxs
                        ) = self.crowd_agent[human_num].actor_critic.act(
                            rollouts_obs,
                            rollouts_hidden_s,
                            mask
                        )
                        crowd_action[human_num] = action
                        crowd_value[human_num] = value
                        crowd_action_log_probs[human_num] = action_log_probs
                        crowd_rnn_hxs[human_num] = rnn_hxs

                # env transition
                (
                    (crowd_obs, robot_obs), 
                    (robot_rews, crowd_rews_dict), 
                    (robot_discounts,human_discounts_dict), 
                    (infos,crowd_infos_dict)
                )= self.train_env.step(
                    {"robot_action":copy.deepcopy(robot_action),
                     "crowd_action":copy.deepcopy(crowd_action)}
                    )
                
                # write to rollout storage
                for human_num in self.rollouts_crowd_agent.keys():
                    self.rollouts_crowd_agent[human_num].insert(
                        crowd_obs[human_num],
                        crowd_rnn_hxs[human_num],
                        crowd_action[human_num],
                        crowd_action_log_probs[human_num],
                        crowd_value[human_num],
                        crowd_rews_dict[human_num],
                        human_discounts_dict[human_num] #mask 1 for non-terminal step, 0 for terminal step
                        )
                

                # analyze info and log episode status
                self.agent_monitor.saveInfoVecEnv((infos,crowd_infos_dict))

            
            # store the stepped experience to buffer
            crowd_next_value = {}
            with torch.no_grad():
                for human_num in self.rollouts_crowd_agent.keys():
                    rollouts_obs = {}
                    for key in self.rollouts_crowd_agent[human_num].obs:
                        rollouts_obs[key]= self.rollouts_crowd_agent[human_num].obs[key][-1]
                    rollouts_hidden_s= self.rollouts_crowd_agent[human_num].recurrent_hidden_states[-1]
                    crowd_next_value[human_num] = self.crowd_agent[human_num].actor_critic.get_value(
                                                            rollouts_obs, 
                                                            rollouts_hidden_s,
                                                            self.rollouts_crowd_agent[human_num].masks[-1]
                                                            ).detach()

            # compute advantage and gradient, and update the network parameters
            for human_num in self.rollouts_crowd_agent.keys():
                self.rollouts_crowd_agent[human_num].compute_returns(
                    crowd_next_value[human_num], 
                    self.cfg.crowd_policy.use_gae, 
                    self.cfg.crowd_policy.discount,
                    self.cfg.crowd_policy.gae_lambda
                    )

            # update
            for human_num in self.crowd_agent.keys():
                metrics.update(
                        self.crowd_agent[human_num].update(
                            self.rollouts_crowd_agent[human_num]
                        )
                )

            for human_num in self.rollouts_crowd_agent.keys():
                self.rollouts_crowd_agent[human_num].after_update()
            
            # try to evaluate #TODO
            # if eval_every_step(self.global_step):
            #     self.logger.log('eval_total_time', self.timer.total_time(),
            #                     self.global_step)
            #     self.phase = 'val'
            #     self.eval()

            self.global_step += 1
            self.logger.log_metrics(metrics, self.global_step, ty='train')
            with self.logger.log_and_dump_ctx(self.global_step,
                                                      ty='train') as log:
                elapsed_time, total_time = self.timer.reset()
                log('fps', self.global_step / elapsed_time)#TODO fix bug
                log('total_time', total_time)
                log('step', self.global_step)
                sr,cr,tr,fi,at,std_t = self.agent_monitor.evaluateCrowdInfo()
                log('crowd success rate',sr)
                log('crowd collide rate',cr)
                log('crowd timeout rate',tr)
                log('crowd frequecy invasion',fi)
                log('crowd average navitime',at)
                log('crowd navitime std',std_t)
                sr,cr,tr,fi,at,std_t = self.agent_monitor.evaluateRobotInfo()
                log('robot success rate',sr)
                log('robot collide rate',cr)
                log('robot timeout rate',tr)
                log('robot frequecy invasion',fi)
                log('robot average navitime',at)
                log('robot navitime std',std_t)

            # save checkpoint to reload
            if not self.global_step % self.cfg.checkpoint_every:
                self.save_checkpoint(self._checkpoint_filepath)
            # try to save snapshot 
            if self.global_step in self.cfg.snapshot_at:
                self.save_checkpoint(self._checkpoint_filepath.with_name(f'snapshot_{self.global_step}.pt'))
                
        self.save_checkpoint(self._checkpoint_filepath)  # make sure we save the final checkpoint
        self.finalize()


@hydra.main(config_path='.', config_name='br_ppo_config', version_base="1.1")
def main(cfg: omgcf.DictConfig) -> None:
    # we assume cfg is a PretrainConfig (but actually not really)
    workspace = Workspace(cfg)  # type: ignore
    workspace.train()


if __name__ == '__main__':
    main()
