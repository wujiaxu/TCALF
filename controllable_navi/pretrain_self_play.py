# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from pathlib import Path
import sys
import faulthandler
faulthandler.enable()

from sympy import sequence
base = Path(__file__).absolute().parents[1]
# we need to add base repo to be able to import controllable_navi
# we need to add controllable_navi to be able to reload legacy checkpoints
for fp in [base, base / "controllable_navi"]:
    assert fp.exists()
    if str(fp) not in sys.path:
        sys.path.append(str(fp))

import os
import json
import pdb  # pylint: disable=unused-import
import logging
import dataclasses
import typing as tp
import warnings
from pathlib import Path

warnings.filterwarnings('ignore', category=DeprecationWarning)


# os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
# os.environ['HYDRA_FULL_ERROR'] = '1'
# # if the default egl does not work, you may want to try:
# # export MUJOCO_GL=glfw
# os.environ['MUJOCO_GL'] = os.environ.get('MUJOCO_GL', 'egl')

import hydra
from hydra.core.config_store import ConfigStore
import numpy as np
import torch
import wandb
import omegaconf as omgcf

from controllable_navi import dmc
from controllable_navi.dm_env_light import specs
from controllable_navi import utils
from controllable_navi import goals as _goals
from controllable_navi.logger import Logger
from controllable_navi.in_memory_replay_buffer import ReplayBuffer
from controllable_navi.video import VideoRecorder
from controllable_navi import agent as agents
from controllable_navi import crowd_sim as crowd_sims
from controllable_navi.crowd_sim.utils.info import *

logger = logging.getLogger(__name__)
torch.backends.cudnn.benchmark = True
# os.environ['WANDB_MODE']='offline'



# # # Config # # #

@dataclasses.dataclass
class Config:
    agent: tp.Any
    crowd_sim: tp.Any 
    max_episode_length:int = 50
    # misc
    seed: int = 11
    device: str = "cuda"
    save_video: bool = False
    use_tb: bool = False
    use_wandb: bool = False
    use_hiplog: bool = False
    # experiment
    experiment: str = "online"
    # task settings
    task: str = "PointGoalNavi"
    obs_type: str = "states"  # [states, pixels]
    frame_stack: int = 3  # only works if obs_type=pixels
    action_repeat: int = 1  # set to 2 for pixels which means 2 action execute between steps
    discount: float = 0.99
    future: float = 0.99  # discount of future sampling, future=1 means no future sampling
    goal_space: tp.Optional[str] = None
    append_goal_to_observation: bool = False
    # eval
    num_eval_episodes: int = 10
    custom_reward: tp.Optional[str] = None  # activates custom eval if not None
    final_tests: int = 10
    # checkpoint
    snapshot_at: tp.Tuple[int, ...] = (100000, 200000, 500000, 800000, 1000000, 1500000,
                                       2000000, 3000000, 4000000, 5000000, 9000000, 10000000)
    # snapshot_at: tp.Tuple[int, ...] = (10000, 20000, 50000, 80000, 100000, 150000,
    #                                    200000, 300000, 400000, 500000, 900000, 1000000)
    checkpoint_every: int = 100000#100000
    load_model: tp.Optional[str] = None
    # training
    num_seed_frames: int = 4000
    replay_buffer_episodes: int = 500 #default 5000
    update_encoder: bool = True
    batch_size: int = omgcf.II("agent.batch_size")


@dataclasses.dataclass
class PretrainConfig(Config):
    # mode
    reward_free: bool = True
    # train settings
    num_train_frames: int = 4000010#2000010 #TODO need more step for converge when goal reward is 0.25
    # snapshot
    eval_every_frames: int = 10000
    load_replay_buffer: tp.Optional[str] = None
    # replay buffer
    # replay_buffer_num_workers: int = 4
    # nstep: int = omgcf.II("agent.nstep")
    # misc
    save_train_video: bool = False


# loaded as base_pretrain in pretrain.yaml
# we keep the yaml since it's easier to configure plugins from it
ConfigStore.instance().store(name="workspace_config", node=PretrainConfig)
# # # Implem # # #


def make_agent(
    obs_type: str, obs_spec, action_spec, num_expl_steps: int, cfg: omgcf.DictConfig
) -> tp.Union[agents.DDPGAgent,agents.FBDDPGAgent]:
    cfg.obs_type = obs_type
    cfg.obs_shape = obs_spec.shape_dict
    cfg.action_shape = (action_spec.num_values, ) if isinstance(action_spec, specs.DiscreteArray) \
        else action_spec.shape
    cfg.num_expl_steps = num_expl_steps
    # reference https://zenn.dev/gesonanko/articles/417d43669cf2af (use hydra to instantiate an object)
    return hydra.utils.instantiate(cfg)


C = tp.TypeVar("C", bound=Config) # Can be any subtype of Config


def _update_legacy_class(obj: tp.Any, classes: tp.Sequence[tp.Type[tp.Any]]) -> tp.Any:
    """Updates a legacy class (eg: agent.FBDDPGAgent) to the new
    class (controllable_navi.agent.FBDDPGAgent)

    Parameters
    ----------
    obj: Any
        Object to update
    classes: Types
        Possible classes to update the object to. If current name is one of the classes
        name, the object class will be remapped to it.
    """
    classes = tuple(classes)
    if not isinstance(obj, classes):
        clss = {x.__name__: x for x in classes}
        cls = clss.get(obj.__class__.__name__, None)
        if cls is not None:
            logger.warning(f"Promoting legacy object {obj.__class__} to {cls}")
            obj.__class__ = cls


def _init_eval_meta(workspace: "BaseWorkspace", custom_reward: tp.Optional[_goals.BaseReward] = None) -> agents.MetaDict:
    
    #special = (agents.FBDDPGAgent, agents.SFAgent, agents.SFSVDAgent, agents.APSAgent, agents.NEWAPSAgent, agents.GoalSMAgent, agents.UVFAgent)
    special = (agents.FBDDPGAgent,agents.APSAgent)
    ag = workspace.agent
    _update_legacy_class(ag, special)
    # we need to check against name for legacy reason when reloading old checkpoints
    if not isinstance(ag, special) or not len(workspace.replay_loader):
        return workspace.agent.init_meta()
    if custom_reward is not None:
        try:  # if the custom reward implements a goal, return it
            goal = custom_reward.get_goal(workspace.cfg.goal_space)
            return workspace.agent.get_goal_meta(goal)
        except Exception:  # pylint: disable=broad-except
            pass
        #if not isinstance(workspace.agent, agents.SFSVDAgent):
        # we cannot fully type because of the FBBDPG string check :s
        num_steps = workspace.agent.cfg.num_inference_steps  # type: ignore
        obs_list, reward_list = [], []
        next_obs_list, action_list = [],[]
        batch_size = 0
        while batch_size < num_steps:
            batch = workspace.replay_loader.sample(workspace.cfg.batch_size, custom_reward=custom_reward)
            batch = batch.to(workspace.cfg.device)
            #need filter
            obs_used = batch.obs[batch.reward!=0]
            next_obs_used = batch.next_obs[batch.reward!=0]
            action_used = batch.action[batch.reward!=0]
            reward_used = batch.reward[batch.reward!=0]
            obs_list.append(obs_used)
            reward_list.append(reward_used)
            action_list.append(action_used)
            next_obs_list.append(next_obs_used)
            batch_size += obs_used.size(0)
        obs, reward = torch.cat(obs_list, 0), torch.cat(reward_list, 0)  # type: ignore
        next_obs, action = torch.cat(next_obs_list, 0), torch.cat(action_list, 0)  # type: ignore
        obs_t, reward_t = obs[:num_steps], reward[:num_steps]
        next_obs_t, action_t = next_obs[:num_steps], action[:num_steps]
        return workspace.agent.infer_meta_from_obs_and_rewards(obs_t, action_t,reward_t,next_obs_t)

    return workspace.agent.init_meta()


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
        # goal_spec: tp.Optional[specs.Array] = None
        # if cfg.goal_space is not None:
        #     g = _goals.goals.funcs[cfg.goal_space][cfg.task]()
        #     goal_spec = specs.Array((len(g),), np.float32, 'goal')

        # create envs
        # task = PRIMAL_TASKS[self.domain]
        task = cfg.task

        self.domain = task.split('_', maxsplit=1)[0]

        self.train_env = self._make_env()
        self.eval_env = self._make_env(phase='val')
        # create agent
        self.agent = make_agent(cfg.obs_type,
                                self.train_env.observation_spec(),
                                self.train_env.action_spec(),
                                cfg.num_seed_frames // cfg.action_repeat,
                                cfg.agent)
        # create logger
        self.logger = Logger(self.work_dir,
                             use_tb=cfg.use_tb,
                             use_wandb=cfg.use_wandb,
                             use_hiplog=cfg.use_hiplog)
        # if cfg.use_tb:#TODO debug @ GD_aps.py
        #     self.agent.add_to_tb(self.logger._sw)

        if cfg.use_wandb:
            exp_name = '_'.join([
                cfg.experiment, cfg.agent.name, self.domain
            ])
            wandb.init(project="controllable_navi", group=cfg.agent.name, name=exp_name,  # mode="disabled",
                       config=omgcf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True))  # type: ignore

        if cfg.use_hiplog:
            # record config now that it is filled
            parts = ("snapshot", "_type", "_shape", "num_", "save_", "frame", "device", "use_tb", "use_wandb")
            skipped = [x for x in cfg if any(y in x for y in parts)]  # type: ignore
            self.logger.hiplog.flattened({x: y for x, y in cfg.items() if x not in skipped})  # type: ignore
            self.logger.hiplog(workdir=self.work_dir.stem)
            for rm in ("agent/use_tb", "agent/use_wandb", "agent/device"):
                del self.logger.hiplog._content[rm]
            self.logger.hiplog(observation_size=np.prod(self.train_env.observation_spec().shape))

        # # create replay buffer
        # self._data_specs: tp.List[tp.Any] = [self.train_env.observation_spec(),
        #                                      self.train_env.action_spec(), ]
        # if cfg.goal_space is not None:
        #     if cfg.goal_space not in _goals.goal_spaces.funcs[self.domain]:
        #         raise ValueError(f"Unregistered goal space {cfg.goal_space} for domain {self.domain}")
        
        self.replay_loader = ReplayBuffer(max_episodes=cfg.replay_buffer_episodes, 
                                            discount=cfg.discount, future=cfg.future,
                                            max_episode_length=self.cfg.max_episode_length+1)
        cam_id = 0 # if 'quadruped' not in self.domain else 2

        self.video_recorder = VideoRecorder(self.work_dir if cfg.save_video else None,
                                            camera_id=cam_id, use_wandb=self.cfg.use_wandb)

        self.timer = utils.Timer()
        self.global_step = 0
        self.global_episode = 0
        self.eval_rewards_history: tp.List[float] = []
        self._checkpoint_filepath = self.work_dir / "models" / "latest.pt"
        
        if self._checkpoint_filepath.exists():
            self.load_checkpoint(self._checkpoint_filepath)
        elif cfg.load_model is not None:
            self.load_checkpoint(cfg.load_model, exclude=["replay_loader"])

        self.reward_cls: tp.Optional[_goals.BaseReward] = None

    def _make_env(self,phase='train') -> dmc.EnvWrapper:
        # cfg = self.cfg
        if self.domain == "crowdnavi":
            
            return dmc.EnvWrapper(crowd_sims.build_multirobotworld_task(self.cfg.crowd_sim,self.cfg.task.split('_')[1],phase,discount=self.cfg.discount,observation_type=self.cfg.obs_type,max_episode_length=self.cfg.max_episode_length))
        else:
            raise NotImplementedError
        # return dmc.make(cfg.task, cfg.obs_type, cfg.frame_stack, cfg.action_repeat, cfg.seed,
        #                 goal_space=cfg.goal_space, append_goal_to_observation=cfg.append_goal_to_observation)

    @property
    def global_frame(self) -> int:
        return self.global_step * self.cfg.action_repeat

    def _make_custom_reward(self) -> tp.Union[None,_goals.BaseReward,tp.List[_goals.BaseReward]]:
        """Creates a custom reward function if provided in configuration
        else returns None
        """
        if self.cfg.custom_reward is None:
            return None
        if isinstance(self.cfg.custom_reward,omgcf.listconfig.ListConfig):
            return [_goals.get_reward_function(custom_reward) for custom_reward in self.cfg.custom_reward] 
        return _goals.get_reward_function(self.cfg.custom_reward)

    def eval(self) -> None:
        self.agent.train(False)
        step, episode = 0, 0
        success_num = 0
        eval_until_episode = utils.Until(self.cfg.num_eval_episodes)
        rewards: tp.List[float] = []
        # seed = 12 * self.cfg.num_eval_episodes + len(rewards)
        custom_reward = self._make_custom_reward()
        if custom_reward is not None:
            if isinstance(custom_reward,tp.List):
                meta = []
                for reward in custom_reward:
                    meta.append(_init_eval_meta(self,reward))
            else:
                meta = _init_eval_meta(self, custom_reward)
        else:
            meta = _init_eval_meta(self)
        z_correl = 0.0
        # is_d4rl_task = self.cfg.task.split('_')[0] == 'd4rl'
        # TODO add info to calculate success rate slp collision rate navi time
        actor_success: tp.List[float] = []
        total_robot_number = 0
        while eval_until_episode(episode):
            time_step_multi = self.eval_env.reset()
            robot_num = len(time_step_multi)
            total_robot_number+=robot_num
            episode_step = 0 
            metas = []
            total_reward = []
            # if self.agent.use_sequence: #TODO
            #     obs_ = np.zeros((self.cfg.max_episode_length+1,) + self.train_env.observation_spec().shape, dtype=np.float32)
            #     obs_[episode_step] = time_step.observation
            #     mask = np.zeros(self.cfg.max_episode_length+1, dtype=np.float32)
            #     mask[episode_step] = 1
            #     obs = (obs_,mask)
            # else:
            #     obs = time_step.observation
            if isinstance(meta,tp.List):
                metas+=meta
                if len(custom_reward)<robot_num:
                    for i in range(robot_num-len(custom_reward)):
                        meta_random = _init_eval_meta(self)
                        metas.append(meta_random)
                print(metas)
            else:
                for i in range(robot_num):
                    if custom_reward is None:
                        meta_random = _init_eval_meta(self)
                    metas.append(meta_random)
            total_reward = [0.0]*robot_num 
            self.video_recorder.init(self.eval_env, enabled=True) #enabled=(episode == 0) force the recorder only save episode 0
            while not all([ts.last() for ts in time_step_multi]):
                actions = []
                with torch.no_grad(), utils.eval_mode(self.agent):
                    for time_step,meta_ in zip(time_step_multi,metas):
                        if time_step.last():
                            actions.append(np.zeros(2))
                            continue
                        action = self.agent.act(time_step.observation,
                                                meta_,
                                                self.global_step,
                                                eval_mode=True)
                        actions.append(action)
                    # print(episode_step,action)
                    # input()
                time_step_multi = self.eval_env.step(np.array(actions))
                self.video_recorder.record(self.eval_env)

                # for legacy reasons, we need to check the name :s
                # if custom_reward is not None: #TODO
                #     time_step.reward = custom_reward.from_env(self.eval_env)
                # total_reward += time_step.reward
                for i, time_step in enumerate(time_step_multi):
                    if time_step.last(): continue
                    total_reward[i] += self.cfg.discount**episode_step*time_step.reward
                step += 1
                episode_step+=1
                # if self.agent.use_sequence:
                #     obs_[episode_step] = time_step.observation
                #     mask[episode_step] = 1
                #     obs = (obs_,mask)
                # else:
                #     obs = time_step.observation
            # if is_d4rl_task:
            #     normalized_scores.append(self.eval_env.get_normalized_score(total_reward))
            for time_step in time_step_multi: 
                # if time_step.last(): continue
                success_num = success_num+1 if time_step.info.contain(ReachGoal()) else success_num #this seemly no working!! TODO debug
            rewards+=total_reward
            episode += 1
            self.video_recorder.save(f'{self.global_frame}_{episode}.mp4')

        self.agent.train(True)
        self.eval_rewards_history.append(float(np.mean(rewards)))
        with self.logger.log_and_dump_ctx(self.global_frame, ty='eval') as log:
            # if is_d4rl_task:
            #     log('episode_normalized_score', float(100 * np.mean(normalized_scores)))
            log('episode_reward', self.eval_rewards_history[-1])
            if len(rewards) > 1:
                log('episode_reward#std', float(np.std(rewards)))
            log('episode_length', step * self.cfg.action_repeat / episode)
            log('episode', self.global_episode)
            log('z_correl', z_correl / episode)
            log('step', self.global_step)
            log('success rate', float(success_num)/total_robot_number)
            if actor_success:
                log('actor_sucess', float(np.mean(actor_success)))

    _CHECKPOINTED_KEYS = ('agent', 'global_step', 'global_episode', "replay_loader")

    def save_checkpoint(self, fp: tp.Union[Path, str], exclude: tp.Sequence[str] = ()) -> None:
        logger.info(f"Saving checkpoint to {fp}")
        exclude = list(exclude)
        assert all(x in self._CHECKPOINTED_KEYS for x in exclude)
        fp = Path(fp)
        fp.parent.mkdir(exist_ok=True, parents=True)
        assert isinstance(self.replay_loader, ReplayBuffer), "Is this buffer designed for checkpointing?"
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
        _update_legacy_class(payload, (ReplayBuffer,))
        if isinstance(payload, ReplayBuffer):  # compatibility with pure buffers pickles
            payload = {"replay_loader": payload}
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
            if name == "agent":
                self.agent.init_from(val)
            elif name == "replay_loader":
                _update_legacy_class(val, (ReplayBuffer,))
                assert isinstance(val, ReplayBuffer)
                # pylint: disable=protected-access
                # drop unecessary meta which could make a mess
                val._current_episode.clear()  # make sure we can start over
                val._future = self.cfg.future
                val._discount = self.cfg.discount
                val._max_episodes = len(val._storage["discount"])
                self.replay_loader = val
            else:
                assert hasattr(self, name)
                setattr(self, name, val)
                if name == "global_episode":
                    logger.warning(f"Reloaded agent at global episode {self.global_episode}")

    def finalize(self,num_eval_episodes=None,custom_task=None,single_robot=False) -> None:
        print("Running final test", flush=True)
        repeat = 1 #self.cfg.final_tests
        if not repeat:
            return

        if custom_task is None:
            domain_tasks = {
                "crowdnavi":[
                            #  'PointGoalNavi',
                            'PassLeftSide',
                            'PassRightSide',
                            #  'AwayFromHuman',
                            #  'LowSpeed'
                            ] #'FollowWall',
            }
        else:
            domain_tasks = {
                "crowdnavi":[
                            custom_task
                            ] 
            }
        if self.domain not in domain_tasks:
            return
        eval_hist = self.eval_rewards_history
        rewards = {}
        for name in domain_tasks[self.domain]:
            self.global_step+=1
            task = "_".join([self.domain, name])
            self.cfg.task = task
            if single_robot:
                self.cfg.custom_reward = [task]
            else:
                self.cfg.custom_reward = task  # for the replay buffer
            self.cfg.seed += 1  # for the sake of avoiding similar seeds
            self.eval_env = self._make_env(phase='test')
            self.eval_rewards_history = []
            if num_eval_episodes is not None:
                self.cfg.num_eval_episodes=num_eval_episodes
            else:    self.cfg.num_eval_episodes = 10
            for _ in range(repeat):
                self.eval()
            rewards[task] = self.eval_rewards_history
        self.eval_rewards_history = eval_hist  # restore
        with (self.work_dir / "test_rewards.json").open("w") as f:
            json.dump(rewards, f)


class Workspace(BaseWorkspace[PretrainConfig]):
    def __init__(self, cfg: PretrainConfig) -> None:
        super().__init__(cfg)
        # self.train_video_recorder = TrainVideoRecorder(self.work_dir if cfg.save_train_video else None,
        #                                                camera_id=self.video_recorder.camera_id, use_wandb=self.cfg.use_wandb)
        if not self._checkpoint_filepath.exists():  # don't relay if there is a checkpoint
            if cfg.load_replay_buffer is not None:
                # if self.cfg.task.split('_')[0] == "d4rl":
                #     d4rl_replay_buffer_builder = D4RLReplayBufferBuilder()
                #     self.replay_storage = d4rl_replay_buffer_builder.prepare_replay_buffer_d4rl(self.train_env, self.agent.init_meta(), self.cfg)
                #     self.replay_loader = self.replay_storage
                # else:
                self.load_checkpoint(cfg.load_replay_buffer, only=["replay_loader"])

    def _init_meta(self):
        # if isinstance(self.agent, agents.GoalTD3Agent) and isinstance(self.reward_cls, _goals.MazeMultiGoal):
        #     meta = self.agent.init_meta(self.reward_cls)
        # elif isinstance(self.agent, agents.GoalSMAgent) and len(self.replay_loader) > 0:
        #     meta = self.agent.init_meta(self.replay_loader)
        # else:
        meta = self.agent.init_meta()
        return meta

    def train(self) -> None:
        # predicates
        train_until_step = utils.Until(self.cfg.num_train_frames,
                                       self.cfg.action_repeat)
        seed_until_step = utils.Until(self.cfg.num_seed_frames,
                                      self.cfg.action_repeat)
        eval_every_step = utils.Every(self.cfg.eval_every_frames,
                                      self.cfg.action_repeat)
        # if self.cfg.custom_reward is not None:
        #     raise NotImplementedError("Custom reward not implemented in pretrain.py train loop (see anytrain.py)")

        episode_step, z_correl = 0, 0.0
        time_step_multi = self.train_env.reset()
        
        # if self.agent.use_sequence: TODO
        #     obs_ = np.zeros((self.cfg.max_episode_length+1,) + self.train_env.observation_spec().shape, dtype=np.float32)
        #     obs_[episode_step] = time_step.observation
        #     mask = np.zeros(self.cfg.max_episode_length+1, dtype=np.float32)
        #     mask[episode_step] = 1
        #     obs = (obs_,mask)
        # else:
        #     obs = time_step.observation
        robot_number = len(time_step_multi)
        episode_reward = []
        metas = []
        for _ in range(robot_number):
            meta = self._init_meta()
            metas.append(meta)
            episode_reward.append(0.0)
        self.replay_loader.add_multi(time_step_multi, metas)
        # self.train_video_recorder.init(time_step.observation)
        metrics = None
        # physics_agg = dmc.PhysicsAggregator()

        while train_until_step(self.global_step):
            if all([ts.last() for ts in time_step_multi]):
                success = 0
                start_goal_dists = []
                final_goal_dists = []
                for time_step in time_step_multi:
                    if time_step.info.contain(ReachGoal()):
                        success+=1.0
                    task_info = time_step.info.task_info
                    start_goal_dists.append(np.sqrt((task_info["gx"]-task_info["sx"])**2+(task_info["gy"]-task_info["sy"])**2))
                    final_goal_dists.append(time_step.observation[-5])
                success_rate = success/robot_number
                start_goal_dist = sum(start_goal_dists)/len(start_goal_dists)
                final_goal_dist = sum(final_goal_dists)/len(final_goal_dists)
                self.global_episode += 1
                # self.train_video_recorder.save(f'{self.global_frame}.mp4')
                # wait until all the metrics schema is populated
                if metrics is not None:
                    # log stats
                    elapsed_time, total_time = self.timer.reset()
                    episode_frame = episode_step * self.cfg.action_repeat
                    with self.logger.log_and_dump_ctx(self.global_frame,
                                                      ty='train') as log:
                        log('fps', episode_frame / elapsed_time)
                        log('total_time', total_time)
                        log('episode_reward', sum(episode_reward)/robot_number)
                        log('episode_length', episode_frame)
                        log('episode', self.global_episode)
                        log('buffer_size', len(self.replay_loader))
                        log('step', self.global_step)
                        log('z_correl', z_correl)
                        log('start goal dist',start_goal_dist)
                        log('final goal dist',final_goal_dist)
                        log('success rate',success_rate)
                        # TODO record success rate

                        # for key, val in physics_agg.dump():
                        #     log(key, val)
                if self.cfg.use_hiplog and self.logger.hiplog.content:
                    self.logger.hiplog.write()
                
                    # self.logger.log_model_weights('actor',self.global_frame,self.agent.actor)
                    # self.logger.log_model_weights('critic',self.global_frame,self.agent.critic)
                    
                # reset env
                time_step_multi = self.train_env.reset()
                robot_number = len(time_step_multi)
                metas = []
                episode_reward = []
                for _ in range(robot_number):
                    meta = self._init_meta()
                    metas.append(meta)
                    episode_reward.append(0.0)
                self.replay_loader.add_multi(time_step_multi, metas)
                # self.train_video_recorder.init(time_step.observation)
                
                episode_step = 0
                # episode_reward = 0.0
                z_correl = 0.0

                # if self.agent.use_sequence: TODO
                #     obs_ = np.zeros((self.cfg.max_episode_length+1,) + self.train_env.observation_spec().shape, dtype=np.float32)
                #     obs_[episode_step] = time_step.observation
                #     mask = np.zeros(self.cfg.max_episode_length+1, dtype=np.float32)
                #     mask[episode_step] = 1
                #     obs = (obs_,mask)
                # else:
                #     obs = time_step.observation

            # try to evaluate
            if eval_every_step(self.global_step):
                self.logger.log('eval_total_time', self.timer.total_time(),
                                self.global_frame)
                # if self.cfg.custom_reward == "maze_multi_goal":
                #     self.eval_maze_goals()
                # elif self.domain == "grid":
                #     self.eval_grid_goals()
                # else:
                if not seed_until_step(self.global_step):
                    self.logger.log_distribution(self.global_frame,self.replay_loader,save_to_file=True)
                self.eval()
            # TODO consider whether comment out meta update is ok? (currently I want one episode one z, so I don't update meta during episode)
            # meta = self.agent.update_meta(meta, self.global_step, time_step, finetune=False, replay_loader=self.replay_loader)
            # sample action
            actions = []
            with torch.no_grad(), utils.eval_mode(self.agent):
                for time_step,meta in zip(time_step_multi,metas):
                    action = self.agent.act(time_step.observation,
                                            meta,
                                            self.global_step,
                                            eval_mode=False)
                    actions.append(action)
            actions = np.array(actions)
            # try to update the agent
            if not seed_until_step(self.global_step):
                # TODO: reward_free should be handled in the agent update itself !
                # TODO: the commented code below raises incompatible type "Generator[EpisodeBatch[ndarray[Any, Any]], None, None]"; expected "ReplayBuffer"
                # replay = (x.with_no_reward() if self.cfg.reward_free else x for x in self.replay_loader)
                # if isinstance(self.agent, agents.GoalTD3Agent) and isinstance(self.reward_cls, _goals.MazeMultiGoal):
                #     metrics = self.agent.update(self.replay_loader, self.global_step, self.reward_cls)
                #else:
                metrics = self.agent.update(self.replay_loader, self.global_step)
                self.logger.log_metrics(metrics, self.global_frame, ty='train')

            # take env step
            time_step_multi = self.train_env.step(actions)
            # physics_agg.add(self.train_env)
            for i, time_step in enumerate(time_step_multi):
                episode_reward[i] += self.cfg.discount**episode_step*time_step.reward #NEW: consider discount
            self.replay_loader.add_multi(time_step_multi, metas)
            # self.train_video_recorder.record(time_step.observation)
            if isinstance(self.agent, agents.FBDDPGAgent):
                z_correl += self.agent.compute_z_correl(time_step, meta)
            episode_step += 1
            self.global_step += 1
            # if self.agent.use_sequence: TODO
            #     obs_[episode_step] = time_step.observation
            #     mask[episode_step] = 1
            #     obs = (obs_,mask)
            # else:
            #     obs = time_step.observation
            # save checkpoint to reload
            if not self.global_frame % self.cfg.checkpoint_every:
                self.save_checkpoint(self._checkpoint_filepath)
            # try to save snapshot 
            if self.global_frame in self.cfg.snapshot_at:
                self.save_checkpoint(self._checkpoint_filepath.with_name(f'snapshot_{self.global_frame}.pt'))
                
        self.save_checkpoint(self._checkpoint_filepath)  # make sure we save the final checkpoint
        self.finalize()


@hydra.main(config_path='.', config_name='base_config', version_base="1.1")
def main(cfg: omgcf.DictConfig) -> None:
    # we assume cfg is a PretrainConfig (but actually not really)
    workspace = Workspace(cfg)  # type: ignore
    workspace.train()


if __name__ == '__main__':
    main()
