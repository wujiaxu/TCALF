# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from pathlib import Path
import sys
base = Path(__file__).absolute().parents[1]
# we need to add base repo to be able to import url_benchmark
# we need to add url_benchmarl to be able to reload legacy checkpoints
for fp in [base, base / "url_benchmark",base / "controllable_navi"]:
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


os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
# if the default egl does not work, you may want to try:
# export MUJOCO_GL=glfw
os.environ['MUJOCO_GL'] = os.environ.get('MUJOCO_GL', 'egl')

import hydra
from hydra.core.config_store import ConfigStore
import numpy as np
import torch
import wandb
import omegaconf as omgcf
# from dm_env import specs

from url_benchmark import dmc
from dm_env import specs
from url_benchmark import utils
# from url_benchmark import goals as _goals
from controllable_navi import goals as _goals
from controllable_navi.logger import Logger
from controllable_navi.in_memory_replay_buffer import ReplayBuffer
from url_benchmark.video import TrainVideoRecorder, VideoRecorder
# from url_benchmark import agent as agents
from controllable_navi import agent as agents
from controllable_navi import crowd_sim as crowd_sims
from controllable_navi.crowd_sim.utils.info import *
# from url_benchmark.d4rl_benchmark import D4RLReplayBufferBuilder, D4RLWrapper
# from url_benchmark.gridworld.env import build_gridworld_task

logger = logging.getLogger(__name__)
torch.backends.cudnn.benchmark = True
# os.environ['WANDB_MODE']='offline'

# from url_benchmark.dmc_benchmark import PRIMAL_TASKS


# # # Config # # #

@dataclasses.dataclass
class Config:
    agent: tp.Any
    crowd_sim: tp.Any 
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
) -> tp.Union[agents.FBDDPGAgent, agents.DDPGAgent]:
    cfg.obs_type = obs_type
    cfg.obs_shape = obs_spec.shape
    cfg.action_shape = (action_spec.num_values, ) if isinstance(action_spec, specs.DiscreteArray) \
        else action_spec.shape
    cfg.num_expl_steps = num_expl_steps
    # reference https://zenn.dev/gesonanko/articles/417d43669cf2af (use hydra to instantiate an object)
    return hydra.utils.instantiate(cfg)


C = tp.TypeVar("C", bound=Config) # Can be any subtype of Config


def _update_legacy_class(obj: tp.Any, classes: tp.Sequence[tp.Type[tp.Any]]) -> tp.Any:
    """Updates a legacy class (eg: agent.FBDDPGAgent) to the new
    class (url_benchmark.agent.FBDDPGAgent)

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
        batch_size = 0
        while batch_size < num_steps:
            batch = workspace.replay_loader.sample(workspace.cfg.batch_size, custom_reward=custom_reward)
            batch = batch.to(workspace.cfg.device)
            obs_list.append(batch.next_goal if workspace.cfg.goal_space is not None else batch.next_obs)
            reward_list.append(batch.reward)
            batch_size += batch.next_obs.size(0)
        obs, reward = torch.cat(obs_list, 0), torch.cat(reward_list, 0)  # type: ignore
        obs_t, reward_t = obs[:num_steps], reward[:num_steps]
        return workspace.agent.infer_meta_from_obs_and_rewards(obs_t, reward_t)
        # else:
        #     assert isinstance(workspace.agent, agents.SFSVDAgent)
        #     obs_list, reward_list, action_list = [], [], []
        #     batch_size = 0
        #     while batch_size < workspace.agent.cfg.num_inference_steps:
        #         batch = workspace.replay_loader.sample(workspace.cfg.batch_size, custom_reward=custom_reward)
        #         batch = batch.to(workspace.cfg.device)
        #         obs_list.append(batch.goal if workspace.cfg.goal_space is not None else batch.obs)
        #         action_list.append(batch.action)
        #         reward_list.append(batch.reward)
        #         batch_size += batch.next_obs.size(0)
        #     obs, reward, action = torch.cat(obs_list, 0), torch.cat(reward_list, 0), torch.cat(action_list, 0)  # type: ignore
        #     obs_t, reward_t, action_t = obs[:workspace.agent.cfg.num_inference_steps], reward[:workspace.agent.cfg.num_inference_steps],\
        #         action[:workspace.agent.cfg.num_inference_steps]
        #     return workspace.agent.infer_meta_from_obs_action_and_rewards(obs_t, action_t, reward_t)

    # if workspace.cfg.goal_space is not None:
    #     funcs = _goals.goals.funcs.get(workspace.cfg.goal_space, {})
    #     if workspace.cfg.task in funcs:
    #         g = funcs[workspace.cfg.task]()
    #         return workspace.agent.get_goal_meta(g)
    return workspace.agent.infer_meta(workspace.replay_loader)


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
        # TODO debug RuntimeError: mat1 and mat2 shapes cannot be multiplied (1024x4113 and 4114x1024) aps.critic
        # print(self.train_env.observation_spec(),self.train_env.action_spec())
        # crowd navi
        #Array(shape=(4102,), dtype=dtype('float32'), name='observation_occupancy_map') 
        # Array(shape=(2,), dtype=dtype('float32'), name='action')
        # walker
        # Array(shape=(24,), dtype=dtype('float32'), name='observation') 
        # BoundedArray(shape=(6,), dtype=dtype('float32'), name='action', minimum=-1.0, maximum=1.0)
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
                                            max_episode_length=201)
        cam_id = 0 # if 'quadruped' not in self.domain else 2

        self.video_recorder = VideoRecorder(self.work_dir if cfg.save_video else None,
                                            camera_id=cam_id, use_wandb=self.cfg.use_wandb)

        self.timer = utils.Timer()
        self.global_step = 0
        self.global_episode = 0
        self.eval_rewards_history: tp.List[float] = []
        self._checkpoint_filepath = self.work_dir / "models" / "latest.pt"
        # print(self._checkpoint_filepath)<-/root/controllable_agent/url_benchmark/exp_local/2024.04.01/131258_aps_crowdnavi_PointGoalNavi_online/models/latest.pt  this is just a file name, no file till checkpoint
        if self._checkpoint_filepath.exists():
            self.load_checkpoint(self._checkpoint_filepath)
        elif cfg.load_model is not None:
            self.load_checkpoint(cfg.load_model, exclude=["replay_loader"])

        self.reward_cls: tp.Optional[_goals.BaseReward] = None

    def _make_env(self,phase='train') -> dmc.EnvWrapper:
        # cfg = self.cfg
        if self.domain == "crowdnavi":
            
            return dmc.EnvWrapper(crowd_sims.build_crowdworld_task(self.cfg.crowd_sim,self.cfg.task.split('_')[1],phase,discount=self.cfg.discount,observation_type=self.cfg.obs_type))
        else:
            raise NotImplementedError
        # return dmc.make(cfg.task, cfg.obs_type, cfg.frame_stack, cfg.action_repeat, cfg.seed,
        #                 goal_space=cfg.goal_space, append_goal_to_observation=cfg.append_goal_to_observation)

    @property
    def global_frame(self) -> int:
        return self.global_step * self.cfg.action_repeat

    def _make_custom_reward(self, seed: int) -> tp.Optional[_goals.BaseReward]:
        """Creates a custom reward function if provided in configuration
        else returns None
        """
        if self.cfg.custom_reward is None:
            return None
        return _goals.get_reward_function(self.cfg.custom_reward)

    def eval(self) -> None:
        step, episode = 0, 0
        success_num = 0
        eval_until_episode = utils.Until(self.cfg.num_eval_episodes)
        physics_agg = dmc.PhysicsAggregator()
        rewards: tp.List[float] = []
        normalized_scores: tp.List[float] = []
        meta = _init_eval_meta(self)  # Don't work
        z_correl = 0.0
        # is_d4rl_task = self.cfg.task.split('_')[0] == 'd4rl'
        # TODO add info to calculate success rate slp collision rate navi time
        actor_success: tp.List[float] = []
        while eval_until_episode(episode):
            time_step = self.eval_env.reset()
            episode_step = 0
            # create custom reward if need be (if field exists)
            seed = 12 * self.cfg.num_eval_episodes + len(rewards)
            custom_reward = self._make_custom_reward(seed=seed)
            if custom_reward is not None:
                meta = _init_eval_meta(self, custom_reward)
            # if self.domain == "grid":
            #     meta = _init_eval_meta(self)
            total_reward = 0.0
            self.video_recorder.init(self.eval_env, enabled=True) #enabled=(episode == 0) force the recorder only save episode 0
            while not time_step.last():
                with torch.no_grad(), utils.eval_mode(self.agent):
                    action = self.agent.act(time_step.observation,
                                            meta,
                                            self.global_step,
                                            eval_mode=True)
                time_step = self.eval_env.step(action)
                physics_agg.add(self.eval_env)
                self.video_recorder.record(self.eval_env)
                # for legacy reasons, we need to check the name :s
                if isinstance(self.agent, agents.FBDDPGAgent):
                    if self.agent.cfg.additional_metric:
                        z_correl += self.agent.compute_z_correl(time_step, meta)
                        actor_success.extend(self.agent.actor_success)
                if custom_reward is not None:
                    time_step.reward = custom_reward.from_env(self.eval_env)
                # total_reward += time_step.reward
                total_reward+= self.cfg.discount**episode_step*time_step.reward
                step += 1
                episode_step+=1
            # if is_d4rl_task:
            #     normalized_scores.append(self.eval_env.get_normalized_score(total_reward))
            success_num = success_num+1 if time_step.info.contain(ReachGoal()) else success_num #this seemly no working!! TODO debug
            rewards.append(total_reward)
            episode += 1
            self.video_recorder.save(f'{self.global_frame}_{episode}.mp4')

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
            log('success rate', float(success_num)/self.cfg.num_eval_episodes)
            if actor_success:
                log('actor_sucess', float(np.mean(actor_success)))
            if isinstance(self.agent, agents.FBDDPGAgent):
                log('z_norm', np.linalg.norm(meta['z']).item())
            for key, val in physics_agg.dump():
                log(key, val)

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

    def finalize(self) -> None:
        print("Running final test", flush=True)
        repeat = 1 #self.cfg.final_tests
        if not repeat:
            return

        # if self.cfg.custom_reward == "maze_multi_goal":
        #     eval_hist = self.eval_rewards_history
        #     rewards = {}
        #     self.eval_rewards_history = []
        #     self.cfg.num_eval_episodes = repeat
        #     self.eval_maze_goals()
        #     rewards["rewards"] = self.eval_rewards_history
        #     self.eval_rewards_history = eval_hist  # restore
        # else:
        domain_tasks = {
            # "cheetah": ['walk', 'walk_backward', 'run', 'run_backward'],
            # "quadruped": ['stand', 'walk', 'run', 'jump'],
            # "walker": ['stand', 'walk', 'run', 'flip'],
            "crowdnavi":[
                        #  'PointGoalNavi',
                         'PassLeftSide',
                         'PassRightSide',
                        #  'AwayFromHuman',
                        #  'LowSpeed'
                         ] #'FollowWall',
        }
        if self.domain not in domain_tasks:
            return
        eval_hist = self.eval_rewards_history
        rewards = {}
        for name in domain_tasks[self.domain]:
            self.global_step+=1
            task = "_".join([self.domain, name])
            self.cfg.task = task
            self.cfg.custom_reward = task  # for the replay buffer
            self.cfg.seed += 1  # for the sake of avoiding similar seeds
            self.eval_env = self._make_env(phase='test')
            self.eval_rewards_history = []
            self.cfg.num_eval_episodes = 10
            for _ in range(repeat):
                self.eval()
            rewards[task] = self.eval_rewards_history
        self.eval_rewards_history = eval_hist  # restore
        with (self.work_dir / "test_rewards.json").open("w") as f:
            json.dump(rewards, f)


class Workspace(BaseWorkspace[PretrainConfig]):
    def __init__(self, cfg: PretrainConfig) -> None:
        super().__init__(cfg)
        self.train_video_recorder = TrainVideoRecorder(self.work_dir if cfg.save_train_video else None,
                                                       camera_id=self.video_recorder.camera_id, use_wandb=self.cfg.use_wandb)
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

        episode_step, episode_reward, z_correl = 0, 0.0, 0.0
        time_step = self.train_env.reset()
        meta = self._init_meta()
        self.replay_loader.add(time_step, meta)
        self.train_video_recorder.init(time_step.observation)
        metrics = None
        physics_agg = dmc.PhysicsAggregator()

        while train_until_step(self.global_step):
            if time_step.last():
                success_rate = 1.0 if time_step.info.contain(ReachGoal()) else 0.0
                task_info = time_step.info.task_info
                start_goal_dist = np.sqrt((task_info["gx"]-task_info["sx"])**2+(task_info["gy"]-task_info["sy"])**2)
                # if start_goal_dist < 6:
                #     print(task_info)
                #     raise ValueError
                self.global_episode += 1
                self.train_video_recorder.save(f'{self.global_frame}.mp4')
                # wait until all the metrics schema is populated
                if metrics is not None:
                    # log stats
                    elapsed_time, total_time = self.timer.reset()
                    episode_frame = episode_step * self.cfg.action_repeat
                    with self.logger.log_and_dump_ctx(self.global_frame,
                                                      ty='train') as log:
                        log('fps', episode_frame / elapsed_time)
                        log('total_time', total_time)
                        log('episode_reward', episode_reward)
                        log('episode_length', episode_frame)
                        log('episode', self.global_episode)
                        log('buffer_size', len(self.replay_loader))
                        log('step', self.global_step)
                        log('z_correl', z_correl)
                        log('start goal dist',start_goal_dist)
                        log('final goal dist',time_step.observation[-5])
                        log('success rate',success_rate)
                        # TODO record success rate

                        for key, val in physics_agg.dump():
                            log(key, val)
                if self.cfg.use_hiplog and self.logger.hiplog.content:
                    self.logger.hiplog.write()

                # reset env
                time_step = self.train_env.reset()
                meta = self._init_meta()
                self.replay_loader.add(time_step, meta)
                self.train_video_recorder.init(time_step.observation)
                # try to save snapshot
                if self.global_frame in self.cfg.snapshot_at:
                    self.save_checkpoint(self._checkpoint_filepath.with_name(f'snapshot_{self.global_frame}.pt'))
                episode_step = 0
                episode_reward = 0.0
                z_correl = 0.0

            # try to evaluate
            if eval_every_step(self.global_step):
                self.logger.log('eval_total_time', self.timer.total_time(),
                                self.global_frame)
                # if self.cfg.custom_reward == "maze_multi_goal":
                #     self.eval_maze_goals()
                # elif self.domain == "grid":
                #     self.eval_grid_goals()
                # else:
                self.eval()
            # TODO consider whether comment out meta update is ok? (currently I want one episode one z, so I don't update meta during episode)
            # meta = self.agent.update_meta(meta, self.global_step, time_step, finetune=False, replay_loader=self.replay_loader)
            # sample action
            with torch.no_grad(), utils.eval_mode(self.agent):
                action = self.agent.act(time_step.observation,
                                        meta,
                                        self.global_step,
                                        eval_mode=False)
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
            time_step = self.train_env.step(action)
            physics_agg.add(self.train_env)
            episode_reward += self.cfg.discount**episode_step*time_step.reward #NEW: consider discount
            self.replay_loader.add(time_step, meta)
            self.train_video_recorder.record(time_step.observation)
            if isinstance(self.agent, agents.FBDDPGAgent):
                z_correl += self.agent.compute_z_correl(time_step, meta)
            episode_step += 1
            self.global_step += 1
            # save checkpoint to reload
            if not self.global_frame % self.cfg.checkpoint_every:
                self.save_checkpoint(self._checkpoint_filepath)
        self.save_checkpoint(self._checkpoint_filepath)  # make sure we save the final checkpoint
        self.finalize()


@hydra.main(config_path='.', config_name='base_config', version_base="1.1")
def main(cfg: omgcf.DictConfig) -> None:
    # we assume cfg is a PretrainConfig (but actually not really)
    workspace = Workspace(cfg)  # type: ignore
    workspace.train()


if __name__ == '__main__':
    main()
