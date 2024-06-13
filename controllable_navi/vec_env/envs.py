import os

# import gym
import numpy as np
import torch
# from gym.spaces.box import Box
# from gym.spaces.dict import Dict

from controllable_navi.vec_env.vec_env import VecEnvWrapper
from controllable_navi.vec_env.dummy_vec_env import DummyVecEnv
from controllable_navi.vec_env.shmem_vec_env import ShmemVecEnv

from controllable_navi import crowd_sim as crowd_sims
from controllable_navi import dmc


def make_env(cfg, seed, rank, task, human_num, discount, observation_type, max_episode_length, envNum=1):
    def _thunk():
        if envNum > 1:
            phase = 'train'
        else:
            phase = 'test'
        # 1. env = gym.make(env_id) #TODO
        env = dmc.EnvWrapper(
                crowd_sims.build_multirobotworld_task(
                    cfg,
                    task,
                    phase,
                    discount=discount,
                    observation_type=observation_type,
                    max_episode_length=max_episode_length))
        """
        2. seeding
        envSeed = seed + rank if seed is not None else None
        # environment.render_axis = ax
        env.thisSeed = envSeed
        env.nenv = envNum
        env.seed(seed + rank)
        """
        env.set_multi_env_seed(envNum,seed + rank)
        env.random_fix_human_num(human_num)
        print(env)

        return env

    return _thunk


def make_vec_envs(cfg,
                  seed,
                  num_processes,
                  gamma,
                  task,
                  human_nums, # random 
                  discount,
                  observation_type,
                  max_episode_length,
                  device,
                  wrap_pytorch=True, pretext_wrapper=False):
    envs = [
        make_env(cfg, seed, i, task, human_nums[i], discount, observation_type,max_episode_length,
                 envNum=num_processes)
        for i in range(num_processes)
    ]
    test = False if len(envs) > 1 else True

    if len(envs) > 1:
        envs = ShmemVecEnv(envs, human_nums,context='fork')
    else:
        envs = DummyVecEnv(envs,human_nums)
    # for collect data in supervised learning, we don't need to wrap pytorch
    if wrap_pytorch:
        envs = VecPyTorch(envs, device)
    if pretext_wrapper: #TODO for rnn
        pass
    #     if gamma is None: 
    #         envs = RNNVecEnv(envs, ret=False, ob=False, test=test)
    #     else:
    #         envs = RNNVecEnv(envs, gamma=gamma, ob=False, ret=False, test=test)

    return envs


# Checks whether done was caused my timit limits or not
# class TimeLimitMask(gym.Wrapper):
#     def step(self, action):
#         obs, rew, done, info = self.env.step(action)
#         if done and self.env._max_episode_steps == self.env._elapsed_steps:
#             info['bad_transition'] = True

#         return obs, rew, done, info

#     def reset(self, **kwargs):
#         return self.env.reset(**kwargs)


# # Can be used to test recurrent policies for Reacher-v2
# class MaskGoal(gym.ObservationWrapper):
#     def observation(self, observation):
#         if self.env._elapsed_steps > 0:
#             observation[-2:] = 0
#         return observation


# class TransposeObs(gym.ObservationWrapper):
#     def __init__(self, env=None):
#         """
#         Transpose observation space (base class)
#         """
#         super(TransposeObs, self).__init__(env)


# class TransposeImage(TransposeObs):
#     def __init__(self, env=None, op=[2, 0, 1]):
#         """
#         Transpose observation space for images
#         """
#         super(TransposeImage, self).__init__(env)
#         assert len(op) == 3, "Error: Operation, " + str(op) + ", must be dim3"
#         self.op = op
#         obs_shape = self.observation_space.shape
#         self.observation_space = Box(
#             self.observation_space.low[0, 0, 0],
#             self.observation_space.high[0, 0, 0], [
#                 obs_shape[self.op[0]], obs_shape[self.op[1]],
#                 obs_shape[self.op[2]]
#             ],
#             dtype=self.observation_space.dtype)

#     def observation(self, ob):
#         return ob.transpose(self.op[0], self.op[1], self.op[2])


class VecPyTorch(VecEnvWrapper):
    def __init__(self, venv, device):
        """Return only every `skip`-th frame"""
        super(VecPyTorch, self).__init__(venv)
        self.device = device
        # TODO: Fix data types

    def reset(self):
        obs = self.venv.reset()
        if isinstance(obs, dict):
            for key in obs:
                obs[key]=torch.from_numpy(obs[key]).to(self.device)
        else:
            obs = torch.from_numpy(obs).float().to(self.device)
        return obs

    def step_async(self, actions):
        # if isinstance(actions, torch.LongTensor):
        #     # Squeeze the dimension for discrete actions
        #     actions = actions.squeeze(1)
        actions = actions.cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        if isinstance(obs, dict):
            for key in obs:
                obs[key] = torch.from_numpy(obs[key]).to(self.device)
        else:
            obs = torch.from_numpy(obs).float().to(self.device)
        reward = torch.from_numpy(reward).unsqueeze(dim=1).float()
        return obs, reward, done, info

    # def render_traj(self, path, episode_num): 
    #     if self.venv.num_envs == 1:
    #         return self.venv.envs[0].env.render_traj(path, episode_num) #TODO
    #     else:
    #         for i, curr_env in enumerate(self.venv.envs):
    #             curr_env.env.render_traj(path, str(episode_num) + '.' + str(i))

