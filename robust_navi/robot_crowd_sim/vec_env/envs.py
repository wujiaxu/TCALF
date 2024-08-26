import os
import numpy as np
import torch

from robust_navi.robot_crowd_sim.vec_env.vec_env import VecEnvWrapper
from robust_navi.robot_crowd_sim.vec_env.shmem_vec_env import ShmemVecEnv

from robust_navi.robot_crowd_sim.core.simulator import build_robotcrowdworld_task

def make_env(cfg, seed, rank, phase, human_num, discount, max_episode_length, envNum=1):
    def _thunk():
        if phase == 'test' or phase == 'val':
            assert envNum==1

        env = build_robotcrowdworld_task(
                    cfg,
                    phase,
                    human_num,
                    discount=discount,
                    nenv=envNum,
                    thisSeed=seed + rank,
                    max_episode_length=max_episode_length,
                    )
        print(env)

        return env

    return _thunk


def make_vec_envs(cfg,
                  seed,
                  num_processes,
                  agent_nums,
                  phase,
                  discount,
                  max_episode_length,
                  device):
    
    envs = [
        make_env(cfg, seed, i, phase, agent_nums[i], discount,max_episode_length, envNum=num_processes)
        for i in range(num_processes)
    ]

    assert len(envs) > 1
    envs = ShmemVecEnv(envs, agent_nums,context='fork')
    
    envs = VecPyTorch(envs, device)

    return envs


class VecPyTorch(VecEnvWrapper):
    def __init__(self, venv, device):
        """Return only every `skip`-th frame"""
        super(VecPyTorch, self).__init__(venv)
        self.device = device
        # TODO: Fix data types

    def reset(self):
        crowd_obs, robot_obs = self.venv.reset()
        for agent_num in crowd_obs:
            for key in crowd_obs[agent_num].keys():
                crowd_obs[agent_num][key] = torch.from_numpy(crowd_obs[agent_num][key]).to(self.device)
        for key in robot_obs:
            robot_obs[key]=torch.from_numpy(robot_obs[key]).to(self.device)
        return crowd_obs, robot_obs

    def step_async(self, actions):
        """
        actions (dict): 
            robot action (torch tensor): env_numX2 
            crowd action (dict): 
                agent_num=1: env(1)X2
                2: env(2)X2
                ...
                N:env(N)X2
        """
        actions["robot_action"] = actions["robot_action"].cpu().numpy()
        for k in actions["crowd_action"].keys():
            actions["crowd_action"][k] = actions["crowd_action"][k].cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        (
            (crowd_obs, robot_obs), 
            (robot_rews, crowd_rews_dict), 
            (robot_discounts,human_discounts_dict), 
            (infos,crowd_infos_dict)
        ) = self.venv.step_wait()
        for agent_num in crowd_obs:
            for key in crowd_obs[agent_num].keys():
                crowd_obs[agent_num][key] = torch.from_numpy(crowd_obs[agent_num][key]).to(self.device)
        for key in robot_obs:
            robot_obs[key] = torch.from_numpy(robot_obs[key]).to(self.device)
        robot_rews = torch.from_numpy(robot_rews).unsqueeze(dim=1).float()
        robot_discounts = torch.from_numpy(robot_discounts).unsqueeze(dim=1).float()
        for key in crowd_rews_dict:
            crowd_rews_dict[key] = torch.from_numpy(crowd_rews_dict[key]).to(self.device)
        for key in human_discounts_dict:
            human_discounts_dict[key] = torch.from_numpy(human_discounts_dict[key]).to(self.device)
        return (
            (crowd_obs, robot_obs), 
            (robot_rews, crowd_rews_dict), 
            (robot_discounts,human_discounts_dict), 
            (infos,crowd_infos_dict)
        )

