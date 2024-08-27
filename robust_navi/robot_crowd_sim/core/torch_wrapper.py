import torch
from robust_navi.robot_crowd_sim.core.simulator import RobotCrowdSim
class TorchWrapper:

    def __init__(self,env,device):

        self._env:RobotCrowdSim = env
        self.device = device

    def reset(self):

        obs = self._env.reset()

        return self._decode_obs(obs)
    
    def step(self,actions):
        actions["robot_action"] = actions["robot_action"].cpu().numpy()
        actions["crowd_action"]= actions["crowd_action"].cpu().numpy()

        (
            obs, 
            (robot_reward, crowd_reward), 
            (robot_discount,human_discount), 
            (info,crowd_info)
        ) = self._env.step(actions)


        return (
            self._decode_obs(obs),
            (robot_reward, crowd_reward), 
            (robot_discount,human_discount), 
            (info,crowd_info)
        )

    def _decode_obs(self,obs):

        crowd_obs = {}
        robot_obs = {}

        robot_obs["robot_state"] = torch.from_numpy(obs["robot_state"]).unsqueeze(0).to(self.device)
        robot_obs["robot_scan"] = torch.from_numpy(obs["robot_scan"]).unsqueeze(0).to(self.device)
        crowd_obs["full_state"] = torch.from_numpy(obs["full_state"]).unsqueeze(0).to(self.device)
        crowd_obs["crowd_preference"] = torch.from_numpy(obs["crowd_preference"]).unsqueeze(0).to(self.device)

        return robot_obs, crowd_obs
