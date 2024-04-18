from url_benchmark.goals import BaseReward
import numpy as np
import typing as tp

def get_reward_function(name: str) -> BaseReward:
    if name.split("_")[0] == "crowdnavi":
        return CrowdNaviReward(name.split("_")[1])
    else:
        raise NotImplementedError

class CrowdNaviReward(BaseReward):

    def __init__(self, string: str) -> None:
        tasks_specifications = {
                'PointGoalNavi':{
                    "reward_func_ids":0,
                    "discomfort_dist" : 0.2,
                    "speed_limit":1.0,
                },
                'PassLeftSide':{
                    "reward_func_ids": 2,
                    "forbiden_zone_y":0.6,
                    "discomfort_dist" : 0.2,
                    "speed_limit":1.0,
                },
                'PassRightSide':{
                    "reward_func_ids": 2,
                    "forbiden_zone_y" : -0.6,
                    "discomfort_dist" : 0.2,
                    "speed_limit":1.0,
                },
                'FollowWall':{
                    "reward_func_ids": 3,
                    "discomfort_dist" : 0.2,
                    "speed_limit":1.0,
                },
                'AwayFromHuman':{
                    "reward_func_ids": 0,
                    "discomfort_dist" : 0.8,
                    "speed_limit":1.0,
                },
                'LowSpeed':{
                    "reward_func_ids": 0,
                    "speed_limit":0.5,
                }
                            }
        self.task = tasks_specifications[string]
        
    def from_env(self,env):

        return env.base_env.last_reward
    
    def compute_reward(self, obs, action, next_obs,phy):
        reward = 0

        human_num = int(phy[0]-1)
        robot_state = phy[1:6]
        human_states = phy[6:6+5*human_num].reshape(-1,5)

        if self.task["reward_func_ids"] == 0:
            if np.sqrt(next_obs[-6]**2+next_obs[-5]**2)<0.3:
                reward = 0.25
            else:
                reward += 0.2 * (np.sqrt(obs[-6]**2+obs[-5]**2)-np.sqrt(next_obs[-6]**2+next_obs[-5]**2))
                # occu_map = obs[:4096].reshape((64,64))
                # min_dist = self.task["discomfort_dist"]
                # if min_dist == 0.2: TODO
            if action[0]>self.task["speed_limit"]:
                reward -= 0.2 * (action[0]-self.task["speed_limit"])
        elif self.task["reward_func_ids"] == 1:
    
            # dg = next_obs[-6]
            # dgtheta = next_obs[-3]
            # dgv = next_obs[-2]
            # reward = 0.2*(np.exp(-2*dg)+np.exp(-1*dgtheta))+0.2*np.exp(-0.1*dgv)
            pass

        elif self.task["reward_func_ids"] == 2:
            dg = np.sqrt(next_obs[-6]**2+next_obs[-5]**2)
            vx = robot_state[2]
            vy = robot_state[3]
            px = robot_state[0]
            py = robot_state[1]
            if dg<0.3:
                reward = 0.25
            else:
                reward += 0.2 *  (np.sqrt(obs[-6]**2+obs[-5]**2)-np.sqrt(next_obs[-6]**2+next_obs[-5]**2))
                # occu_map = obs[:4096].reshape((64,64))
                # min_dist = self.task["discomfort_dist"]
                # if min_dist == 0.2: TODO
            if action[0]>self.task["speed_limit"]:
                reward -= 0.2 * (action[0]-self.task["speed_limit"])
            if dg > 3:
                rot = np.arctan2(vy,vx)
                for i in range(human_states.shape[0]):
                    px_other = (human_states[i,0]-px)*np.cos(rot) + (human_states[i,1]-py)*np.sin(rot)
                    py_other = -(human_states[i,0]-px)*np.sin(rot) + (human_states[i,1]-py)*np.cos(rot)
                    # da_other = np.sqrt(px_other*px_other + py_other*py_other)

                    vx_other = human_states[i,2]*np.cos(rot) + human_states[i,3]*np.sin(rot)
                    vy_other = -human_states[i,2]*np.sin(rot) + human_states[i,3]*np.cos(rot)
                    # v_other =np.sqrt(vx_other*vx_other + vy_other*vy_other)

                    phi_other = np.arctan2(vy_other,vx_other)
                    # psi_rot = np.arctan2(vy_other-vy,vx_other-vx)
                    direction_diff = phi_other  # because ego-centric ego desirable direction psi=0

                    # passing 
                    if px_other>1 and px_other<4 and abs(direction_diff)>3.*np.pi/4.:
                        if self.task["forbiden_zone_y"]<0 and py_other<0 and py_other>-2: #right
                            reward -= 0.05
                        if self.task["forbiden_zone_y"]>0 and py_other>0 and py_other<2: #left
                            reward -= 0.05
        elif self.task["reward_func_ids"] == 3:
            pass
        else:
            raise NotImplementedError

        return reward
    