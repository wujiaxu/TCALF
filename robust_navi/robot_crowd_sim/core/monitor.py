from robust_navi.robot_crowd_sim.utils.info import *
import numpy as np

class InfoMonitor:

    def __init__(self,env_num,human_nums_env,time_step):

        self._nenv = env_num
        self.human_nums = human_nums_env
        self._time_step = time_step
        self.global_step = 0
        
        self.robot_step = [0]*env_num
        self.robot_episode = 0
        self.robot_success = 0
        self.robot_collide = 0
        self.robot_timeout = 0
        self.robot_invasion = 0
        self.robot_time = []

        self.crowd_step = {}
        for human_num in human_nums_env:
            if human_num not in self.crowd_step:
                self.crowd_step[human_num] = []
            self.crowd_step[human_num].append(0)
        self.crowd_episode = 0
        self.crowd_success = 0
        self.crowd_collide = 0
        self.crowd_timeout = 0
        self.crowd_invasion = 0
        self.crowd_time = []

    def evaluateRobotInfo(self):

        return (
            self.robot_success/self.robot_episode,
            self.robot_collide/self.robot_episode,
            self.robot_timeout/self.robot_episode,
            2*self.robot_invasion/self.global_step,
            np.mean(self.robot_time),
            np.std(self.robot_time)
        )

    def evaluateCrowdInfo(self):

        return (
            self.crowd_success/self.crowd_episode,
            self.crowd_collide/self.crowd_episode,
            self.crowd_timeout/self.crowd_episode,
            2*self.crowd_invasion/self.global_step,
            np.mean(self.crowd_time),
            np.std(self.crowd_time)
        )
    
    def saveInfoVecEnv(self,infos):
        robot_infos, crowd_infos_dict = infos
        self.saveRobotInfoList(robot_infos)
        self.saveCrowdInfoDict(crowd_infos_dict)

    def saveRobotInfoList(self,infos):
        for i, info in enumerate(infos):
            self.global_step+=1
            self.robot_step[i]+=self._time_step
            if isinstance(info,Nothing):
                continue
            if isinstance(info,Discomfort):
                self.robot_invasion+=1
                continue
            if isinstance(info,ReachGoal):
                self.robot_time.append(self.robot_step[i])
                self.robot_success+=1
                self.robot_step[i] = 0
            elif isinstance(info,Collision):
                self.robot_collide+=1
                self.robot_step[i] = 0
            elif isinstance(info,Timeout):
                self.robot_timeout+=1
                self.robot_step[i] = 0
            else:
                raise ValueError
            self.robot_episode+=1
        return 
    
    def saveCrowdInfoDict(self,crowd_infos_dict):

        for human_num in crowd_infos_dict:
            infos = crowd_infos_dict[human_num]
            for i, info in enumerate(infos):
                self.global_step+=1
                self.crowd_step[human_num][i]+=self._time_step
                if isinstance(info,Nothing):
                    continue
                if isinstance(info,Discomfort):
                    self.crowd_invasion+=1
                    continue
                if isinstance(info,ReachGoal):
                    self.crowd_time.append(self.crowd_step[human_num][i])
                    self.crowd_success+=1
                    self.crowd_step[human_num][i] = 0
                elif isinstance(info,Collision):
                    self.crowd_collide+=1
                    self.crowd_step[human_num][i] = 0
                elif isinstance(info,Timeout):
                    self.crowd_timeout+=1
                    self.crowd_step[human_num][i] = 0
                else:
                    raise ValueError
                self.crowd_episode+=1
        return
