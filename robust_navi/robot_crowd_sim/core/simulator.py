import numpy as np
import dataclasses
from hydra.core.config_store import ConfigStore
import omegaconf
import typing as tp

from robust_navi.robot_crowd_sim.core.spaces import NaviObsSpace
from robust_navi.robot_crowd_sim.utils.agent import Agent
from robust_navi.robot_crowd_sim.utils.map import Map
from robust_navi.robot_crowd_sim.utils.sensor import LiDAR
from robust_navi.robot_crowd_sim.utils.render import Render
from robust_navi.robot_crowd_sim.utils.spawner import CircleSpawner
from robust_navi.robot_crowd_sim.utils.action import ActionVW,ActionXY
from robust_navi.robot_crowd_sim.utils.info import *

@dataclasses.dataclass
class RobotCrowdSimConfig:
    _target_: str = "robust_navi.robot_crowd_sim.core.simulator"
    name: str = "RobotCrowdSim"
    scenario: str = "default"
    map_size: float = 8
    with_static_obstacle: bool = True
    regen_map_every: int = 500

    n_laser:int=720
    laser_angle_resolute:float=0.008726646
    laser_min_range:float=0.0
    laser_max_range:float=4.0

    robot_radius: float = 0.3
    robot_v_pref: float = 1.0
    robot_visible: bool = True
    robot_rotation_constrain: float = np.pi/2

    human_radius: float = 0.3
    human_v_pref: float = 1.0
    human_visible: bool = True
    human_rotation_constrain: float = np.pi/2
    human_preference_vector_dim: int = omegaconf.II("crowd_meta_dim")#10

    penalty_collision: float = -1#2.
    penalty_backward: float = 0.2
    reward_goal: float = 2#10#2.
    goal_factor: float = 1#10
    goal_range: float = 0.3
    velo_factor: float = 0.2
    discomfort_dist: float = 1.0
    discomfort_penalty_factor: float = 1.0

    local_map_size: float = 8.0
    grid_size: float = 0.25

cs = ConfigStore.instance()
cs.store(group="crowd_sim_env", name="RobotCrowdSim", node=RobotCrowdSimConfig)

def build_robotcrowdworld_task(cfg:RobotCrowdSimConfig, phase,human_num,
                         discount=1.0,
                         nenv=1,thisSeed=0,
                         max_episode_length=200):
    return  RobotCrowdSim(cfg,phase,human_num,discount,max_episode_length,nenv,thisSeed)

class RobotCrowdSim:
    
    def __init__(self,
                 cfg:RobotCrowdSimConfig,
                 phase:str,
                 human_num:int,
                 discount:float=1.0,
                 max_episode_length:int=50,
                 nenv:int=1,
                 thisSeed:int=0
                 ) -> None:
        
        # setup env
        self.nenv = nenv
        self.thisSeed = thisSeed
        self.phase = phase

        # map setup
        # self._with_static_obstacle = cfg.with_static_obstacle TODO
        # self._regen_map_every = cfg.regen_map_every TODO
        self._map_size = cfg.map_size
        self._map = Map(cfg.map_size)
        self._time_step = 0.25
        # setup agents (robot and crowd)
        self._human_num = human_num
        self.robot = Agent(0,"unicycle",self._time_step,cfg.robot_visible,cfg.robot_v_pref,cfg.robot_radius,cfg.robot_rotation_constrain)
        self.crowd = [Agent(i+1,"holonomic",self._time_step,cfg.human_visible,cfg.human_v_pref,cfg.human_radius) for i in range(human_num)]
        for human in self.crowd:
            human.sample_random_attributes()
        self._human_preference_vector_dim = cfg.human_preference_vector_dim
        self._crowd_preference = None
        self._spawner = CircleSpawner(cfg.map_size,[self.robot]+self.crowd)

        # sensor setup
        self.n_laser=cfg.n_laser
        self.laser_angle_resolute=cfg.laser_angle_resolute
        self.laser_min_range=cfg.laser_min_range
        self.laser_max_range=cfg.laser_max_range
        self.sensor = LiDAR(cfg,self._map,self.robot,self.crowd)

        # setup reward
        self._discount = discount
        self._penalty_collision = cfg.penalty_collision
        self._reward_goal = cfg.reward_goal
        self._goal_factor = cfg.goal_factor
        self._goal_range = cfg.goal_range
        self._penalty_backward = cfg.penalty_backward
        self._velo_factor = cfg.velo_factor
        self._discomfort_penalty_factor = cfg.discomfort_penalty_factor
        self._discomfort_dist = cfg.discomfort_dist

        # setup state space
        self.observation_space = NaviObsSpace(

            shapes={
                'full_state':(7+8*self._human_num,),
                'crowd_preference':(self._human_preference_vector_dim,),
                'robot_state':(7,),
                'robot_scan':(720,)
            },
            dtype=np.float32,
            name='time_aware_state_and_scan'
        )

        # setup episode
        self.case_capacity = {'train': np.iinfo(np.uint32).max - 2000, 'val': 1000, 'test': 1000}
        self.case_size = {'train': np.iinfo(np.uint32).max - 2000, 'val': 2000,
                          'test': 1000}
        self.case_counter = {'train': 0, 'test': 0, 'val': 0}
        self.global_time : float = 0
        self._num_episode_steps : int = 0
        self._max_episode_length: int = max_episode_length

        # setup render
        self._render = Render(self._map,self.robot,self.crowd)


    def reset(self, seed: tp.Optional[int]=None):

        self._num_episode_steps = 0
        train_seed_begin = [0, 10, 100, 1000, 10000]
        val_seed_begin = [0, 10, 100, 1000, 10000]
        test_seed_begin = [0, 10, 100, 1000, 10000]
        base_seed = {'train': self.case_capacity['val'] + self.case_capacity['test'] + train_seed_begin[1],
                     'val': 0 + val_seed_begin[1], 'test': self.case_capacity['val']+test_seed_begin[2]+1000}

        if self.phase == "test" and seed is not None:
            self.random_seed = seed
        else:
            self.random_seed = base_seed[self.phase] + self.case_counter[self.phase] + self.thisSeed
        np.random.seed(self.random_seed)
        self.robot.set(0, -(self._map_size/2-1), 0, (self._map_size/2-1), 0, 0, np.pi/2)
        self.robot.task_done = False
        for agent in self.crowd:
            self._spawner.spawnAgent(agent)

        # TODO randomize crowd preference vector
        self._crowd_preference = self._initHumanPreferenceVector()
        # setup episode 
        self._num_episode_steps = 0
        self.global_time = 0
        self.case_counter[self.phase] = (self.case_counter[self.phase] + int(1*self.nenv)) % self.case_size[self.phase]
        

        return self._genObservation()
    
    def step(self, actions):
        
        robot_action = actions["robot_action"] # shape= (2,)
        crowd_actions = actions["crowd_action"].reshape(-1,2) # shape = (agent_num,2)

        # robot step
        robot_action = ActionVW(
                        ((robot_action[0]+1.)/2.)*self.robot.v_pref,
                        robot_action[1]*self.robot.rotation_constraint)
        if not self.robot.task_done:
            self.robot.step(robot_action)

        # crowd step
        human_actions = [
                            ActionXY(
                                crowd_actions[i,0]*self.crowd[i].v_pref,
                                crowd_actions[i,1]*self.crowd[i].v_pref
                                ) for i in range(len(self.crowd))
                        ]
        for i in range(len(self.crowd)):
            if self.crowd[i].task_done: continue
            self.crowd[i].step(human_actions[i])
        self.global_time+=self._time_step
        observation = self._genObservation()
        robot_human_collision, robot_human_min_dist = self._checkRobotCollision()
        human_human_collision, human_human_min_dist = self._checkCrowdCollision()

        robot_reward,robot_discount,info = self._calRobotReward(robot_human_collision, robot_human_min_dist)
        crowd_reward,human_discount,crowd_info = self._calCrowdReward(robot_human_collision, human_human_collision,human_human_min_dist)

        # on termination
        """
        col: crowd
        row: robot
                    goal    |   collision   |   timeout |   default
        goal        reset           -           reset       reset crowd
        
        collision   reset         reset         reset       reset crowd
                    crowd
        
        timeout     reset           -           reset       -

        default     reset           -           -           do nothing
                    robot
        """
        # need reset either agent when it completes the task but its counterpart do not
        if robot_discount==0 and human_discount!=0:
            self._reset_robot()
        if robot_discount!=0 and human_discount==0:
            self._reset_crowd(human_human_collision)

        return (
            observation, 
            (robot_reward, crowd_reward), 
            (robot_discount,human_discount), 
            (info,crowd_info)
        )
    
    def observation_spec(self,):
        return self.observation_space.shape_dict
    
    def action_spec(self,):
        return {'robot_action':(2,),'crowd_action':(2*self._human_num,)}
    
    def render(self, mode="human"):
        return self._render.rend(mode,self.global_time)
    
    def close(self):

        return
    
    def _reset_crowd(self,human_human_collision):
        if any(human_human_collision): #human human collision
            for i in range(len(self.crowd)-1):
                for j in range(i+1,len(self.crowd)):
                    xy = np.array([self.crowd[i].px-self.crowd[j].px,self.crowd[i].py-self.crowd[j].py])
                    direction_ji = xy/np.linalg.norm(xy) 
                    dist = np.linalg.norm(xy)-self.crowd[i].radius-self.crowd[j].radius
                    if dist<0:
                        self.crowd[i].px += direction_ji[0]*2*dist
                        self.crowd[i].py += direction_ji[1]*2*dist
        else: # all human reach goal before robot
            for agent in self.crowd:
                agent.set(agent.px,agent.py,
                       -agent.px,-agent.py,
                       0,0,np.arctan2(-agent.py,-agent.px))
            self._crowd_preference = self._initHumanPreferenceVector()
        for agent in self.crowd:
            agent.task_done = False
        return 
    
    def _reset_robot(self):
        # in case robot reach goal first
        self.robot.set(self.robot.px,self.robot.py,
                       -self.robot.px,-self.robot.py,
                       0,0,np.arctan2(-self.robot.py,-self.robot.px))
        self.robot.task_done = False
        return
    
    def _initHumanPreferenceVector(self):
        task = np.random.randn(self._human_preference_vector_dim)
        task = task / np.linalg.norm(task)
        return task
    
    def _genObservation(self):
        
        full_state = [self._num_episode_steps/self._max_episode_length,
                     np.log(self._max_episode_length)]+list(self.robot.get_observable_state())
        for human in self.crowd:
            full_state += list(human.get_full_state())

        dxy = np.array(self.robot.get_goal_position())-np.array(self.robot.get_position())
        dg = np.linalg.norm(dxy)
        goal_direction = np.arctan2(dxy[1],dxy[0])
        hf = (self.robot.theta-goal_direction)% (2 * np.pi)
        if hf > np.pi:
            hf -= 2 * np.pi
        # transform dxy to base frame
        vx = (self.robot.vx * np.cos(goal_direction) + self.robot.vy * np.sin(goal_direction))
        vy = (self.robot.vy * np.cos(goal_direction) - self.robot.vx * np.sin(goal_direction)) 
        
        scan, scan_end = self.sensor.getScan()
        ob = {}
        ob['full_state'] = np.array(full_state,dtype=np.float32)
        ob['crowd_preference'] = self._crowd_preference.astype(np.float32)
        ob['robot_state'] = np.array([self._num_episode_steps/self._max_episode_length,
                                            np.log(self._max_episode_length),
                                            dg, hf,vx,vy,self.robot.radius],dtype=np.float32)
        ob['robot_scan'] = np.clip(scan,self.laser_min_range,self.laser_max_range).astype(np.float32)

        return ob
    
    def _checkRobotCollision(self):
        collide = [False]*self._human_num
        min_dist = np.inf
        for i,agent in enumerate(self.crowd):
            dist = np.linalg.norm([self.robot.px-agent.px,self.robot.py-agent.py])-self.robot.radius-agent.radius
            if dist < min_dist:
                min_dist = dist
            if dist < 0:
                collide[i] = True
        return collide, min_dist
    
    def _checkCrowdCollision(self):
        min_dist = [np.inf]*self._human_num
        collide = [False]*self._human_num
        for i,agent in enumerate(self.crowd):
            for j, other in enumerate(self.crowd):
                if agent.id==other.id:
                    continue
                dist = np.linalg.norm([agent.px-other.px,agent.py-other.py])-agent.radius-other.radius
                if dist < min_dist[i]:
                    min_dist[i] = dist
                if dist < 0:
                    collide[i]=True
                    collide[j]=True
        return collide,min_dist
    
    def _calRobotReward(self,robot_human_collision, robot_human_min_dist):

        if any(robot_human_collision):
            reward = self._penalty_collision
            discount = 0.0
            episode_info = Collision()
            self.robot.task_done = True
        elif self.robot.dg<self._goal_range:
            reward = self._reward_goal* (1-self.global_time/self._max_episode_length)
            discount = 0.0
            episode_info = ReachGoal()
            self.robot.task_done = True
        elif self._num_episode_steps >= self._max_episode_length:
            reward = -self._reward_goal
            discount = 0.0
            episode_info = Timeout()
            self.robot.task_done = True
        else:
            reward = self._goal_factor*(self.robot.prev_dg-self.robot.dg)
            discount = self._discount
            episode_info = Nothing()
            if robot_human_min_dist<self._discomfort_dist:
                reward += self._discomfort_penalty_factor*(robot_human_min_dist-self._discomfort_dist)
                discount = self._discount
                episode_info = Discomfort(robot_human_min_dist)

        return reward, discount, episode_info
    
    def _calCrowdReward(self,robot_human_collision, human_human_collision,human_human_min_dist):
        
        reach_goal = []
        for human in self.crowd:
            reached_goal = human.dg<self._goal_range
            if not human.task_done and reach_goal:
                human.task_done = True
            reach_goal.append(reached_goal)

        if any(robot_human_collision) or any(human_human_collision):
            reward = self._penalty_collision
            discount = 0.0
            episode_info = Collision()
            for i in range(len(human_human_collision)):
                if human_human_collision[i]:
                    self.crowd[i].task_done=True
            for i in range(len(robot_human_collision)):
                if robot_human_collision[i]:
                    self.crowd[i].task_done=True
        elif all(reach_goal):
            reward = self._reward_goal #* (1-self.global_time/self._max_episode_length)
            discount = 0.0
            episode_info = ReachGoal()
        elif self._num_episode_steps >= self._max_episode_length:
            reward = 0.0#-self._reward_goal
            discount = 0.0
            for human in self.crowd:
                human.task_done = True
            episode_info = Timeout()
        else:
            reward = 0.0
            discount = self._discount
            for i,human in enumerate(self.crowd):
                if human.task_done:continue
                reward += self._goal_factor * (human.prev_dg-human.dg)/self._human_num
                if human_human_min_dist[i]<self._discomfort_dist:
                    reward += self._discomfort_penalty_factor*(human_human_min_dist[i]-self._discomfort_dist)/self._human_num
            if min(human_human_min_dist)<self._discomfort_dist:
                episode_info = Discomfort(min(human_human_min_dist))
            else:
                episode_info = Nothing()
        return reward, discount, episode_info