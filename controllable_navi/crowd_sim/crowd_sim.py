# import logging
# import random
# import math
import typing as tp
import dataclasses
import shapely
from shapely.geometry import Polygon, LineString, MultiPolygon, LinearRing,Point

import matplotlib.pyplot as plt
from matplotlib import collections as mc
import matplotlib.lines as mlines
from matplotlib import patches
import numpy as np
from numpy.linalg import norm
import copy

from torch import Value

from controllable_navi.crowd_sim.C_library.motion_plan_lib import *
# from controllable_navi.crowd_sim.policy.policy_factory import policy_factory
# from controllable_navi.crowd_sim.utils.state import tensor_to_joint_state, JointState,ExFullState
from controllable_navi.crowd_sim.utils.action import ActionVW
from controllable_navi.crowd_sim.utils.human import Human
from controllable_navi.crowd_sim.utils.robot import Robot
from controllable_navi.crowd_sim.utils.info import *
# from controllable_navi.crowd_sim.utils.utils import point_to_segment_dist
from controllable_navi.crowd_sim.utils.obstacle import *

import controllable_navi.dm_env_light as dm_env
from controllable_navi.dm_env_light import specs
import enum
from controllable_navi.dmc import ExtendedTimeStep #,TimeStep
from hydra.core.config_store import ConfigStore
# import omegaconf

# DEBUG = True
# laser scan parameters
# number of all laser beams
n_laser = 720
laser_angle_resolute = 0.008726646
laser_min_range = 0.27
laser_max_range = 6.0

@dataclasses.dataclass
class InformedTimeStep(ExtendedTimeStep):
    info: tp.Any

class ObservationType(enum.IntEnum):
    TOY_STATE = enum.auto()
    TIME_AWARE_TOY_STATE = enum.auto()
    AGENT_ONEHOT = enum.auto()
    GRID = enum.auto()
    AGENT_GOAL_POS = enum.auto()
    AGENT_POS = enum.auto()
    OCCU_FLOW = enum.auto()
    OCCU_MAP = enum.auto()
    RAW_SCAN = enum.auto()
    TIME_AWARE_RAW_SCAN = enum.auto()
    SEMANTIC_SCAN = enum.auto()

class NaviObsSpace:

  def __init__(self, shapes:dict, dtype, name: str):
    """Initializes a new `Array` spec.

    Args:
      shape: An iterable specifying the array shape.
      dtype: numpy dtype or string specifying the array dtype.
      name: Optional string containing a semantic name for the corresponding
        array. Defaults to `None`.

    Raises:
      TypeError: If `shape` is not an iterable of elements convertible to int,
      or if `dtype` is not convertible to a numpy dtype.
    """
    self._shape_total = 0
    self._shapes = {}
    for compo_name in shapes.keys():
        dim = 1
        shape = shapes[compo_name]
        for d in shape:
            dim*=d
        self._shape_total+=dim
        self._shapes[compo_name] = (shape,dim)
    self._dtype = np.dtype(dtype)
    self._name = name

  def get_shape(self,name):
      return self._shapes[name]

  @property
  def shape_dict(self):
    """Returns a `tuple` specifying the array shape."""
    return self._shapes
  @property
  def shape(self):
    """Returns a `tuple` specifying the array shape."""
    return (self._shape_total,)

  @property
  def dtype(self):
    """Returns a numpy dtype specifying the array dtype."""
    return self._dtype

  @property
  def name(self):
    """Returns the name of the Array."""
    return self._name

#TODO add static obstacle to phys
class CrowdPhysics:

    def __init__(self,env):
        self._env:CrowdWorld = env
        # data buffer
        # [agent_num, agent_states, static_obs]
        # item          size        content
        # agent_num:    1           current agent number
        # agent_states  5XN         x,y,vx,vy,r
        # static_obs    100         obstacle_num inf-divider (x,y)-polygon inf-divider (x,y)-polygon
    def get_state(self):
        # get robot-crowd physics
        physics = [1+len(self._env.humans)]
        physics += list(self._env.robot.get_observable_state().to_tuple())
        for human in self._env.humans:
            physics += list(human.get_observable_state().to_tuple())
        physics += (1+(1+self._env._human_num[1])*5-len(physics))*[-1]
        static_obstacle = np.ones(100,dtype=np.float32)*np.inf
        static_obstacle[0] = 1+len(self._env._layout)
        i = 1
        for obstacle in self._env._layout["vertices"]:
            for edge in obstacle[:-1]:
                static_obstacle[i]=edge[0]
                static_obstacle[i+1]=edge[1]
                i+=2
            # static_obstacle[i] = np.inf
            i+=1

        x_b,y_b = self._env._layout["boundary"].coords.xy
        x_b,y_b = list(x_b),list(y_b)
        for x,y in zip(x_b,y_b):
            static_obstacle[i]=x
            static_obstacle[i+1]=y
            i+=2
        # static_obstacle[i] = np.inf
        return np.hstack([np.array(physics,dtype=np.float32),static_obstacle])
    def render(self, height,width,camera_id):
        if self._env.render_axis is None:
            # self._env.config.env.render = False
            self._env.init_render_ax(None)
        return self._env.render(return_rgb=True)

@dataclasses.dataclass
class CrowdSimConfig:
    _target_: str = "controllable_navi.crowd_sim.crowd_sim.CrowdWorld"
    name: str = "CrowdWorld"
    scenario: str = "default"
    max_human_num: int = 6 
    map_size: float = 8
    with_static_obstacle: bool = True
    regen_map_every: int = 500

    human_policy: str = 'socialforce'
    human_radius: float = 0.3
    human_v_pref: float = 1.0
    human_visible: bool = True

    robot_policy: str = 'none'
    robot_radius: float = 0.3
    robot_v_pref: float = 1.0
    robot_visible: bool = True
    robot_rotation_constrain: float = np.pi/2

    penalty_collision: float = -1#2.
    reward_goal: float = 10#2.
    goal_factor: float = 10
    goal_range: float = 0.3
    velo_factor: float = 0.2
    discomfort_penalty_factor: float = 1.0

    local_map_size: float = 8.0
    grid_size: float = 0.25

cs = ConfigStore.instance()
cs.store(group="crowd_sim", name="CrowdWorld", node=CrowdSimConfig)

def build_crowdworld_task(cfg:CrowdSimConfig, task,phase,
                         discount=1.0,
                         observation_type="of",
                         max_episode_length=200):
    if observation_type == "of":
        obs_type = ObservationType.OCCU_FLOW
    elif observation_type =="om":
        obs_type = ObservationType.OCCU_MAP
    elif observation_type =="raw_scan":
        obs_type = ObservationType.RAW_SCAN
    elif observation_type == "toy_state":
        obs_type = ObservationType.TOY_STATE
    elif observation_type =="time_aware_raw_scan":
        obs_type = ObservationType.TIME_AWARE_RAW_SCAN
    elif observation_type == "time_aware_toy_state":
        obs_type = ObservationType.TIME_AWARE_TOY_STATE
    else:
        raise NotImplementedError

    # currently robot and human class need Config class to init
    class Config(object):
        def __init__(self):
            pass
    class EnvConfig(object):

        env = Config()
        env.human_num = [1,cfg.max_human_num]
        env.map_size = cfg.map_size

        humans = Config()
        humans.visible = cfg.human_visible
        humans.policy = cfg.human_policy
        humans.radius = cfg.human_radius
        humans.v_pref = cfg.human_v_pref

        robot = Config()
        robot.visible = cfg.robot_visible
        robot.policy = cfg.robot_policy
        robot.radius = cfg.robot_radius
        robot.v_pref = cfg.robot_v_pref
        robot.rotation_constraint = cfg.robot_rotation_constrain
 
    config = EnvConfig()

    # pick base reward
    tasks_specifications = {
                # 'PointGoalExplore':{
                #     "reward_func_ids":0,
                #     "discomfort_dist" : 0.0,
                #     "speed_limit":5.0,
                # },
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
    return CrowdWorld(phase,cfg, config,obs_type,discount,max_episode_length,**tasks_specifications[task])


class CrowdWorld(dm_env.Environment):

    def __init__(self, phase,
                 cfg:CrowdSimConfig, 
                 config,
               observation_type=ObservationType.OCCU_FLOW,
               discount=1.0,
               max_episode_length=50,
               forbiden_zone_y = 0.6,
               discomfort_dist = 0.2,
               to_wall_dist = 0.2,
               speed_limit = 1.0,
               reward_func_ids=0) -> None:
        #TODO: modify input to adapt crowdsim
        if observation_type not in ObservationType:
            raise ValueError('observation_type should be a ObservationType instace.')
        self.phase = phase
        self.config = config
        self.physics = CrowdPhysics(self)
        self._scenario = cfg.scenario
        self._map_size = self.config.env.map_size #circle
        self._with_static_obstacle = cfg.with_static_obstacle
        self._regen_map_every = cfg.regen_map_every
        self._layout: dict = {"polygon":[],"vertices":[],"boundary":None}
        map_boundary = LinearRing( ((-self._map_size/2., self._map_size/2.), 
                                    (self._map_size/2., self._map_size/2.),
                                    (self._map_size/2., -self._map_size/2.),
                                    (-self._map_size/2.,-self._map_size/2.)) )
        #map_ = {"polygon":[],"vertices":[],"boundary":map_boundary}
        self._layout["boundary"] = map_boundary

        self._human_num = self.config.env.human_num
        self._time_step = 0.25
        self.robot = Robot(config,"robot")
        self.humans : list[Human] = []
        # #human_policy = 'orca'
        # self._centralized_planner = policy_factory['centralized_' + 'orca']()
        # self._centralized_planner.time_step = self.time_step

        self._discount = discount
        self._reward_func_id = reward_func_ids
        self._reward_func_tabel = {0:self.point_goal_navi_reward,
                                   2:self.pass_human_reward,
                                   3:self.follow_wall_reward}
        self._penalty_collision = cfg.penalty_collision
        self._reward_goal = cfg.reward_goal
        self._goal_factor = cfg.goal_factor
        self._goal_range = cfg.goal_range
        self._velo_factor = cfg.velo_factor
        self._discomfort_penalty_factor = cfg.discomfort_penalty_factor
        self._discomfort_dist = discomfort_dist
        self._forbiden_zone_y = forbiden_zone_y
        self._to_wall_dis = to_wall_dist
        self._speed_limit = speed_limit
        self._step_info = InfoList()
        self._observation_type = observation_type
        self._local_map_size = cfg.local_map_size
        self._grid_size = cfg.grid_size
        self._occu_map_size = int((self._local_map_size//self._grid_size)**2) 


        self.case_capacity = {'train': np.iinfo(np.uint32).max - 2000, 'val': 1000, 'test': 1000}
        self.case_size = {'train': np.iinfo(np.uint32).max - 2000, 'val': 2000,
                          'test': 1000}
        self.case_counter = {'train': 0, 'test': 0, 'val': 0}
        self._state = None
        self._current_scan = np.zeros(n_laser, dtype=np.float32)
        self._semantic_occ_map_queue = []
        # self._goal_state = None
        self.global_time : float = 0
        self._num_episode_steps : int = 0
        self._max_episode_length: int = max_episode_length
        self._last_dg : float = 0.
        self.last_reward : tp.Optional[float] = None
        # self._task_info = {"sx":None,"sy":None,"gx":None,"gy":None} merged to _step_info

        # for visualization
        self.scan_intersection = None
        self.render_axis = None
        self.text = None
    
    def init_render_ax(self,ax):
        if ax is None:
            fig, ax = plt.subplots(figsize=(8,8)) 
        
        plt.xlim(-self._map_size/2.-1,self._map_size/2.+1)
        plt.ylim(-self._map_size/2.-1,self._map_size/2.+1)
        self.text = ax.text(0, self._map_size/2.+1.2, 'v:{}[m/s]'.format(0.), fontsize=14, color='black', ha='center', va='top')
        self.render_axis = ax
        # if self.config.env.render:
        #     plt.ion()
        #     plt.show()
    
    def point_goal_navi_reward(self,action):
        done = False
        reward = 0
        dxy = np.array(self.robot.get_goal_position())-np.array(self.robot.get_position())
        dg = np.linalg.norm(dxy)
        if dg <self._goal_range:
                reward = self._reward_goal * (1-self.global_time/self._max_episode_length)
                done = True
                self._step_info.add(ReachGoal())
        reward += self._goal_factor * (self._last_dg-dg)
            # occu_map = obs[:4096].reshape((64,64))
            # min_dist = self.task["discomfort_dist"]
            # if min_dist == 0.2: TODO

        self._last_dg = dg # type: ignore

        # if action[0]>self._speed_limit:
        #     reward -= self._velo_factor * (action[0]-self._speed_limit)
        #     self._step_info.add(ViolateSpeedLimit())
        return reward,done

    def pass_human_reward(self,action):
        reward,done = self.point_goal_navi_reward(action)

        dx = self.robot.gx - self.robot.px
        dy = self.robot.gy - self.robot.py
        dg = np.sqrt(dx*dx + dy*dy)
        if dg <= 1:
            return reward, done
        rot = self.robot.theta #np.arctan2(self.robot.vy,self.robot.vx)
        # vx = self.robot.vx*np.cos(rot) + self.robot.vy*np.sin(rot)
        # vy = self.robot.vx*np.sin(rot) + self.robot.vy*np.cos(rot)

        for agent in self.humans:
            px_other = (agent.px-self.robot.px)*np.cos(rot) + (agent.py-self.robot.py)*np.sin(rot)
            py_other = -(agent.px-self.robot.px)*np.sin(rot) + (agent.py-self.robot.py)*np.cos(rot)
            # da_other = np.sqrt(px_other*px_other + py_other*py_other)

            vx_other = agent.vx*np.cos(rot) + agent.vy*np.sin(rot)
            vy_other = -agent.vx*np.sin(rot) + agent.vy*np.cos(rot)
            # v_other =np.sqrt(vx_other*vx_other + vy_other*vy_other)

            phi_other = np.arctan2(vy_other,vx_other)
            # psi_rot = np.arctan2(vy_other-vy,vx_other-vx)
            direction_diff = phi_other  # because ego-centric ego desirable direction psi=0

            # passing 
            if px_other>1 and px_other<4 and abs(direction_diff)>3.*np.pi/4.:
                if self._forbiden_zone_y<0 and py_other<0 and py_other>-2: #right
                    reward -= 0.05
                    self._step_info.add(ViolateSidePreference('right'))
                if self._forbiden_zone_y>0 and py_other>0 and py_other<2: #left
                    reward -= 0.05
                    self._step_info.add(ViolateSidePreference('left'))

        return reward,done #TODO
    def follow_wall_reward(self,action):
        done = False
        reward = 0
        return reward,done #TODO
    
    def cal_reward(self,action):
        # TODO cal goal reward and collision panelty as default 
        # import auxilliary task from goal class and cal reward
        reward, done= self._reward_func_tabel[self._reward_func_id](action)
        # if done:
        #     return reward, done
        # safety_penalty = 0.0
        # discomfort = False
        # closest_dist = self._discomfort_dist
        # for i, human in enumerate(self.humans): 
        #     dist = norm(np.array([human.px - self.robot.px,human.py - self.robot.py]))
        #     if dist < self._discomfort_dist:
        #         discomfort = True
        #         safety_penalty = safety_penalty + (dist - self._discomfort_dist) #value<0
        #         if closest_dist>dist:
        #             closest_dist=dist
        # reward += self._discomfort_penalty_factor*safety_penalty
        # if discomfort:
        #     self._step_info.add(Discomfort(closest_dist))

        return reward, done
    
    def randomize_map(self):
        
        if not self._with_static_obstacle:
            return 
        
        # reset map
        self._layout["polygon"] = []
        self._layout["vertices"] = []

        if self._scenario == "default":
            boundary_polygon = get_default_scenario(self._map_size)
        elif self._scenario == "hallway":
            boundary_polygon = get_hallway_scenario(self._map_size)
        elif self._scenario == "cross":
            boundary_polygon = get_cross_scenario(self._map_size)
        elif self._scenario == "doorway":
            boundary_polygon = get_doorway_scenario(self._map_size)
        else:
            raise NotImplementedError
        if isinstance(boundary_polygon,MultiPolygon):
            for polygon in boundary_polygon.geoms:
                self._layout["polygon"].append(polygon)
                x, y = polygon.exterior.xy
                x,y = list(x),list(y)
                self._layout["vertices"].append([(x[i],y[i]) for i in range(len(x))])
        else:
            # Plot the boundary polygon
            self._layout["polygon"].append(boundary_polygon)
            x, y = boundary_polygon.exterior.xy
            x,y= list(x),list(y)
            self._layout["vertices"].append([(x[i],y[i]) for i in range(len(x))])
        return 
    
    def generate_human(self, human=None, non_stop=False, square=False):
        if human is None:
            human = Human(self.config, 'humans')
        # random human attribute
        human.sample_random_attributes()
        if square is False and non_stop is False:
            counter = 2e4
            while True:
                angle = np.random.random() * np.pi * 2
                # add some noise to simulate all the possible cases robot could meet with human
                px_noise = (np.random.random() - 0.5) * human.v_pref*2
                py_noise = (np.random.random() - 0.5) * human.v_pref*2
                px = (self._map_size /2. - 1)* np.cos(angle) + px_noise
                py = (self._map_size /2. - 1) * np.sin(angle) + py_noise

                collider = Point(px, py).buffer(human.radius)
                collide = False
                # print(self._layout["polygon"])
                for polygon in self._layout["polygon"]:
                    if polygon.intersects(collider):
                        collide = True
                        break
                if collide:
                    continue
                collider = Point(-px, -py).buffer(human.radius)
                collide = False
                # print(self._layout["polygon"])
                for polygon in self._layout["polygon"]:
                    if polygon.intersects(collider):
                        collide = True
                        break
                if collide:
                    continue
                for agent in [self.robot] + self.humans:
                    min_dist = human.radius + agent.radius + 0.2#self._discomfort_dist
                    if norm((px - agent.px, py - agent.py)) < min_dist or \
                            norm((-px - agent.gx, -py - agent.gy)) < min_dist:
                        collide = True
                        break
                if not collide or counter<0:
                    break
                counter-=1
            human.start_pos.append((px, py))
            human.set(px, py, -px, -py, 0, 0, 0)
        elif square is False and non_stop is True:
            raise NotImplementedError
        x,y = self._layout["boundary"].coords.xy
        human.policy.set_static_obstacle(self._layout["vertices"]+[[(x[i],y[i]) for i in range(len(x))]])

        return human
    
    def generate_robot(self):
        # if self._random_goal_velocity: #TODO add goal velo
        #     goal_theta = (np.random.random() - 0.5) * np.pi *2
        #     goal_v = np.random.random() * self.robot.v_pref
        # else:
        goal_theta = np.pi/2.
        goal_v = 0.
        while True:
            angle = np.random.random() * np.pi * 2
            # add some noise to simulate all the possible cases robot could meet with human
            px_noise = (np.random.random() - 0.5) * self.robot.v_pref
            py_noise = (np.random.random() - 0.5) * self.robot.v_pref
            px = (self._map_size /2. - 1) * np.cos(angle) + px_noise
            py = (self._map_size /2. - 1) * np.sin(angle) + py_noise
            
            angle = np.random.random() * np.pi * 2
            # add some noise to simulate all the possible cases robot could meet with human
            gx_noise = (np.random.random() - 0.5) * self.robot.v_pref
            gy_noise = (np.random.random() - 0.5) * self.robot.v_pref
            gx = (self._map_size /2. - 1) * np.cos(angle)+ gx_noise
            gy = (self._map_size /2. - 1) * np.sin(angle)+ gy_noise
            if (gx-px)**2+(gy-py)**2<6**2:
                continue
            collider = Point(px, py).buffer(self.robot.radius)
            collide = False
            # print(self._layout["polygon"])
            for polygon in self._layout["polygon"]:
                if polygon.intersects(collider):
                    collide = True
                    break
            if collide:
                continue
            collider = Point(gx, gy).buffer(self.robot.radius)
            collide = False
            # print(self._layout["polygon"])
            for polygon in self._layout["polygon"]:
                if polygon.intersects(collider):
                    collide = True
                    break
            if not collide:
                break
        # if (gx-px)**2+(gy-py)**2<6**2:
        #     raise ValueError
        self.robot.set(px,py,gx,gy, 0, 0, np.pi / 2, goal_theta=goal_theta, goal_v=goal_v)
        self.robot.kinematics = "unicycle"
        return 
    
    def observation_spec(self):
        # if self._observation_type is ObservationType.OCCU_MAP:
        #     return specs.Array(
        #         shape=(self._occu_map_size*2+5,), #map + [dgx,dgy,dgtheta,dgv,vx,vy,w]_robot_frame
        #         dtype=np.float32,
        #         name='observation_occupancy_map'
        #     )
        # elif self._observation_type is ObservationType.OCCU_FLOW:
        #     return specs.Array(
        #         shape=(self._occu_map_size*4+8,), dtype=np.float32, name='observation_occupancy_flow')
        if self._observation_type is ObservationType.RAW_SCAN:
            return NaviObsSpace({"scan":(720,),"robot_state":(5,)}, dtype=np.float32, name='relative pose to goal')
        elif self._observation_type is ObservationType.TOY_STATE:
            return NaviObsSpace({"toy_joint_state":(10,)}, dtype=np.float32, name='relative pose to goal') 
        elif self._observation_type is ObservationType.TIME_AWARE_RAW_SCAN:
            return NaviObsSpace({"scan":(720,),"robot_state":(7,)}, dtype=np.float32, name='relative pose to goal')
        elif self._observation_type is ObservationType.TIME_AWARE_TOY_STATE:
            return NaviObsSpace({"toy_joint_state":(12,)}, dtype=np.float32, name='relative pose to goal') 
        else:
            raise NotImplementedError

    def action_spec(self):
        return specs.BoundedArray(shape=(2,), dtype=np.float32, name='action', minimum=-1.0, maximum=1.0)
    #specs.Array(shape=(2,), dtype=np.float32, name='action')
    
    # TODO add custom case no. input
    def reset(self):
        self.humans = []
        self._num_episode_steps = 0
        train_seed_begin = [0, 10, 100, 1000, 10000]
        val_seed_begin = [0, 10, 100, 1000, 10000]
        test_seed_begin = [0, 10, 100, 1000, 10000]
        base_seed = {'train': self.case_capacity['val'] + self.case_capacity['test'] + train_seed_begin[1],
                     'val': 0 + val_seed_begin[1], 'test': self.case_capacity['val']+test_seed_begin[2]+1000}

        
        self.random_seed = base_seed[self.phase] + self.case_counter[self.phase]
        np.random.seed(self.random_seed)
        if self.case_counter[self.phase]%self._regen_map_every==0:
            self.randomize_map()

        if self._human_num[1]==1: #TOY ENV
            self.robot.set(0,-2, 0, 2, 0,0, np.pi / 2, goal_theta=0, goal_v=0)
            self.robot.kinematics = "unicycle"
            human = Human(self.config, 'humans')
            human.start_pos.append((0, 2))
            human.set(0, 2, 0,-2, 0, 0, 0)
            self.humans.append(human)
        elif self.phase == 'test' and self.case_counter[self.phase] == 0:
            self.robot.set(0,-3, 0, 3, 0,0, np.pi / 2, goal_theta=0, goal_v=0)
            self.robot.kinematics = "unicycle"
            human = Human(self.config, 'humans')
            human.start_pos.append((0, 3))
            human.set(0, 3, 0,-3, 0, 0, 0)
            self.humans.append(human)
        elif self.phase == 'test' and self.case_counter[self.phase] == 1:
            self.robot.set(-3,-3, 3, 3, 0,0, np.pi / 2, goal_theta=0, goal_v=0)
            self.robot.kinematics = "unicycle"
            human = Human(self.config, 'humans')
            human.start_pos.append((3, 3))
            human.set(3, 3, -3,-3, 0, 0, 0)
            self.humans.append(human)
        elif self.phase == 'test' and self.case_counter[self.phase] == 2:
            self.robot.set(-3,0, 3, 0, 0,0, np.pi / 2, goal_theta=0, goal_v=0)
            self.robot.kinematics = "unicycle"
            human = Human(self.config, 'humans')
            human.start_pos.append((3, 0))
            human.set(3, 0, -3,0, 0, 0, 0)
            self.humans.append(human)
        else:
            # set robot init state, goal pos and goal speed
            self.generate_robot()
            
            human_num = np.random.randint(self._human_num[0],self._human_num[1]+1)
            for i in range(human_num):
                self.humans.append(self.generate_human())
        self._num_episode_steps = 0
        self.global_time = 0
        self.case_counter[self.phase] = (self.case_counter[self.phase] + 1) % self.case_size[self.phase]
        self.robot.time_step = self._time_step
        for agent in self.humans:
            agent.time_step = self._time_step
            agent.policy.time_step = self._time_step

        self._semantic_occ_map_queue = [np.zeros((int(self._local_map_size/self._grid_size),
                                      int(self._local_map_size/self._grid_size),
                                      2),dtype=np.float32),
                                      np.zeros((int(self._local_map_size/self._grid_size),
                                      int(self._local_map_size/self._grid_size),
                                      2),dtype=np.float32)]
        self._state = self.get_obs()
        dxy = np.array(self.robot.get_goal_position())-np.array(self.robot.get_position())
        self._last_dg = norm(dxy)
        self._step_info.reset()
        self._step_info.add(Nothing())
        self._step_info.get_task_info(self.robot.px,self.robot.py,self.robot.gx,self.robot.gy)

        return InformedTimeStep(
            step_type=dm_env.StepType.FIRST,
            action=np.array([0,0]),
            reward=0.0,
            discount=1.0,
            observation=self._state,
            info = copy.deepcopy(self._step_info),
            # physics=np.array(physics,dtype=np.float32),
            )
    
    def get_static_scan(self):
        num_line = sum([len(obstacle)-1 for obstacle in self._layout["vertices"]])+4 #for boundary
        num_circle = 0
        scan = np.zeros(n_laser, dtype=np.float32)
        scan_end = np.zeros((n_laser, 2), dtype=np.float32)
        robot_pose = np.array([self.robot.px, self.robot.py, self.robot.theta])
        # print(self._layout)
        InitializeEnv(num_line, num_circle, n_laser, laser_angle_resolute)
        x_b,y_b = self._layout["boundary"].coords.xy
        for i in range(4):
            line = [(x_b[i],y_b[i]),(x_b[i+1],y_b[i+1])]
            set_lines(4 * i    , line[0][0])
            set_lines(4 * i + 1, line[0][1])
            set_lines(4 * i + 2, line[1][0])
            set_lines(4 * i + 3, line[1][1])
        line_pointer = 4*4
        for obstacle in self._layout["vertices"][::-1]:
            for i in range(len(obstacle)-1):
                line = [obstacle[i],obstacle[i+1]]
                set_lines(line_pointer+4 * i    , line[0][0])
                set_lines(line_pointer+4 * i + 1, line[0][1])
                set_lines(line_pointer+4 * i + 2, line[1][0])
                set_lines(line_pointer+4 * i + 3, line[1][1])
            line_pointer+=(len(obstacle)-1)*4
    
        set_robot_pose(robot_pose[0], robot_pose[1], robot_pose[2])
        cal_laser()
        
        for i in range(n_laser):
            scan[i] = get_scan(i)
            scan_end[i, :] = np.array([get_scan_line(4 * i + 2), get_scan_line(4 * i + 3)])
            ### used for visualization
            # self.scan_intersection.append([(get_scan_line(4 * i + 0), get_scan_line(4 * i + 1)), \
            #                                (get_scan_line(4 * i + 2), get_scan_line(4 * i + 3))])
            # ### used for visualization
        ReleaseEnv()

        return scan,scan_end
    
    def get_dynamic_scan(self):
        num_line = 4
        num_circle = len(self.humans)
        scan = np.zeros(n_laser, dtype=np.float32)
        scan_end = np.zeros((n_laser, 2), dtype=np.float32)
        robot_pose = np.array([self.robot.px, self.robot.py, self.robot.theta])
        # print(self._layout)
        InitializeEnv(num_line, num_circle, n_laser, laser_angle_resolute)
        x_b,y_b = self._layout["boundary"].coords.xy
        for i in range(4):
            line = [(x_b[i],y_b[i]),(x_b[i+1],y_b[i+1])]
            set_lines(4 * i    , line[0][0])
            set_lines(4 * i + 1, line[0][1])
            set_lines(4 * i + 2, line[1][0])
            set_lines(4 * i + 3, line[1][1])
        
        for i in range (num_circle):
            set_circles(3 * i    , self.humans[i].px)
            set_circles(3 * i + 1, self.humans[i].py)
            set_circles(3 * i + 2, self.humans[i].radius)
        set_robot_pose(robot_pose[0], robot_pose[1], robot_pose[2])
        cal_laser()
        
        for i in range(n_laser):
            scan[i] = get_scan(i)
            scan_end[i, :] = np.array([get_scan_line(4 * i + 2), get_scan_line(4 * i + 3)])
            ### used for visualization
            # self.scan_intersection.append([(get_scan_line(4 * i + 0), get_scan_line(4 * i + 1)), \
            #                                (get_scan_line(4 * i + 2), get_scan_line(4 * i + 3))])
            # ### used for visualization
        ReleaseEnv()

        return scan, scan_end
    
    def get_obs(self):
        robot_v = np.sqrt(self.robot.vx**2+self.robot.vy**2)
        robot_w = self.robot.w
        dxy = np.array(self.robot.get_goal_position())-np.array(self.robot.get_position())
        dg = norm(dxy)
        goal_direction = np.arctan2(dxy[1],dxy[0])
        hf = self.robot.theta-goal_direction
        # transform dxy to base frame
        vx = (self.robot.vx * np.cos(goal_direction) + self.robot.vy * np.sin(goal_direction))
        vy = (self.robot.vy * np.cos(goal_direction) - self.robot.vx * np.sin(goal_direction)) 
        
        self._current_scan = np.zeros(n_laser, dtype=np.float32)
        # semantic_occu_map = np.zeros((int(self._local_map_size/self._grid_size),
        #                               int(self._local_map_size/self._grid_size),
        #                               2),dtype=np.float32)

        self.scan_intersection = []
        static_ray_length,static_scan = self.get_static_scan()
        dynamic_ray_length,dynamic_scan = self.get_dynamic_scan()

        # print(len(self.scan_intersection))
        # boundary_indexs = np.where(static_scan==dynamic_scan)[0]
        human_ray_indexs = np.where(dynamic_ray_length<static_ray_length)[0].tolist()
        obstacle_ray_indexs = np.where(static_ray_length<=dynamic_ray_length)[0].tolist()
        scan_human_layer_local = (dynamic_scan-np.array([self.robot.px,self.robot.py])).dot(
            np.array([[np.cos(self.robot.theta),-np.sin(self.robot.theta)],
                      [np.sin(self.robot.theta),np.cos(self.robot.theta)]])
        )
        scan_obstacle_layer_local = (static_scan-np.array([self.robot.px,self.robot.py])).dot(
            np.array([[np.cos(self.robot.theta),-np.sin(self.robot.theta)],
                      [np.sin(self.robot.theta),np.cos(self.robot.theta)]])
        )
        for i in range(n_laser):
            if i not in obstacle_ray_indexs:
                continue
            self._current_scan[i] = static_ray_length[i]
            self.scan_intersection.append([(self.robot.px, self.robot.py), \
                                        (static_scan[i,0],static_scan[i,1])])
            # x = scan_obstacle_layer_local[i,0]
            # y = scan_obstacle_layer_local[i,1]
            # try:
            #     grid_x = int((x+self._local_map_size/2.)/self._grid_size)
            #     grid_y = int((y+self._local_map_size/2.)/self._grid_size)
            # except:
            #     print(np.array([self.robot.px,self.robot.py]),x, self._local_map_size, self._grid_size)
            # if grid_x>=0 and grid_x<semantic_occu_map.shape[0] \
            # and grid_y>=0 and grid_y<semantic_occu_map.shape[1]:
            #     semantic_occu_map[grid_x,grid_y,0] = 1.0
        for i in range(n_laser):
            if i not in human_ray_indexs:
                continue
            self._current_scan[i] = dynamic_ray_length[i]
            self.scan_intersection.append([(self.robot.px, self.robot.py), \
                                        (dynamic_scan[i,0],dynamic_scan[i,1])])
            # x = scan_human_layer_local[i,0]
            # y = scan_human_layer_local[i,1]
            # try:
            #     grid_x = int((x+self._local_map_size/2.)/self._grid_size)
            #     grid_y = int((y+self._local_map_size/2.)/self._grid_size)
            # except:
            #     print(x, self._local_map_size, self._grid_size)
            # if grid_x>=0 and grid_x<semantic_occu_map.shape[0] \
            # and grid_y>=0 and grid_y<semantic_occu_map.shape[1]:
            #     semantic_occu_map[grid_x,grid_y,1] = 1.0
        

        if self._observation_type == ObservationType.RAW_SCAN:
            return np.hstack([np.clip(self._current_scan,0.,self._local_map_size/2.),
                              np.array([dg,hf,vx,vy,self.robot.radius],dtype=np.float32)]) 
        elif self._observation_type == ObservationType.TIME_AWARE_RAW_SCAN:
            return np.hstack([np.clip(self._current_scan,0.,self._local_map_size/2.),
                              np.array([self._num_episode_steps/self._max_episode_length,
                                        np.log(self._max_episode_length),
                                        dg, hf,vx,vy,self.robot.radius],dtype=np.float32)]) 
        # elif self._observation_type == ObservationType.OCCU_MAP:
        #     # if heading_diff_cosine>np.cos(np.pi/4): 
        #     #     reward += 0.0005 TODO drl-vo reward term

        #     # dgtheta = self.robot.goal_theta-self.robot.theta
        #     # dgv = self.robot.goal_v-np.linalg.norm(np.array([self.robot.vx,self.robot.vy]))

        #     return np.hstack([semantic_occu_map[...,1].flatten(),
        #                       semantic_occu_map[...,0].flatten(),
        #                       np.array([dg,hf,vx,vy,self.robot.radius],dtype=np.float32)]) 
        # elif self._observation_type == ObservationType.OCCU_FLOW:
        #     self._semantic_occ_map_queue.pop(0)
        #     self._semantic_occ_map_queue.append(semantic_occu_map)
        #     return np.hstack([self._semantic_occ_map_queue[1][...,1].flatten(),
        #                       self._semantic_occ_map_queue[1][...,0].flatten(),
        #                       self._semantic_occ_map_queue[0][...,1].flatten(),
        #                       self._semantic_occ_map_queue[0][...,0].flatten(),
        #                       np.array([self.robot.px,self.robot.py,self.robot.theta, self.robot.gx,self.robot.gy,robot_v,robot_w,self.robot.radius],dtype=np.float32)])
        elif self._observation_type == ObservationType.TOY_STATE:
            assert len(self.humans)==1
            #[dg, vpref, vx, vy, r, θ, ˜ vx, ˜ vy, ˜ px, ˜ py, ˜ r, r +˜ r, cos(θ), sin(θ), da],
            px = self.humans[0].px - self.robot.px
            py = self.humans[0].py - self.robot.py
            h_px = px * np.cos(goal_direction) + py * np.sin(goal_direction)
            h_py = py * np.cos(goal_direction) - px * np.sin(goal_direction)
            h_vx = self.humans[0].vx * np.cos(goal_direction) + self.humans[0].vy * np.sin(goal_direction)
            h_vy = self.humans[0].vy * np.cos(goal_direction) - self.humans[0].vx * np.sin(goal_direction)
            return np.hstack([np.array([h_px,h_py,h_vx,h_vy,self.humans[0].radius+self.robot.radius],dtype=np.float32),
                              np.array([dg,hf,vx,vy,self.robot.radius],dtype=np.float32)]) 
        elif self._observation_type == ObservationType.TIME_AWARE_TOY_STATE:
            assert len(self.humans)==1
            #[dg, vpref, vx, vy, r, θ, ˜ vx, ˜ vy, ˜ px, ˜ py, ˜ r, r +˜ r, cos(θ), sin(θ), da],
            px = self.humans[0].px - self.robot.px
            py = self.humans[0].py - self.robot.py
            h_px = px * np.cos(goal_direction) + py * np.sin(goal_direction)
            h_py = py * np.cos(goal_direction) - px * np.sin(goal_direction)
            h_vx = self.humans[0].vx * np.cos(goal_direction) + self.humans[0].vy * np.sin(goal_direction)
            h_vy = self.humans[0].vy * np.cos(goal_direction) - self.humans[0].vx * np.sin(goal_direction)
            return np.hstack([np.array([h_px,h_py,h_vx,h_vy,self.humans[0].radius+self.robot.radius],dtype=np.float32),
                              np.array([self._num_episode_steps/self._max_episode_length,
                                        np.log(self._max_episode_length),
                                        dg,hf,vx,vy,self.robot.radius],dtype=np.float32)]) 
        else:
            # rethink of dg,vx,vy,dgtheta,dgv,self.robot.radius
            # there is no information about theta of the robot in goal coordinate
            # dgtheta is pose diff 
            # need state representation such that s,a can specify r and s'
            raise NotImplementedError
        
    def compute_observation_for(self, agent):
        if agent == self.robot:
            raise NotImplementedError
        else:
            ob = [other_human.get_observable_state() for other_human in self.humans if other_human != agent]
            if self.robot.visible:
                ob += [self.robot.get_observable_state()]
        return ob

    def step(self, action):
        # simulation
        human_actions = []
        for human in self.humans:
            ob = self.compute_observation_for(human)
            human_actions.append(human.act(ob))
        robot_action = ActionVW(((action[0]+1.)/2.)*self.robot.v_pref,action[1]*self.robot.rotation_constraint)
        # update all agents
        self.robot.step(robot_action)
        for human, human_action in zip(self.humans, human_actions):
            human.step(human_action)
        self._state = self.get_obs()
        self.global_time += self._time_step
        self._num_episode_steps += 1

        # collision detection between the robot and humans
        collision = False
        dmin =np.min(self._current_scan)
        if dmin <= self.robot.radius:
            collision = True
            
        self._step_info.reset()
        # cal reward
        step_type = dm_env.StepType.MID
        discount = 1.0
        if collision:
            reward = self._penalty_collision
            discount = 0
            step_type = dm_env.StepType.LAST
            self._step_info.add(Collision())
        elif (self._max_episode_length is not None and
            self._num_episode_steps >= self._max_episode_length):
            reward = -self._reward_goal
            step_type = dm_env.StepType.LAST
            discount = 0
            self._step_info.add(Timeout())
        else:
            reward,done= self.cal_reward(robot_action)
            if done:
                step_type = dm_env.StepType.LAST
                discount = 0
            
        self.last_reward = reward
        if self._step_info.empty():
            self._step_info.add(Nothing())

        return InformedTimeStep(
            step_type=step_type,
            action=action,
            reward=np.float32(reward),
            discount=discount,
            observation=self._state,
            info=copy.deepcopy(self._step_info)
            # physics=np.array(physics,dtype=np.float32),
            )

    def render(self, return_rgb=True):
        # x,y = self._layout["boundary"].exterior.xy
        # plt.plot(x, y)
        plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'
        ax=self.render_axis
        robot_color = 'yellow'
        goal_color = 'red'
        artists = []
        goal=mlines.Line2D([self.robot.gx], [self.robot.gy], color=goal_color, marker='*', linestyle='None', markersize=15, label='Goal')
        ax.add_artist(goal)
        artists.append(goal)

        # add robot
        robotX,robotY=self.robot.get_position()

        robot=plt.Circle((robotX,robotY), self.robot.radius, fill=True, color=robot_color)
        ax.add_artist(robot)
        artists.append(robot)

        plt.legend([robot, goal], ['Robot', 'Goal'], bbox_to_anchor=(0.85, 0.85), loc='upper left', fontsize=16)

        human_circles = [plt.Circle(human.get_position(), human.radius, fill=False) for human in self.humans]


        for i in range(len(self.humans)):
            ax.add_artist(human_circles[i])
            artists.append(human_circles[i])
        
        for obstacle in self._layout["vertices"]:
            polygon = patches.Polygon(obstacle[:-1], closed=True, edgecolor='b', facecolor='none')
            ax.add_patch(polygon)
            artists.append(polygon)

        x_b,y_b = self._layout["boundary"].coords.xy
        x_b,y_b = list(x_b),list(y_b)
        boundary = [(x_b[i],y_b[i]) for i in range(len(x_b))]
        polygon = patches.Polygon(boundary[:-1], closed=True, edgecolor='b', facecolor='none')
        ax.add_patch(polygon)
        artists.append(polygon)
        # for agent in self.humans + [self.robot]:
        #     x,y = agent.collider.exterior.xy
        #     ax.plot(x, y)
        if self.scan_intersection is not None:
                ii = 0
                lines = []
                while ii < n_laser:
                    lines.append(self.scan_intersection[ii])
                    ii = ii + 36
                lc = mc.LineCollection(lines,linewidths=1,linestyles='--',alpha=0.2)
                ax.add_artist(lc)
                artists.append(lc)
        if return_rgb:
            fig = plt.gcf()
            self.text.set_text('v:{}[m/s]'.format(norm([self.robot.vx,self.robot.vy])))
            # plt.axis('tight')
            # plt.subplots_adjust(0, 0, 1, 1, 0, 0)
            fig.canvas.draw()
            data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            w, h = fig.canvas.get_width_height()
            data = data.reshape((h, w, 3))
            for item in artists:
                item.remove()
            return data
        else:
            plt.pause(0.1)
            
            for item in artists:
                item.remove()
           

if __name__ == "__main__":
    crowd_sim = build_crowdworld_task("point_goal_navi","train")
    step_ex = crowd_sim.reset()
    print(step_ex)

