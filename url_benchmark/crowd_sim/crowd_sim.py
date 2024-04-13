import logging
import random
import math
import typing as tp
import shapely
from shapely.geometry import Polygon, LineString, MultiPolygon, LinearRing,Point

import gym
from gym import spaces
import matplotlib.pyplot as plt
from matplotlib import collections as mc
import matplotlib.lines as mlines
from matplotlib import patches
import numpy as np
from numpy.linalg import norm
from url_benchmark.crowd_sim.C_library.motion_plan_lib import *

from url_benchmark.crowd_sim.policy.policy_factory import policy_factory
from url_benchmark.crowd_sim.utils.state import tensor_to_joint_state, JointState,ExFullState
from url_benchmark.crowd_sim.utils.action import ActionRot,ActionVW
from url_benchmark.crowd_sim.utils.human import Human
from url_benchmark.crowd_sim.utils.robot import Robot
from url_benchmark.crowd_sim.utils.info import *
from url_benchmark.crowd_sim.utils.utils import point_to_segment_dist
from url_benchmark.crowd_sim.utils.obstacle import *

import dm_env
from dm_env import specs
import enum
from url_benchmark.dmc import ExtendedTimeStep,TimeStep

DEBUG = True
# laser scan parameters
# number of all laser beams
n_laser = 1800
laser_angle_resolute = 0.003490659
laser_min_range = 0.27
laser_max_range = 6.0

class ObservationType(enum.IntEnum):
    STATE_INDEX = enum.auto()
    AGENT_ONEHOT = enum.auto()
    GRID = enum.auto()
    AGENT_GOAL_POS = enum.auto()
    AGENT_POS = enum.auto()
    OCCU_FLOW = enum.auto()
    OCCU_MAP = enum.auto()
    RAW_SCAN = enum.auto()
    SEMANTIC_SCAN = enum.auto()

class CrowdPhysics:

    def __init__(self,env):
        self._env = env
    def get_state(self):
        # get robot-crowd physics
        physics = [1+len(self._env.humans)]
        physics += list(self._env.robot.get_observable_state().to_tuple())
        for human in self._env.humans:
            physics += list(human.get_observable_state().to_tuple())
        physics += (1+(1+self._env._human_num[1])*5-len(physics))*[-1]
        return np.array(physics,dtype=np.float32)
    def render(self, height,width,camera_id):
        if self._env.render_axis is None:
            self._env.config.env.render = False
            self._env.init_render_ax(None)
        return self._env.render(return_rgb=True)

def build_crowdworld_task(task,phase,
                         discount=0.9,
                         observation_type=ObservationType.OCCU_MAP,
                         max_episode_length=200):
    class Config(object):
        def __init__(self):
            pass
    class EnvConfig(object):

        env = Config()
        env.render = True

        humans = Config()
        humans.visible = True
        humans.policy = 'socialforce'
        humans.radius = 0.3
        humans.v_pref = 1.0
        humans.sensor = 'coordinates'

        robot = Config()
        robot.visible = True
        robot.policy = 'none'
        robot.radius = 0.3
        robot.v_pref = 1.0
        robot.sensor = 'coordinates'
        robot.rotation_constraint = np.pi/6
    
    config = EnvConfig()

    tasks_specifications = {
                'PointGoalNavi':{
                    "random_goal_velocity":False,
                    "reward_func_ids":0,
                    "discomfort_dist" : 0.2,
                },
                'Tracking':{
                    "random_goal_velocity":True,
                    "reward_func_ids": 1
                },
                'PassLeftSide':{
                    "random_goal_velocity":False,
                    "reward_func_ids": 2,
                    "forbiden_zone_y":0.6,
                },
                'PassRightSide':{
                    "random_goal_velocity":False,
                    "reward_func_ids": 2,
                    "forbiden_zone_y" : -0.6,
                },
                'FollowWall':{
                    "random_goal_velocity":False,
                    "reward_func_ids": 3
                },
                'AwayFromHuman':{
                    "random_goal_velocity":False,
                    "reward_func_ids": 0,
                    "discomfort_dist" : 0.8,
                },
                'LowSpeed':{
                    "random_goal_velocity":False,
                    "reward_func_ids": 0,
                    "speed_limit":0.5,
                }
                            }
    return CrowdWorld(phase,config,observation_type,discount,max_episode_length,**tasks_specifications[task])


class CrowdWorld(dm_env.Environment):

    def __init__(self, phase,config,
               observation_type=ObservationType.OCCU_FLOW,
               discount=0.9,
               max_episode_length=None,
               forbiden_zone_y = 0.6,
               discomfort_dist = 0.2,
               to_wall_dist = 0.2,
               speed_limit = 1.0,
               reward_func_ids=[0],
               random_goal_velocity=False) -> None:
        #TODO: modify input to adapt crowdsim
        if observation_type not in ObservationType:
            raise ValueError('observation_type should be a ObservationType instace.')
        self.phase = phase
        self.config = config
        self.physics = CrowdPhysics(self)

        self._map_size = 10 #circle
        self._layout = None
        self._human_num = [1,6]
        self._time_step = 0.25
        self.robot = Robot(config,"robot")
        self.humans : list[Human] = []
        # #human_policy = 'orca'
        # self._centralized_planner = policy_factory['centralized_' + 'orca']()
        # self._centralized_planner.time_step = self.time_step

        self._discount = discount
        self._reward_func_id = reward_func_ids
        self._reward_func_tabel = {0:self.point_goal_navi_reward,
                                   1:self.human_tracking_reward,
                                   2:self.pass_human_reward,
                                   3:self.follow_wall_reward}
        self._random_goal_velocity = random_goal_velocity
        self._penalty_collision = -0.25
        self._reward_goal = 0.25
        self._goal_factor = 0.2
        self._goal_range = 0.3
        self._reward_velo = 0.25
        self._velo_factor = 0.2
        self._velo_range = 0.1
        self._discomfort_penalty_factor = 1.0
        self._discomfort_dist = discomfort_dist
        self._forbiden_zone_y = forbiden_zone_y
        self._to_wall_dis = to_wall_dist
        self._speed_limit = speed_limit

        self._observation_type = observation_type
        self._local_map_size = 8
        self._grid_size = 0.25
        self._occu_map_size = int((self._local_map_size//self._grid_size)**2) 


        self.case_capacity = {'train': np.iinfo(np.uint32).max - 2000, 'val': 1000, 'test': 1000}
        self.case_size = {'train': np.iinfo(np.uint32).max - 2000, 'val': 2000,
                          'test': 1000}
        self.case_counter = {'train': 0, 'test': 0, 'val': 0}
        self._state = None
        self._current_scan = None
        self._goal_state = None
        self.global_time = None
        self._num_episode_steps = 0
        self._max_episode_length = max_episode_length
        self._last_dg = None
        self.last_reward = None

        # for visualization
        self.scan_intersection = None
        self.render_axis = None
        
    
    def init_render_ax(self,ax):
        if ax is None:
            fig, ax = plt.subplots(figsize=(5,5)) 
        ax.set_xlim(-self._map_size/2.-1,self._map_size/2.+1)
        ax.set_ylim(-self._map_size/2.-1,self._map_size/2.+1)
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
                reward = self._reward_goal
                done = True
        else:
            reward += self._goal_factor * (self._last_dg-dg)
            # occu_map = obs[:4096].reshape((64,64))
            # min_dist = self.task["discomfort_dist"]
            # if min_dist == 0.2: TODO
        if action[0]>self._speed_limit:
            reward -= self._velo_factor * (action[0]-self._speed_limit)
                
        return reward,done 
    def human_tracking_reward(self,action):
        #np.array([dg,vx,vy,dgtheta,dgv,self.robot.radius],dtype=np.float32)
        done = False
        reward = 0

        dxy = np.array(self.robot.get_goal_position())-np.array(self.robot.get_position())
        dg = np.linalg.norm(dxy)
        dgtheta = self.robot.goal_theta-self.robot.theta
        dgv = self.robot.goal_v-np.linalg.norm(np.array([self.robot.vx,self.robot.vy]))
        reward = self._goal_factor*(np.exp(-2*dg)+np.exp(-1*dgtheta))+self._velo_factor*np.exp(-0.1*dgv)

        return reward,done #TODO
    def pass_human_reward(self,action):
        reward,done = self.point_goal_navi_reward(action)

        dx = self.robot.gx - self.robot.px
        dy = self.robot.gy - self.robot.py
        dg = np.sqrt(dx*dx + dy*dy)
        if dg <= 3:
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
                if self._forbiden_zone_y>0 and py_other>0 and py_other<2: #left
                    reward -= 0.05

        return reward,done #TODO
    def follow_wall_reward(self,action):
        done = False
        reward = 0
        
        return reward,done #TODO
    
    def cal_reward(self,action):
        reward, done =self._reward_func_tabel[self._reward_func_id](action)
        safety_penalty = 0.0
        for i, human in enumerate(self.humans): 
            closest_dist = norm(np.array([human.px - self.robot.px,human.py - self.robot.py]))
            if closest_dist < self._discomfort_dist:
                safety_penalty = safety_penalty + (closest_dist - self._discomfort_dist) #value<0
        reward += self._discomfort_penalty_factor*safety_penalty
        return reward, done
    
    def randomize_map(self):
        map_boundary = LinearRing( ((-self._map_size/2.-1, self._map_size/2.+1), 
                                    (self._map_size/2.+1, self._map_size/2.+1),
                                    (self._map_size/2.+1, -self._map_size/2.-1),
                                    (-self._map_size/2.-1,-self._map_size/2.-1)) )
        map_ = {"polygon":[],"vertices":[],"boundary":map_boundary}
        #gen triangle
        x = (np.random.random() - 0.5)*self._map_size
        y = (np.random.random() - 0.5)*self._map_size
        angle1 = (np.random.random() - 0.5)*np.pi/3. + 2*np.pi/3.
        angle2 = (np.random.random() - 0.5)*np.pi/3. + 2*np.pi/3.
        radius = np.random.random() * self._map_size/6.
        triangle = genTriangle((x,y), [angle1,angle2], radius)
        #gen rectangle
        x = (np.random.random() - 0.5)*self._map_size
        y = (np.random.random() - 0.5)*self._map_size
        w = np.random.random() * self._map_size/6.
        h = np.random.random() * self._map_size/6.
        yaw = (np.random.random() - 0.5)*2*np.pi
        rectangle = genRectangle(x,y,w,h,yaw)
        #gen hex
        x = (np.random.random() - 0.5)*self._map_size
        y = (np.random.random() - 0.5)*self._map_size
        radius = np.random.random() * self._map_size/6.
        yaw = (np.random.random() - 0.5)*2*np.pi
        hex = genHexagon(x,y,radius,yaw)

        triangle = Polygon(triangle)
        rectangle = Polygon(rectangle)
        hex = Polygon(hex)

        boundary_polygon = triangle.union(rectangle).union(hex)
        if isinstance(boundary_polygon,MultiPolygon):
            for polygon in boundary_polygon.geoms:
                map_["polygon"].append(polygon)
                x, y = polygon.exterior.xy
                x,y = list(x),list(y)
                map_["vertices"].append([(x[i],y[i]) for i in range(len(x))])
        else:
            # Plot the boundary polygon
            map_["polygon"].append(boundary_polygon)
            x, y = boundary_polygon.exterior.xy
            x,y= list(x),list(y)
            map_["vertices"].append([(x[i],y[i]) for i in range(len(x))])
        return map_
    
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
                px = self._map_size * np.cos(angle)/2. + px_noise
                py = self._map_size * np.sin(angle)/2. + py_noise

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
                            norm((px - agent.gx, py - agent.gy)) < min_dist:
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
        if self._random_goal_velocity:
            goal_theta = (np.random.random() - 0.5) * np.pi *2
            goal_v = np.random.random() * self.robot.v_pref
        else:
            goal_theta = np.pi/2.
            goal_v = 0.
        while True:
            angle = np.random.random() * np.pi * 2
            # add some noise to simulate all the possible cases robot could meet with human
            px_noise = (np.random.random() - 0.5) * self.robot.v_pref
            py_noise = (np.random.random() - 0.5) * self.robot.v_pref
            px = self._map_size * np.cos(angle)/2. + px_noise
            py = self._map_size * np.sin(angle)/2. + py_noise
            
            angle = np.random.random() * np.pi * 2
            # add some noise to simulate all the possible cases robot could meet with human
            gx_noise = (np.random.random() - 0.5) * self.robot.v_pref
            gy_noise = (np.random.random() - 0.5) * self.robot.v_pref
            gx = self._map_size * np.cos(angle)/2.+ gx_noise
            gy = self._map_size * np.sin(angle)/2.+ gy_noise
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
        self.robot.set(px,py,gx,gy, 0, 0, np.pi / 2, goal_theta=goal_theta, goal_v=goal_v)
        self.robot.kinematics = "unicycle"
        return 
    
    def observation_spec(self):
        if self._observation_type is ObservationType.OCCU_MAP:
            return specs.Array(
                shape=(self._occu_map_size+6,), #map + [dgx,dgy,dgtheta,dgv,vx,vy,w]_robot_frame
                dtype=np.float32,
                name='observation_occupancy_map'
            )
        elif self._observation_type is ObservationType.OCCU_FLOW:
            return specs.Array(
                shape=(self._occu_map_size*3+6,), dtype=np.float32, name='observation_occupancy_flow')

    def action_spec(self):
        return specs.BoundedArray(shape=(2,), dtype=np.float32, name='action', minimum=-1.0, maximum=1.0)
    #specs.Array(shape=(2,), dtype=np.float32, name='action')
    
    def reset(self):
        self._num_episode_steps = 0
        train_seed_begin = [0, 10, 100, 1000, 10000]
        val_seed_begin = [0, 10, 100, 1000, 10000]
        test_seed_begin = [0, 10, 100, 1000, 10000]
        base_seed = {'train': self.case_capacity['val'] + self.case_capacity['test'] + train_seed_begin[1],
                     'val': 0 + val_seed_begin[1], 'test': self.case_capacity['val']+test_seed_begin[2]+1000}

        
        self.random_seed = base_seed[self.phase] + self.case_counter[self.phase]
        np.random.seed(self.random_seed)
        self._layout= self.randomize_map()

        # set robot init state, goal pos and goal speed
        self.generate_robot()
        
        human_num = np.random.randint(self._human_num[0],self._human_num[1]+1)
        self.humans = []
        for i in range(human_num):
            self.humans.append(self.generate_human())
        
        self._num_episode_steps = 0
        self.global_time = 0
        self.case_counter[self.phase] = (self.case_counter[self.phase] + 1) % self.case_size[self.phase]
        self.robot.time_step = self._time_step
        for agent in self.humans:
            agent.time_step = self._time_step
            agent.policy.time_step = self._time_step

        self._state = self.get_obs()
        self._last_dg = self._state[-6]
    
        return ExtendedTimeStep(
            step_type=dm_env.StepType.FIRST,
            action=np.array([0,0]),
            reward=0.0,
            discount=self._discount,
            observation=self._state,
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
        self._current_scan = np.zeros(n_laser, dtype=np.float32)
        semantic_occu_map = np.zeros((int(self._local_map_size/self._grid_size),
                                      int(self._local_map_size/self._grid_size),
                                      2),dtype=np.float32)

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
        if self._observation_type == ObservationType.OCCU_MAP:
            for i in range(n_laser):
                if i not in obstacle_ray_indexs:
                    continue
                self._current_scan[i] = static_ray_length[i]
                self.scan_intersection.append([(self.robot.px, self.robot.py), \
                                            (static_scan[i,0],static_scan[i,1])])
                x = scan_obstacle_layer_local[i,0]
                y = scan_obstacle_layer_local[i,1]
                grid_x = int((x+self._local_map_size/2.)/self._grid_size)
                grid_y = int((y+self._local_map_size/2.)/self._grid_size)
                if grid_x>=0 and grid_x<semantic_occu_map.shape[0] \
                and grid_y>=0 and grid_y<semantic_occu_map.shape[1]:
                    semantic_occu_map[grid_x,grid_y,0] = 1.0
            for i in range(n_laser):
                if i not in human_ray_indexs:
                    continue
                self._current_scan[i] = dynamic_ray_length[i]
                self.scan_intersection.append([(self.robot.px, self.robot.py), \
                                            (dynamic_scan[i,0],dynamic_scan[i,1])])
                x = scan_human_layer_local[i,0]
                y = scan_human_layer_local[i,1]
                grid_x = int((x+self._local_map_size/2.)/self._grid_size)
                grid_y = int((y+self._local_map_size/2.)/self._grid_size)
                if grid_x>=0 and grid_x<semantic_occu_map.shape[0] \
                and grid_y>=0 and grid_y<semantic_occu_map.shape[1]:
                    semantic_occu_map[grid_x,grid_y,1] = 1.0

            dxy = np.array(self.robot.get_goal_position())-np.array(self.robot.get_position())
            dg = np.linalg.norm(dxy)
            da = np.arctan2(dxy[1],dxy[0])
            vx = (self.robot.vx * np.cos(da) + self.robot.vy * np.sin(da))
            vy = (self.robot.vy * np.cos(da) - self.robot.vx * np.sin(da))

            dgtheta = self.robot.goal_theta-self.robot.theta
            dgv = self.robot.goal_v-np.linalg.norm(np.array([self.robot.vx,self.robot.vy]))

            return np.hstack([semantic_occu_map[...,1].flatten()+semantic_occu_map[...,0].flatten()*0.5,
                              np.array([dg,vx,vy,dgtheta,dgv,self.robot.radius],dtype=np.float32)]) 
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
        robot_action = ActionVW(action[0]*self.robot.v_pref,action[1]*self.robot.rotation_constraint)
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
        
        # cal reward
        step_type = dm_env.StepType.MID
        if (self._max_episode_length is not None and
            self._num_episode_steps >= self._max_episode_length):
          reward = 0
          step_type = dm_env.StepType.LAST
          discount = self._discount
        elif collision:
            reward = self._penalty_collision
            discount = self._discount
            step_type = dm_env.StepType.LAST
        else:
            reward,done = self.cal_reward(action)
            if done:
                step_type = dm_env.StepType.LAST
            discount = self._discount

        self._last_dg = self._state[-6]
        self.last_reward = reward

        return ExtendedTimeStep(
            step_type=step_type,
            action=action,
            reward=np.float32(reward),
            discount=discount,
            observation=self._state,
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

        plt.legend([robot, goal], ['Robot', 'Goal'], fontsize=16)

        human_circles = [plt.Circle(human.get_position(), human.radius, fill=False) for human in self.humans]


        for i in range(len(self.humans)):
            ax.add_artist(human_circles[i])
            artists.append(human_circles[i])
        
        for obstacle in self._layout["vertices"]:
            polygon = patches.Polygon(obstacle[:-1], closed=True, edgecolor='b', facecolor='none')
            ax.add_patch(polygon)
            artists.append(polygon)
        # for agent in self.humans + [self.robot]:
        #     x,y = agent.collider.exterior.xy
        #     ax.plot(x, y)
        if return_rgb:
            fig = plt.gcf()
            plt.axis('tight')
            # plt.subplots_adjust(0, 0, 1, 1, 0, 0)
            fig.canvas.draw()
            data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            w, h = fig.canvas.get_width_height()
            data = data.reshape((h, w, 3))
            plt.close(fig)
            return data
        else:
            if self.scan_intersection is not None:
                ii = 0
                lines = []
                while ii < n_laser:
                    lines.append(self.scan_intersection[ii])
                    ii = ii + 36
                lc = mc.LineCollection(lines,linewidths=1,linestyles='--',alpha=0.2)
                ax.add_artist(lc)
                artists.append(lc)
            plt.pause(0.1)
            
            for item in artists:
                item.remove()
           

if __name__ == "__main__":
    crowd_sim = build_crowdworld_task("point_goal_navi","train")
    step_ex = crowd_sim.reset()
    print(step_ex)

