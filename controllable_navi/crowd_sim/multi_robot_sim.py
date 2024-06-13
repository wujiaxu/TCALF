import pdb
import dataclasses
from shapely.geometry import MultiPolygon, LinearRing,Point
import matplotlib.pyplot as plt
from matplotlib import collections as mc
import matplotlib.lines as mlines
from matplotlib import patches
import numpy as np
from numpy.linalg import norm
import copy

from controllable_navi.crowd_sim.C_library.motion_plan_lib import *
# from controllable_navi.crowd_sim.policy.policy_factory import policy_factory
# from controllable_navi.crowd_sim.utils.state import tensor_to_joint_state, JointState,ExFullState
from controllable_navi.crowd_sim.utils.action import ActionVW
from controllable_navi.crowd_sim.utils.robot import Robot
from controllable_navi.crowd_sim.utils.info import *
# from controllable_navi.crowd_sim.utils.utils import point_to_segment_dist
from controllable_navi.crowd_sim.utils.obstacle import *

import controllable_navi.dm_env_light as dm_env
from controllable_navi.dm_env_light import specs
import enum
from hydra.core.config_store import ConfigStore
from controllable_navi.crowd_sim.crowd_sim import InformedTimeStep,NaviObsSpace

# laser scan parameters
# number of all laser beams
from controllable_navi.crowd_sim.crowd_sim import n_laser,laser_angle_resolute,laser_min_range,laser_max_range


class MultiObservationType(enum.IntEnum):
    RAW_SCAN = enum.auto()
    TIME_AWARE_RAW_SCAN = enum.auto()

class MultiRobotPhysics:

    def __init__(self,env):
        self._env:MultiRobotWorld = env
        # data buffer
        # [agent_num, agent_states, static_obs]
        # item          size        content
        # agent_num:    1           current agent number
        # agent_states  5XN         x,y,vx,vy,r
        # static_obs    100         obstacle_num inf-divider (x,y)-polygon inf-divider (x,y)-polygon
    def get_state(self,id):
        # get robot-crowd physics
        physics = [self._env.current_robot_num]
        physics += list(self._env.robots[id].get_observable_state().to_tuple())
        for robot in self._env.robots[:self._env.current_robot_num]:
            if robot.id == id:
                continue
            physics += list(robot.get_observable_state().to_tuple())
        physics += (1+len(self._env.robots)*5-len(physics))*[-1]
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
class MultiRobotSimConfig:
    _target_: str = "controllable_navi.crowd_sim.multi_robot_sim.MultiRobotWorld"
    name: str = "MultiRobotWorld"
    scenario: str = "default"
    max_robot_num: int = 6 
    map_size: float = 8
    with_static_obstacle: bool = True
    regen_map_every: int = 500

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
cs.store(group="crowd_sim", name="MultiRobotWorld", node=MultiRobotSimConfig)

def build_multirobotworld_task(cfg:MultiRobotSimConfig, task,phase,
                         discount=1.0,
                         observation_type="time_aware_raw_scan",
                         max_episode_length=200):
    
    if observation_type =="raw_scan":
        obs_type = MultiObservationType.RAW_SCAN
    elif observation_type =="time_aware_raw_scan":
        obs_type = MultiObservationType.TIME_AWARE_RAW_SCAN
    else:
        raise NotImplementedError

    # currently robot and human class need Config class to init
    class Config(object):
        def __init__(self):
            pass
    class EnvConfig(object):

        env = Config()
        env.robot_num = [2,cfg.max_robot_num]
        env.map_size = cfg.map_size

        robot = Config()
        robot.visible = cfg.robot_visible
        robot.policy = 'none'
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
    return MultiRobotWorld(phase,cfg, config,obs_type,discount,max_episode_length,**tasks_specifications[task])


class MultiRobotWorld(dm_env.Environment):

    def __init__(self, phase,
                 cfg:MultiRobotSimConfig, 
                 config,
               observation_type=MultiObservationType.TIME_AWARE_RAW_SCAN,
               discount=1.0,
               max_episode_length=50,
               forbiden_zone_y = 0.6,
               discomfort_dist = 0.2,
               to_wall_dist = 0.2,
               speed_limit = 1.0,
               reward_func_ids=0) -> None:
        #TODO: modify input to adapt crowdsim
        if observation_type not in MultiObservationType:
            raise ValueError('observation_type should be a ObservationType instace.')
        
        self.nenv = 1
        self.thisSeed = 0
        self.phase = phase
        self.config = config
        self.physics = MultiRobotPhysics(self)
        self._scenario = cfg.scenario
        self._map_size = self.config.env.map_size #circle
        self._with_static_obstacle = cfg.with_static_obstacle
        self._regen_map_every = cfg.regen_map_every
        self._layout: dict = {"polygon":[],"vertices":[],"boundary":None}
        map_boundary = LinearRing( ((-self._map_size/2., self._map_size/2.), 
                                    (self._map_size/2., self._map_size/2.),
                                    (self._map_size/2., -self._map_size/2.),
                                    (-self._map_size/2.,-self._map_size/2.)) )
        self._layout["boundary"] = map_boundary

        self._robot_num = self.config.env.robot_num
        self._fix_robot_num = False
        self.current_robot_num = 0
        self._time_step = 0.25
        self.robots : list[Robot] = [Robot(config,"robot",id) for id in range(self._robot_num[1])]
        for robot in self.robots:
            robot.time_step = self._time_step
            robot.kinematics = "unicycle"
        self._step_info: list[InfoList]= [InfoList() for _ in range(self._robot_num[1])]
        self._current_scan = np.zeros(n_laser, dtype=np.float32)

        self._discount = discount
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
        self._observation_type = observation_type
        self._local_map_size = cfg.local_map_size
        self._grid_size = cfg.grid_size
        self._occu_map_size = int((self._local_map_size//self._grid_size)**2) 


        self.case_capacity = {'train': np.iinfo(np.uint32).max - 2000, 'val': 1000, 'test': 1000}
        self.case_size = {'train': np.iinfo(np.uint32).max - 2000, 'val': 2000,
                          'test': 1000}
        self.case_counter = {'train': 0, 'test': 0, 'val': 0}
        self._state = None
        
        self.global_time : float = 0
        self._num_episode_steps : int = 0
        self._max_episode_length: int = max_episode_length
        

        # for visualization
        self.render_axis = None

    def set_multi_env_seed(self,nenv,seed):
        self.thisSeed = seed
        self.nenv = nenv
        return
    
    def random_fix_human_num(self,human_num):
        assert human_num>=self._robot_num[0]
        assert human_num<self._robot_num[1]+1
        self.current_robot_num = human_num #np.random.randint(self._robot_num[0],self._robot_num[1]+1)
        self._fix_robot_num=True
        return  
    
    def init_render_ax(self,ax):
        if ax is None:
            fig, ax = plt.subplots(figsize=(8,8)) 
        
        plt.xlim(-self._map_size/2.-1,self._map_size/2.+1)
        plt.ylim(-self._map_size/2.-1,self._map_size/2.+1)
        #self.text = ax.text(0, self._map_size/2.+1.2, 'v:{}[m/s]'.format(0.), fontsize=14, color='black', ha='center', va='top')
        self.render_axis = ax
        # if self.config.env.render:
        #     plt.ion()
        #     plt.show()
    
    def cal_reward(self,robot):
        done = False
        reward = 0
        dxy = np.array(robot.get_goal_position())-np.array(robot.get_position())
        dg = np.linalg.norm(dxy)
        if dg <self._goal_range:
                reward = self._reward_goal * (1-self.global_time/self._max_episode_length)
                done = True
                self._step_info[robot.id].add(ReachGoal())
        reward += self._goal_factor * (robot._last_dg-dg)

        robot._last_dg = dg # type: ignore

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
    
    def generate_robot(self, robot, non_stop=False, square=False):
        # random human attribute
        # robot.sample_random_attributes()
        if square is False and non_stop is False:
            counter = 2e4
            while True:
                angle = np.random.random() * np.pi * 2
                # add some noise to simulate all the possible cases robot could meet with human
                px_noise = (np.random.random() - 0.5) * robot.v_pref*2
                py_noise = (np.random.random() - 0.5) * robot.v_pref*2
                px = (self._map_size /2. - 1)* np.cos(angle) + px_noise
                py = (self._map_size /2. - 1) * np.sin(angle) + py_noise

                collider = Point(px, py).buffer(robot.radius)
                collide = False
                # print(self._layout["polygon"])
                for polygon in self._layout["polygon"]:
                    if polygon.intersects(collider):
                        collide = True
                        break
                if collide:
                    continue
                collider = Point(-px, -py).buffer(robot.radius)
                collide = False
                # print(self._layout["polygon"])
                for polygon in self._layout["polygon"]:
                    if polygon.intersects(collider):
                        collide = True
                        break
                if collide:
                    continue
                for agent in self.robots:
                    if agent.id >= robot.id:
                        continue
                    min_dist = robot.radius + agent.radius + 0.2#self._discomfort_dist
                    if norm((px - agent.px, py - agent.py)) < min_dist or \
                            norm((-px - agent.gx, -py - agent.gy)) < min_dist:
                        collide = True
                        break
                if not collide or counter<0:
                    break
                counter-=1
            # robot.start_pos.append((px, py))
            robot.set(px, py, -px, -py, 0, 0, np.arctan2(-py,-px))
        elif square is False and non_stop is True:
            raise NotImplementedError

        return 
    
    def observation_spec(self):
        if self._observation_type is MultiObservationType.RAW_SCAN:
            return NaviObsSpace({"scan":(720,),"robot_state":(5,)}, dtype=np.float32, name='relative pose to goal')
        elif self._observation_type is MultiObservationType.TIME_AWARE_RAW_SCAN:
            return NaviObsSpace({"scan":(720,),"robot_state":(7,)}, dtype=np.float32, name='relative pose to goal')
        else:
            raise NotImplementedError

    def action_spec(self):
        return specs.BoundedArray(shape=(2,), dtype=np.float32, name='action', minimum=-1.0, maximum=1.0)
    #specs.Array(shape=(2,), dtype=np.float32, name='action')
    
    # TODO add custom case no. input
    def reset(self,test_case=None):
        self._num_episode_steps = 0
        train_seed_begin = [0, 10, 100, 1000, 10000]
        val_seed_begin = [0, 10, 100, 1000, 10000]
        test_seed_begin = [0, 10, 100, 1000, 10000]
        base_seed = {'train': self.case_capacity['val'] + self.case_capacity['test'] + train_seed_begin[1],
                     'val': 0 + val_seed_begin[1], 'test': self.case_capacity['val']+test_seed_begin[2]+1000}

        
        self.random_seed = base_seed[self.phase] + self.case_counter[self.phase] + self.thisSeed
        np.random.seed(self.random_seed)
        if self.case_counter[self.phase]%self._regen_map_every==0:
            self.randomize_map()

        if self._robot_num[1]==2: #TOY ENV
            self.robots[0].set(0,-2, 0, 2, 0,0, np.arctan2(2,0))
            self.robots[1].set(0,2, 0, -2, 0,0, np.arctan2(-2,0))
            self.current_robot_num = 2
        else:            
            if self.current_robot_num == 0 or self._fix_robot_num==False:
                self.current_robot_num = np.random.randint(self._robot_num[0],self._robot_num[1]+1)
            else: pass # the robot num will remain the same for vec-env fixed data buffer
            for i in range(self.current_robot_num):
                self.generate_robot(self.robots[i])
        self._num_episode_steps = 0
        self.global_time = 0
        self.case_counter[self.phase] = (self.case_counter[self.phase] + + int(1*self.nenv)) % self.case_size[self.phase]
        
        self._state, _ = self.get_obs()
        for robot in self.robots[:self.current_robot_num]:
            dxy = np.array(robot.get_goal_position())-np.array(robot.get_position())
            robot._last_dg = norm(dxy)
            robot.task_done = False
        for i, step_info in enumerate(self._step_info[:self.current_robot_num]):
            step_info.reset()
            step_info.add(Nothing())
            step_info.get_task_info(self.robots[i].px,self.robots[i].py,self.robots[i].gx,self.robots[i].gy)

        return [InformedTimeStep(
            step_type=dm_env.StepType.FIRST,
            action=np.array([0,0]),
            reward=0.0,
            discount=1.0,
            observation=self._state[i],
            info = copy.deepcopy(self._step_info[i]),
            ) for i in range(self.current_robot_num)]
    
    def get_static_scan(self,id:int):
        num_line = sum([len(obstacle)-1 for obstacle in self._layout["vertices"]])+4 #for boundary
        num_circle = 0
        scan = np.zeros(n_laser, dtype=np.float32)
        scan_end = np.zeros((n_laser, 2), dtype=np.float32)
        robot_pose = np.array([self.robots[id].px, self.robots[id].py, self.robots[id].theta])
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
    
    def get_dynamic_scan(self,id:int):
        num_line = 4
        num_circle = self.current_robot_num-1
        scan = np.zeros(n_laser, dtype=np.float32)
        scan_end = np.zeros((n_laser, 2), dtype=np.float32)
        robot_pose = np.array([self.robots[id].px, self.robots[id].py, self.robots[id].theta])
        # print(self._layout)
        InitializeEnv(num_line, num_circle, n_laser, laser_angle_resolute)
        x_b,y_b = self._layout["boundary"].coords.xy
        for i in range(4):
            line = [(x_b[i],y_b[i]),(x_b[i+1],y_b[i+1])]
            set_lines(4 * i    , line[0][0])
            set_lines(4 * i + 1, line[0][1])
            set_lines(4 * i + 2, line[1][0])
            set_lines(4 * i + 3, line[1][1])
        
        i=0
        for robot in self.robots[:self.current_robot_num]:
            if robot.id == id:
                continue
            set_circles(3 * i    , robot.px)
            set_circles(3 * i + 1, robot.py)
            set_circles(3 * i + 2, robot.radius)
            i+=1
        set_robot_pose(robot_pose[0], robot_pose[1], robot_pose[2])
        cal_laser() #memory leak
        
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
        multi_robot_obs = []
        collisions = []
        for id in range(self.current_robot_num):
            if self.robots[id].task_done == True:
                collisions.append(False)
                multi_robot_obs.append(np.zeros(self.observation_spec().shape,
                                                dtype=self.observation_spec().dtype))
                continue
            dxy = np.array(self.robots[id].get_goal_position())-np.array(self.robots[id].get_position())
            dg = norm(dxy)
            goal_direction = np.arctan2(dxy[1],dxy[0])
            hf = (self.robots[id].theta-goal_direction)% (2 * np.pi)
            if hf > np.pi:
                hf -= 2 * np.pi
            # transform dxy to base frame
            vx = (self.robots[id].vx * np.cos(goal_direction) + self.robots[id].vy * np.sin(goal_direction))
            vy = (self.robots[id].vy * np.cos(goal_direction) - self.robots[id].vx * np.sin(goal_direction)) 
            
            static_ray_length,static_scan = self.get_static_scan(id)
            dynamic_ray_length,dynamic_scan = self.get_dynamic_scan(id)

            human_ray_indexs = np.where(dynamic_ray_length<static_ray_length)[0].tolist()
            obstacle_ray_indexs = np.where(static_ray_length<=dynamic_ray_length)[0].tolist()

            for i in range(n_laser):
                if i not in obstacle_ray_indexs:
                    continue
                self._current_scan[i] = static_ray_length[i]
            for i in range(n_laser):
                if i not in human_ray_indexs:
                    continue
                self._current_scan[i] = dynamic_ray_length[i]

            dmin =np.min(self._current_scan)
            if dmin <= self.robots[id].radius:
                collisions.append(True)
            else:
                collisions.append(False)
        
            if self._observation_type == MultiObservationType.RAW_SCAN:
                multi_robot_obs.append(np.hstack([np.clip(self._current_scan,0.,self._local_map_size/2.),
                                np.array([dg,hf,vx,vy,self.robots[id].radius],dtype=np.float32)]) )
            elif self._observation_type == MultiObservationType.TIME_AWARE_RAW_SCAN:
                multi_robot_obs.append(
                                np.hstack([np.clip(self._current_scan,0.,self._local_map_size/2.),
                                np.array([self._num_episode_steps/self._max_episode_length,
                                            np.log(self._max_episode_length),
                                            dg, hf,vx,vy,self.robots[id].radius],dtype=np.float32)]) )
            else:
                raise NotImplementedError
        
        return multi_robot_obs,collisions

    def step(self, actions):
        """
        #actions: BX2
        """
        for i, robot in enumerate(self.robots[:self.current_robot_num]):
            if robot.task_done == True:
                continue
            robot_action = ActionVW(((actions[i,0]+1.)/2.)*robot.v_pref,actions[i,1]*robot.rotation_constraint)
            # update all agents
            robot.step(robot_action)

        self._state,collisions = self.get_obs()
        
        self.global_time += self._time_step
        self._num_episode_steps += 1
        
        discounts = []
        rewards = []
        step_types = []
        for i, robot in enumerate(self.robots[:self.current_robot_num]):
            if robot.task_done == True:
                rewards.append(0.)
                discounts.append(0.)
                step_type = dm_env.StepType.LAST
                step_types.append(step_type)
                continue
            self._step_info[i].reset()
            step_type = dm_env.StepType.MID
            
            # cal reward
            discount = 1.0
            if collisions[i]:
                reward = self._penalty_collision
                discount = 0
                step_type = dm_env.StepType.LAST
                self._step_info[i].add(Collision())
                robot.task_done = True
            elif (self._max_episode_length is not None and
                self._num_episode_steps >= self._max_episode_length):
                reward = -self._reward_goal
                step_type = dm_env.StepType.LAST
                discount = 0
                self._step_info[i].add(Timeout())
                robot.task_done = True
            else:
                reward,done= self.cal_reward(robot)
                if done:
                    step_type = dm_env.StepType.LAST
                    discount = 0
                    robot.task_done = True
            rewards.append(reward)
            discounts.append(discount)
            step_types.append(step_type)
            self.last_reward = reward
            if self._step_info[i].empty():
                self._step_info[i].add(Nothing())

        return [InformedTimeStep(
            step_type=step_types[i],
            action=actions[i],
            reward=np.float32(rewards[i]),
            discount=discounts[i],
            observation=self._state[i],
            info = copy.deepcopy(self._step_info[i]),
            ) for i in range(self.current_robot_num)]

    def render(self, return_rgb=True):
        # x,y = self._layout["boundary"].exterior.xy
        # plt.plot(x, y)
        plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'
        ax=self.render_axis
        robot_color = 'yellow'
        goal_color = 'red'
        artists = []
        for robot in self.robots[:self.current_robot_num]:
            goal=mlines.Line2D([robot.gx], [robot.gy], color=goal_color, marker='*', linestyle='None', markersize=15, label='Goal')
            ax.add_artist(goal)
            artists.append(goal)

            # add robot
            robotX,robotY=robot.get_position()

            robot_disk=plt.Circle((robotX,robotY), robot.radius, fill=True, color=robot_color)
            ax.add_artist(robot_disk)
            artists.append(robot_disk)
        plt.legend([robot_disk, goal], ['Robot', 'Goal'], bbox_to_anchor=(0.85, 0.85), loc='upper left', fontsize=16)
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
        
        if return_rgb:
            fig = plt.gcf()
            # self.text.set_text('v:{}[m/s]'.format(norm([self.robot.vx,self.robot.vy])))
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

    def close(self):

        #TODO

        return
           

if __name__ == "__main__":
    from pathlib import Path
    import sys
    import faulthandler
    faulthandler.enable()

    base = Path(__file__).absolute().parents[1]
    # we need to add base repo to be able to import controllable_navi
    # we need to add controllable_navi to be able to reload legacy checkpoints
    for fp in [base, base / "controllable_navi"]:
        assert fp.exists()
        if str(fp) not in sys.path:
            sys.path.append(str(fp))
    cfg = MultiRobotSimConfig()
    cfg.max_robot_num = 2
    crowd_sim = build_multirobotworld_task(cfg,"point_goal_navi","train")
    step_ex = crowd_sim.reset()
    crowd_sim.robots[0].set(0,0,1,1,0,0,0)
    crowd_sim.robots[1].set(0,0,1,1,0,0,0)
    step_ex = crowd_sim.step(np.array([[1,0],[1,0]]))
    print(crowd_sim._step_info[0].info_list)
    print()
    print(crowd_sim._step_info[1].info_list)

