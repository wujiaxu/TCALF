import abc
import logging
import numpy as np
from numpy.linalg import norm
from controllable_navi.crowd_sim.policy.policy_factory import policy_factory
from controllable_navi.crowd_sim.utils.action import ActionXY, ActionRot, ActionVW
from controllable_navi.crowd_sim.utils.state import ObservableState, FullState, ExFullState

import shapely
from shapely.geometry import Polygon, LineString, MultiPolygon,Point
import typing as tp

class Agent(object):
    def __init__(self, config, section):
        """
        Base class for robot and human. Have the physical attributes of an agent.

        """
        self.visible = getattr(config, section).visible
        self.v_pref = getattr(config, section).v_pref
        self.radius = getattr(config, section).radius
        self.policy = policy_factory[getattr(config, section).policy]()
        # self.sensor = getattr(config, section).sensor
        self.kinematics = self.policy.kinematics if self.policy is not None else None
        self.px : float 
        self.py : float
        self.gx : float
        self.gy : float
        self.vx : float
        self.vy : float
        self.theta : float
        self.goal_theta : float
        self.goal_v : float
        self.time_step : float
        self.w : float

        #TODO deal with different shape
        self.collider = None 

        self.time_sampling = 10

    def print_info(self):
        logging.info('Agent is {} and has {} kinematic constraint'.format(
            'visible' if self.visible else 'invisible', self.kinematics))

    def set_policy(self, policy):
        if self.time_step is None:
            raise ValueError('Time step is None')
        policy.set_time_step(self.time_step)
        self.policy = policy
        self.kinematics = policy.kinematics

    def sample_random_attributes(self):
        """
        Sample agent radius and v_pref attribute from certain distribution
        :return:
        """
        self.v_pref = np.random.uniform(0.5, 1.5)
        self.radius = np.random.uniform(0.2, 0.4)

    def set(self, px:float, py:float, gx:float, gy:float, vx:float, vy:float, theta:float, w:float=0.,
            radius:tp.Optional[float]=None, v_pref:tp.Optional[float]=None,goal_theta:tp.Optional[float]=None,goal_v:tp.Optional[float]=None):
        self.px = px
        self.py = py
        self.sx = px
        self.sy = py
        self.gx = gx
        self.gy = gy
        self.vx = vx
        self.vy = vy
        self.theta = theta
        self.w = w
        if radius is not None:
            self.radius = radius
        if v_pref is not None:
            self.v_pref = v_pref
        if goal_theta is not None:
            self.goal_theta = goal_theta
        if goal_v is not None:
            self.goal_v = goal_v
        self.collider = Point(px, py).buffer(self.radius)

    def get_observable_state(self):
        return ObservableState(self.px, self.py, self.vx, self.vy, self.radius)

    def get_next_observable_state(self, action):
        self.check_validity(action)
        pos = self.compute_position(action, self.time_step)
        next_px, next_py = pos
        if self.kinematics == 'holonomic':
            next_vx = action.vx
            next_vy = action.vy
        else:
            next_vx = action.v * np.cos(self.theta)
            next_vy = action.v * np.sin(self.theta)
        return ObservableState(next_px, next_py, next_vx, next_vy, self.radius)

    def get_full_state(self):
        return FullState(self.px, self.py, self.vx, self.vy, self.radius, self.gx, self.gy, self.v_pref, self.theta)

    def get_ex_full_state(self):
        return ExFullState(self.px, self.py, self.vx, self.vy, self.radius, self.gx, self.gy, self.v_pref, self.theta,self.goal_theta,self.goal_v)
    def get_position(self):
        return self.px, self.py

    def set_position(self, position):
        self.px = position[0]
        self.py = position[1]

    def get_goal_position(self):
        return self.gx, self.gy

    def get_start_position(self):
        return self.sx, self.sy

    def get_velocity(self):
        return self.vx, self.vy

    def set_velocity(self, velocity):
        self.vx = velocity[0]
        self.vy = velocity[1]

    @abc.abstractmethod
    def act(self, ob):
        """
        Compute state using received observation and pass it to policy

        """
        return

    def check_validity(self, action):
        if self.kinematics == 'holonomic':
            assert isinstance(action, ActionXY)
        elif self.kinematics == 'unicycle':
            assert isinstance(action,ActionVW)
        else:
            assert isinstance(action, ActionRot)

    def compute_position(self, action, delta_t):
        self.check_validity(action)
        if self.kinematics == 'holonomic':
            px = self.px + action.vx * delta_t
            py = self.py + action.vy * delta_t
        elif self.kinematics == 'unicycle':
            self.theta = (self.theta + action.w * delta_t) % (2 * np.pi)
            px = self.px + np.cos(self.theta) * action.v * delta_t
            py = self.py + np.sin(self.theta) * action.v * delta_t
        else:
            theta = self.theta + action.r
            px = self.px + np.cos(theta) * action.v * delta_t
            py = self.py + np.sin(theta) * action.v * delta_t

        return px, py

    def step(self, action):
        """
        Perform an action and update the state
        """
        self.check_validity(action)
        for i in range(self.time_sampling):
            pos = self.compute_position(action, self.time_step/self.time_sampling)
            self.px, self.py = pos
            if self.kinematics == 'holonomic':
                self.vx = action.vx
                self.vy = action.vy
            elif self.kinematics == 'unicycle':
                self.vx = action.v * np.cos(self.theta)
                self.vy = action.v * np.sin(self.theta)
                self.w  = action.w
            else:
                self.theta = (self.theta + action.r) % (2 * np.pi)
                self.vx = action.v * np.cos(self.theta)
                self.vy = action.v * np.sin(self.theta)
        self.collider = Point(self.px, self.py).buffer(self.radius)

    def reached_destination(self):
        return norm(np.array(self.get_position()) - np.array(self.get_goal_position())) < self.radius

