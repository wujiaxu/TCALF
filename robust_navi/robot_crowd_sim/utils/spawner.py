import abc
from typing import List
import numpy as np
from robust_navi.robot_crowd_sim.utils.agent import Agent

class Spawner(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def spawnAgent(self,agent:Agent):

        return
    
class CircleSpawner(Spawner):

    def __init__(self,circle_size:float,agents:List[Agent]):
        self._agents:List[Agent]=agents
        self._circle_size = circle_size

    def spawnAgent(self, agent: Agent):

        counter = 2e4
        while True:
            angle = np.random.random() * np.pi * 2
            # add some noise to simulate all the possible cases robot could meet with human
            px_noise = (np.random.random() - 0.5) * agent.v_pref*2
            py_noise = (np.random.random() - 0.5) * agent.v_pref*2
            px = (self._circle_size /2. - 1)* np.cos(angle) + px_noise
            py = (self._circle_size /2. - 1) * np.sin(angle) + py_noise

            # TODO consider obstacle
            collide = False
            for other in self._agents:
                if other.id >= agent.id:
                    continue
                min_dist = agent.radius + other.radius + 0.2#self._discomfort_dist
                if np.linalg.norm((px - other.px, py - other.py)) < min_dist or \
                        np.linalg.norm((-px - other.gx, -py - other.gy)) < min_dist:
                    collide = True
                    break
            if not collide or counter<0:
                break
            counter-=1
        # robot.start_pos.append((px, py))
        agent.set(px, py, -px, -py, 0, 0, np.arctan2(-py,-px))
        agent.task_done = False

        return 