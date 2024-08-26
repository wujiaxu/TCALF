import matplotlib.pyplot as plt
from matplotlib import collections as mc
import matplotlib.lines as mlines
from matplotlib import patches
from typing import List
import numpy as np
from robust_navi.robot_crowd_sim.utils.agent import Agent
from robust_navi.robot_crowd_sim.utils.map import Map
class Render:
    def __init__(self,map,robot,crowd):

        self._map:Map = map
        self._robot:Agent = robot
        self._crowd:List[Agent] = crowd

        fig,self.render_axis = plt.subplots(figsize=(8,8)) 
        
        plt.xlim(-self._map._map_size/2.-1,self._map._map_size/2.+1)
        plt.ylim(-self._map._map_size/2.-1,self._map._map_size/2.+1)

    def rend(self,mode,time=None): #TODO: add timer
        plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'
        ax=self.render_axis
        robot_color = 'yellow'
        goal_color = 'red'
        human_color = 'black'
        artists = []

        # map
        x_b,y_b = self._map.getBoundary()
        x_b,y_b = list(x_b),list(y_b)
        boundary = [(x_b[i],y_b[i]) for i in range(len(x_b))]
        polygon = patches.Polygon(boundary[:-1], closed=True, edgecolor='b', facecolor='none')
        ax.add_patch(polygon)
        artists.append(polygon)

        # add robot
        goal=mlines.Line2D([self._robot.gx], [self._robot.gy], color=goal_color, marker='*', linestyle='None', markersize=15, label='RobotGoal')
        ax.add_artist(goal)
        artists.append(goal)
        robotX,robotY=self._robot.get_position()
        robot_disk=plt.Circle((robotX,robotY), self._robot.radius, fill=True, color=robot_color)
        ax.add_artist(robot_disk)
        artists.append(robot_disk)
        plt.legend([robot_disk, goal], ['Robot', 'RobotGoal'], bbox_to_anchor=(0.85, 0.85), loc='upper left', fontsize=16)

        # add crowd
        for human in self._crowd:
            goal=mlines.Line2D([human.gx], [human.gy], color=human_color, marker='*', linestyle='None', markersize=15, label='HumanGoal')
            ax.add_artist(goal)
            artists.append(goal)
            # add robot
            humanX,humanY=human.get_position()

            human_disk=plt.Circle((humanX,humanY), human.radius, fill=True, color=human_color)
            ax.add_artist(human_disk)
            artists.append(human_disk)
        if mode == "human":
            plt.pause(0.1)
            
            for item in artists:
                item.remove()
        elif mode == "return_rgb":
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
        return