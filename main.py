import math
import time

import numpy as np

from AUV_action.AUV_based import Base_Parameters
from AUV_action.Reward import Reward
from Deep_Q_learning.DQN_environment import DQN_Environment
from Deep_Q_learning.DQN_parameters import State, Init_Parameters
from Deep_Q_learning.power_consumption import Power_Consumption
from map.simulation_map.utility import FindIntersections
from parameter_type import Parameter_base_show, Parameter_graph
from map.simulation_map.base_map import base_map
from map.simulation_map.explorer import explorer
from map.simulation_map.obstacle import Obstacles
from map.simulation_map.divide_environment import Graph
from AUV_action.APF import Artificial_Potential_Field





init_parameters = Init_Parameters()
#环境基本参数
#在设定基本参数的时候，我们需要确保能够被整除



#起点的设置在apf的explorer里








env = DQN_Environment(init_parameters)
env.reset()
t= env.step([0,1,0.9,0])
print("1")

#2.执行DQN，确定探索的边
#3.执行探索
#4.返回结果，确定是否有障碍，返回reward并消耗能量
#5.如果有障碍执行第五步，

#第五步，预到障碍
#1.areas_reset
'''
g = Graph(100,100,5)
g.generate_graph()
g.generate_line_reward()
g.generate_line_reward_by_points(base_map1.get_init_points())
'''
#输入
print("剩余电量")
print("当前距离目标点位置")
#输出
print("第n个点")
#reward
print("消耗的电量")
print("探索到探索点的多少")
print("距离目标点的距离")

#base_map1.set_line_rewards(g.get_line_rewards())

base_parameters = Base_Parameters(1,1,1,1)