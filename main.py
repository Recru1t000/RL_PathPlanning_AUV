import math
import time

from AUV_action.AUV_based import Base_Parameters
from AUV_action.Reward import Reward
from map.simulation_map.utility import FindIntersections
from parameter_type import Parameter_base_show, Parameter_graph
from map.simulation_map.base_map import base_map
from map.simulation_map.explorer import explorer
from map.simulation_map.obstacle import Obstacles
from map.simulation_map.divide_environment import Graph
from AUV_action.APF import Artificial_Potential_Field

obstacles = Obstacles()  # 障碍应该按照左下，右下，右上，左上，左下的方式添加
obstacles.add_obstacles([[(10, 3), (10, 5), (12, 5), (12, 3), (10, 3)]])
obstacles.add_obstacles([[(30, 35), (60, 35), (60, 60), (30, 60), (30, 35)]])
obstacles.add_obstacles([[(10, 20), (20, 20), (20, 25), (10, 25), (10, 20)]])

#explorers = explorer([1, 2,  4, 5, 6, 7, 8], [30, 25,  23, 25, 32, 12, 24], [40, 30])

#环境基本参数
#在设定基本参数的时候，我们需要确保能够被整除
x_xlim = 100
y_ylim = 100
radius = 5
start_point = [10,10]
end_point = [80,70]

base_map1 = base_map(x_xlim, y_ylim, 10)
base_map1.set_obstacles(obstacles.get_obstacles())
#base_map1.set_explorer(explorers)
#explorers = explorer([2, 4, 6, 7], [10, 2, 11, 13], [10, 20])
#base_map1.set_explorer(explorers)
base_map1.set_goal_point([end_point])
#base_map1.collision()
#base_map1.show()

#起点的设置在apf的explorer里
#参数设置
parameter_base_show = Parameter_base_show()#用于传递可视化的参数
reward_function = Reward(start_point,end_point)
parameter_graph = Parameter_graph()
parameter_graph.set_x_xlim(x_xlim)
parameter_graph.set_y_ylim(y_ylim)
parameter_graph.set_griding_range(radius)
parameter_graph.set_reward_function(reward_function)

#第一部，划分环境
graph = Graph(parameter_graph)
graph.generate_graph()

#第二步，确定移动路径
apf = Artificial_Potential_Field(base_map1)
path = apf.move_to_goal()
parameter_base_show.set_base_show_path_points(path)#将移动路径点传递到传递可视化
print(path)

#第三部，赋予权重
#1.给予每个边上AUV要到达的点的坐标，直到这个点超出区域，确定超出区域之前和之后的点，确定该区域边最后的交点。
path_points = FindIntersections().find_intersections(path,x_xlim,radius)
#为路径可视化FindIntersections().plot_grid_and_segments(path,x_xlim,radius,path_points)
#2.赋予权重(目前未考虑一个角到另一个角的情况)
graph.generate_edge_reward(path_points)

#第四步，移动探索
#0.判断AUV区域
#1.遍历下一个点，直到这个点超出区域，确定超出区域之前和之后的点，确定该区域边最后的交点。
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
print(base_map1.get_init_points())
print("剩余电量")
print("当前距离目标点位置")
#输出
print("第n个点")
#reward
print("消耗的电量")
print("探索到探索点的多少")
print("距离目标点的距离")

#base_map1.set_line_rewards(g.get_line_rewards())
base_map1.base_show(parameter_base_show)

base_parameters = Base_Parameters(1,1,1,1)