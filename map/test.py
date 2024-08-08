from AUV_action.Reward import Reward
from Deep_Q_learning.DQN_parameters import Init_Parameters
from map.simulation_map.divide_environment import Graph
from parameter_type import Parameter_graph

x0 = 83
y0 = 76
x1 =27
y1 = 30
a = (y0-y1)/(x0-x1)
b = y0-a*x0

def judge_point(point):
    if point[1]==a*point[0]+b:
        print(False)
        print(str(point[0])+","+str(point[1]))

init_parameters = Init_Parameters()
reward_function = Reward([1,0],[0,1])
parameter_graph = Parameter_graph()
parameter_graph.set_x_xlim(init_parameters.get_x_lim())
parameter_graph.set_y_ylim(init_parameters.get_y_lim())
parameter_graph.set_griding_range(init_parameters.get_radius())
parameter_graph.set_reward_function(reward_function)
# 第一部，划分环境
graph = Graph(parameter_graph)
graph.generate_graph()
areas = graph.get_areas()
for area in areas.get_all_areas():
    point = area.get_left_up_point()
    judge_point(point)
    point = area.get_right_up_point()
    judge_point(point)
    point = area.get_right_down_point()
    judge_point(point)
    point = area.get_left_down_point()
    judge_point(point)


