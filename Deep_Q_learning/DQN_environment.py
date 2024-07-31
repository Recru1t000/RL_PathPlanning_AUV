import math

import numpy as np

from AUV_action.APF import Artificial_Potential_Field
from AUV_action.Reward import Reward
from AUV_action.new_explore import DQN_Explorer
from Deep_Q_learning.DQN_parameters import State
from Deep_Q_learning.power_consumption import Power_Consumption
from Deep_Q_learning.time_consumption import Time_Consumption
from map.simulation_map.base_map import base_map
from map.simulation_map.divide_environment import Graph
from map.simulation_map.explorer import explorer
from map.simulation_map.obstacle import Obstacles
from map.simulation_map.utility import FindIntersections
from parameter_type import Parameter_base_show, Parameter_graph


class DQN_Environment():
    def __init__(self,init_parameters):
        self.state = None
        self.graph = None
        self.power_consumption = Power_Consumption(init_parameters.get_radius(),init_parameters.get_power_consumption_value())
        self.basemap = None
        self.init_parameters = init_parameters
        self.parameter_base_show = Parameter_base_show()
        self.path_points = None
        self.reward = None
        self.time_consumption = None

    def step(self,action):

        #确定探索的方向和半径
        new_explorer = DQN_Explorer(self.graph.get_griding_range())
        areas = self.graph.get_areas()
        area = areas.get_area_AUV_in()
        edge_list = self.choose_edge(action,area)
        explore_angles_and_radius = self.get_explore_angles_and_radius(edge_list,new_explorer)

        #执行能量消耗
        reduce_power = self.power_consumption.explorer_power_consumption(explore_angles_and_radius)
        self.state.reduce_power(reduce_power)#能量消耗的reward

        #执行探索时间消耗
        reduce_time = self.time_consumption.get_explore_time()
        self.state.reduce_time(reduce_time)

        #执行能量消耗，时间消耗，边探索的reward
        self.reward.get_edge_reward(edge_list,reduce_power,self.state.get_auv_point(),self.state.get_target_point(),self.power_consumption.get_power_consumption_full_sensing())
        self.reward.get_power_consumption_reward(reduce_power)
        self.reward.get_time_consumption_reward(reduce_time)

        #执行探索
        explore_obstacle = False
        for key,value in explore_angles_and_radius.items():
            old_explorer = explorer([new_explorer.old_explorer_to_new(key)], [value], self.state.get_auv_point())
            if self.basemap.set_explorer(old_explorer):
                explore_obstacle = True


        #可视化
        self.show_the_path()
        #如果探索到障碍,扣除reward
        if explore_obstacle:
            #重新规划路线
            apf = Artificial_Potential_Field(self.basemap)
            apf.set_explorer(explorer([1], [1], self.state.get_auv_point()))
            path = apf.move_to_goal()
            #重新赋予权重
            path_points = FindIntersections().find_intersections(path, self.init_parameters.get_x_lim(),self.init_parameters.get_radius())#不需要插入第一个点，因为当前AUV必然在边上
            self.graph.generate_edge_reward(path_points)
            self.path_points = path_points
            #扣除reward
            return self.state.next_state(),self.reward.get_reward(),False,False
        #如果没有探索到下一个点的边，扣除reward重新行动。
        else:
            if len(self.path_points)!=0:
                self.state.set_auv_point(self.path_points.pop(0))
            else:
                self.state.set_auv_point(self.init_parameters.get_init_target_point())

            if len(self.path_points)!=0:
                next_point = self.path_points[0]
            else:
                next_point = self.init_parameters.get_init_target_point()

            self.state.set_target_point(next_point)
            self.show_the_path()
            # 执行时间消耗
            reduce_time = self.time_consumption.move_time_consumption(self.state.get_auv_point(),next_point)
            self.state.reduce_time(reduce_time)
            #执行时间reward
            self.reward.get_time_consumption_reward(reduce_time)
            #判断是否到达终点
            if self.state.get_auv_point()==self.init_parameters.get_init_target_point():
                done = True
            else:
                done = False
            return self.state.next_state(),self.reward.get_reward(),done,False

    def reset(self):
        obstacles = Obstacles()  # 障碍应该按照左下，右下，右上，左上，左下的方式添加
        obstacles.add_obstacles([[(10, 3), (10, 5), (12, 5), (12, 3), (10, 3)]])
        obstacles.add_obstacles([[(30, 35), (60, 35), (60, 60), (30, 60), (30, 35)]])
        obstacles.add_obstacles([[(10, 20), (20, 20), (20, 25), (10, 25), (10, 20)]])
        base_map1 = base_map(self.init_parameters.get_x_lim(), self.init_parameters.get_y_lim(), 10)
        base_map1.set_obstacles(obstacles.get_obstacles())
        base_map1.set_goal_point([self.init_parameters.get_init_target_point()])
        self.basemap = base_map1

        # 参数设置
        reward_function = Reward(self.init_parameters.get_init_start_point(), self.init_parameters.get_init_target_point())
        self.reward = reward_function
        parameter_graph = Parameter_graph()
        parameter_graph.set_x_xlim(self.init_parameters.get_x_lim())
        parameter_graph.set_y_ylim(self.init_parameters.get_y_lim())
        parameter_graph.set_griding_range(self.init_parameters.get_radius())
        parameter_graph.set_reward_function(reward_function)
        # 第一部，划分环境
        graph = Graph(parameter_graph)
        graph.generate_graph()
        self.graph = graph

        # 第二步，确定移动路径
        apf = Artificial_Potential_Field(self.basemap)
        apf.set_explorer(explorer([1], [1], self.init_parameters.get_init_start_point()))
        path = apf.move_to_goal()
        self.parameter_base_show.set_base_show_path_points(path)  # 将移动路径点传递到传递可视化

        # 第三部，赋予权重
        # 1.给予每个边上AUV要到达的点的坐标，直到这个点超出区域，确定超出区域之前和之后的点，确定该区域边最后的交点。
        path_points = FindIntersections().find_intersections(path, self.init_parameters.get_x_lim(),
                                                             self.init_parameters.get_radius())
        # 目前的路径点没有计算auv的初始点，需要加入
        path_points.insert(0, (self.init_parameters.get_init_start_point()[0], self.init_parameters.get_init_start_point()[1]))
        print(path_points)
        # 为路径可视化FindIntersections().plot_grid_and_segments(path,x_xlim,radius,path_points)
        # 2.赋予权重(目前未考虑一个角到另一个角的情况)
        graph.generate_edge_reward(path_points)

        # 第四步，移动探索
        # 0.判断AUV区域
        AUV_point = path_points.pop(0)
        next_point = path_points[0]
        self.path_points = path_points
        AUV_area = graph.where_areas_AUV_in(AUV_point, next_point)
        # 1.设置环境
        state = State()
        state.set_power(self.init_parameters.get_init_power())
        state.set_max_power(self.init_parameters.get_init_power())
        state.set_time(self.init_parameters.get_init_time())
        state.set_max_time(self.init_parameters.get_init_time())
        state.set_auv_point(self.init_parameters.get_init_start_point())
        state.set_target_point(next_point)
        state.set_area(AUV_area)
        state.set_max_point(self.init_parameters.get_x_lim())
        self.state = state

        #设置移动消耗
        time_consumption = Time_Consumption()
        time_consumption.set_move_time(self.init_parameters.get_move_time())
        time_consumption.set_explore_time(self.init_parameters.get_explore_time())
        self.time_consumption = time_consumption

    #action为[0,1,2,3]需要大于0.8的才进行探索，如果全部小于0.8则探索最大的。
    def choose_edge(self,action,area):
        edge_list = []
        array = np.array(action)
        indices_greater_than_0_8 = np.where(array > 0.8)[0]

        if len(indices_greater_than_0_8) > 0:
            # 如果存在大于0.8的数值，则输出其索引
            action_indices = indices_greater_than_0_8

        else:
            # 如果不存在大于0.8的数值，则找到最大的数值的索引
            max_index = np.argmax(array)
            action_indices = [max_index]

        # 进行相应的行动
        for ex in action_indices:
            if ex==0:
                edge_list.append(area.get_up_edge())
            elif ex==1:
                edge_list.append(area.get_right_edge())
            elif ex==2:
                edge_list.append(area.get_down_edge())
            elif ex==3:
                edge_list.append(area.get_left_edge())
        return edge_list

    def get_explore_angles_and_radius(self,edge_list,explorer):
        angles_and_radius_list = []
        for edge in edge_list:
            if edge.auv_in_edge(self.state.get_auv_point()):
                #探索到了auv所以在的边
                dis_1 = math.sqrt((self.state.get_auv_point()[0]-edge.get_point_0_x())**2+(self.state.get_auv_point()[1]-edge.get_point_0_y())**2)
                dis_2 = math.sqrt((self.state.get_auv_point()[0] - edge.get_point_1_x()) ** 2 + (self.state.get_auv_point()[1] - edge.get_point_1_y()) ** 2)
                reduce_power = self.power_consumption.has_explored_edge(dis_1+dis_2)
                self.state.reduce_power(reduce_power)
                self.reward.get_has_explored_reward(reduce_power)
                continue
            #todo 目前还有一种情况没有考虑到就是如果正好在顶点上怎么办
            angles_and_radius_list.append(explorer.explore_edge(edge,self.state.get_auv_point()))
        #合并探索角度和半径
        angles_and_radius = angles_and_radius_list.pop(0)
        angles_and_radius = dict((key, value) for d in angles_and_radius for key, value in d.items())
        for a_r in angles_and_radius_list:
            a_r_dict = dict((key, value) for d in a_r for key, value in d.items())
            for key,value in a_r_dict.items():
                if key in angles_and_radius:
                    angles_and_radius[key] = max(value,angles_and_radius[key])
                else:
                    angles_and_radius[key] = value
        return angles_and_radius

    def set_state(self,state):
        self.state = state

    def show_the_path(self):
        self.parameter_base_show.set_base_show_path_points(self.path_points)
        self.basemap.base_show(self.parameter_base_show)