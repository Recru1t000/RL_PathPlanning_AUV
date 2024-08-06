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
        self.power_consumption = Power_Consumption(init_parameters.get_explore_radius(),init_parameters.get_power_consumption_value()/8)
        self.basemap = None
        self.init_parameters = init_parameters
        self.parameter_base_show = Parameter_base_show()
        self.path_points = None
        self.reward = None
        self.time_consumption = None
        self.explored_ob = False


    def step(self,action):
        #action = 1
        #print(action)
        self.reward.reset_reward()
        #确定探索的方向和半径
        new_explorer = DQN_Explorer(self.graph.get_griding_range())
        areas = self.graph.get_areas()
        area = areas.where_area_auv_in(self.state.get_auv_point(),self.state.get_target_point())
        self.state.set_area(area)
        edge_list = self.choose_edge_change_action_to_single(action,area)
        explore_angles_and_radius,edge_list_radius_and_angle = self.get_explore_angles_and_radius(edge_list,new_explorer)

        #执行能量消耗
        reduce_power = self.power_consumption.explorer_power_consumption(explore_angles_and_radius)
        edges_reduce_power = self.power_consumption.edge_explorer_power_consumption_full(edge_list_radius_and_angle)
        self.state.reduce_power(reduce_power)#能量消耗的reward

        #执行探索时间消耗
        reduce_time = self.time_consumption.get_explore_time()
        self.state.reduce_time(reduce_time)

        #执行能量消耗，时间消耗，边探索的reward
        #self.reward.get_edge_reward(edge_list,reduce_power,self.state.get_auv_point(),self.state.get_target_point(),self.power_consumption.get_power_consumption_full_sensing())
        self.reward.new_get_edge_reward(edges_reduce_power,self.explored_ob,self.init_parameters.get_start_explored_power(),(1-self.state.get_power()/self.state.get_max_power()))
        self.reward.get_power_consumption_reward(reduce_power,self.state.get_power(),self.state.get_max_power())
        self.reward.start_explored(self.init_parameters.get_start_explored_power())
        #self.reward.get_time_consumption_reward(reduce_time)

        #执行探索
        explore_obstacle = False
        if not self.explored_ob:
            for key,value in explore_angles_and_radius.items():
                old_explorer = explorer([new_explorer.old_explorer_to_new(key)], [value], self.state.get_auv_point())
                if self.basemap.set_explorer(old_explorer):
                    explore_obstacle = True
        else:
            self.explored_ob = False
        #todo 如果没有探索到必要边怎么办
        has_explored_high_edge = False
        for edge in edge_list:
            if edge.get_reward()==2:
                edge.set_reward(0)
                has_explored_high_edge = True
        #可视化
        #self.show_the_path()
        #如果探索到障碍,扣除reward
        #todo 探测障碍逻辑有问题
        if explore_obstacle and not self.explored_ob:
            #重新规划路线
            apf = Artificial_Potential_Field(self.basemap)
            apf.set_explorer(explorer([1], [1], self.state.get_auv_point()))
            path = apf.move_to_goal()
            #重新赋予权重
            path_points = FindIntersections().find_intersections(path, self.init_parameters.get_x_lim(),self.init_parameters.get_radius())#不需要插入第一个点，因为当前AUV必然在边上
            path_points.append((self.init_parameters.get_init_target_point()[0],self.init_parameters.get_init_target_point()[1]))
            self.graph.generate_edge_reward(path_points)
            path_points.pop(0)
            self.path_points = path_points
            self.state.set_target_point(path_points[0])
            self.explored_ob = True
            #扣除reward
            return self.state.next_state(),self.reward.get_reward(),False,False,False
        #如果没有探索到下一个点的边，扣除reward重新行动。
        elif not has_explored_high_edge:
            return self.state.next_state(), self.reward.get_reward(), False, False, False
        elif has_explored_high_edge:
            if len(self.path_points)!=0:
                self.state.set_auv_point(self.path_points.pop(0))
            else:
                self.state.set_auv_point(self.init_parameters.get_init_target_point())

            if len(self.path_points)!=0:
                next_point = self.path_points[0]
            else:
                next_point = self.init_parameters.get_init_target_point()

            self.state.set_target_point(next_point)
            #可视化
            # 执行时间消耗
            reduce_time = self.time_consumption.move_time_consumption(self.state.get_auv_point(),next_point)
            self.state.reduce_time(reduce_time)
            #执行时间reward
            #self.reward.get_time_consumption_reward(reduce_time)
            #判断是否到达终点
            if self.state.get_auv_point()==(self.init_parameters.get_init_target_point()[0],self.init_parameters.get_init_target_point()[1]):
                done = True
                self.reward.move_to_end_point(self.state.get_power(),self.state.get_power()/self.state.get_max_power())
                print("power:"+str(self.state.get_power()))
                #self.show_the_path()
            else:
                done = False

            #输出查看位置
            self.init_parameters.print_name(self.state.get_auv_point())
            self.init_parameters.print_name(self.state.get_target_point())

            return self.state.next_state(),self.reward.get_reward(),done,False,False

    def reset(self):
        obstacles = Obstacles()  # 障碍应该按照左下，右下，右上，左上，左下的方式添加
        #obstacles.add_obstacles([[(10, 3), (10, 5), (12, 5), (12, 3), (10, 3)]])
        #obstacles.add_obstacles([[(31, 36), (41, 36), (41, 42), (31, 42), (31, 36)]])
        obstacles.add_obstacles([[(11, 11), (12, 11), (12, 13), (11, 13), (11, 11)]])
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
        path_points.append((self.init_parameters.get_init_target_point()[0], self.init_parameters.get_init_target_point()[1]))
        #print(path_points)
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

        return self.state.next_state()

    #action为[0,1,2,3]需要大于0.8的才进行探索，如果全部小于0.8则探索最大的。
    def choose_edge(self,action,area):
        edge_list = []
        array = np.array(action)
        action_indices = list()
        '''
        for i in range(4):
            if array[i]>0.8:
                action_indices.append(i)
        '''
        max_value = 0
        max_index = 0
        if len(action_indices)==0:
            for i in range(4):
                if array[i]>max_value:
                    max_value = array[i]
                    max_index = i
            action_indices.append(max_index)
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

    def choose_edge_change_action_to_single(self,action, area):
        edge_list = []
        if action-8>=0:
            edge_list.append(area.get_left_edge())
            action = action-8
        if action-4>=0:
            edge_list.append(area.get_down_edge())
            action = action-4
        if action-2>=0:
            edge_list.append(area.get_right_edge())
            action = action-2
        if action-1>=0:
            edge_list.append(area.get_up_edge())
            action = action-1
        return edge_list
    def get_explore_angles_and_radius(self,edge_list,explorer):
        edge_list_radius_and_angle = []#用来判断具体每条边消耗的能量
        angles_and_radius_list = []
        for edge in edge_list:
            if edge.auv_in_edge(self.state.get_auv_point()):
                #探索到了auv所以在的边
                dis_1 = math.sqrt((self.state.get_auv_point()[0]-edge.get_point_0_x())**2+(self.state.get_auv_point()[1]-edge.get_point_0_y())**2)
                dis_2 = math.sqrt((self.state.get_auv_point()[0] - edge.get_point_1_x()) ** 2 + (self.state.get_auv_point()[1] - edge.get_point_1_y()) ** 2)
                if self.state.get_auv_point()[0]==edge.get_point_0_x() and edge.get_point_0_x()==edge.get_point_1_x():
                    angles_and_radius_list.append([{'eight':dis_1},{'four': dis_2}])
                if self.state.get_auv_point()[1] == edge.get_point_0_y() and edge.get_point_0_y() == edge.get_point_1_y():
                    angles_and_radius_list.append([{'two': dis_1},{'six': dis_2}])
                edge_list_radius_and_angle.append([edge, dis_1 + dis_2,2])#用来判断具体每条边消耗的能量
                continue
            #todo 目前还有一种情况没有考虑到就是如果正好在顶点上怎么办
            a_and_g = explorer.explore_edge(edge,self.state.get_auv_point())
            angles_and_radius_list.append(a_and_g)

            # 用来判断具体每条边消耗的能量
            dis = 0
            for i in a_and_g:
                for key,value in i.items():
                    dis+=value
            edge_list_radius_and_angle.append([edge,dis,len(a_and_g)])
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
        return angles_and_radius,edge_list_radius_and_angle

    def set_state(self,state):
        self.state = state

    def show_the_path(self):
        self.parameter_base_show.set_base_show_path_points(self.path_points)
        self.basemap.base_show(self.parameter_base_show)
