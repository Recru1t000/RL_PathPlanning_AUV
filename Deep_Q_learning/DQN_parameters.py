import math

import numpy as np

class Init_Parameters():
    def __init__(self):
        self.init_start_point = [41,9]
        self.init_target_point = [41,80]
        self.x_xlim = 100
        self.y_ylim = 100
        self.radius = 5
        self.explore_radius = 7.07
        self.power_consumption_value = 8
        #每移动1m消耗的时间
        self.move_time = 1
        #每次探索的时间
        self.explore_time = 1
        #每次启动探索器所需要消耗的固定能量
        self.start_explored_power = 1
        #设置的最大能量为1.5倍的全探索的能量乘以一条或者一列的格子数
        self.init_power = self.power_consumption_value*math.sqrt((self.init_start_point[0]-self.init_target_point[0])**2+(self.init_start_point[1]-self.init_target_point[1])**2)/self.explore_radius*0.5+math.sqrt((self.init_start_point[0]-self.init_target_point[0])**2+(self.init_start_point[1]-self.init_target_point[1])**2)/self.radius*self.start_explored_power
        #设置初始时间为1.5倍的移动时间乘以起点到终点的距离
        self.init_time = 1.5*self.move_time*math.sqrt((self.init_start_point[0]-self.init_target_point[0])**2+(self.init_start_point[1]-self.init_target_point[1])**2)


        self.print_range = 0
        self.print_max_range = 10

    def get_init_start_point(self):
        return self.init_start_point
    def get_init_target_point(self):
        return self.init_target_point
    def get_init_power(self):
        return self.init_power
    def get_init_time(self):
        return self.init_time
    def get_x_lim(self):
        return self.x_xlim
    def get_y_lim(self):
        return self.y_ylim
    def get_radius(self):
        return self.radius
    def get_explore_radius(self):
        return self.explore_radius
    def get_power_consumption_value(self):
        return self.power_consumption_value
    def get_move_time(self):
        return self.move_time
    def get_explore_time(self):
        return self.explore_time

    def print_name(self,name):
        if self.print_range>=self.print_max_range:
            print(name)

    def set_print_range(self,num):
        self.print_range = num

    def get_start_explored_power(self):
        return self.start_explored_power

    def set_init_start_point(self,init_start_point):
        self.init_start_point = init_start_point

    def set_init_target_point(self,init_target_point):
        self.init_target_point = init_target_point
class DQN_Parameter:
    def __init__(self,state_dim, hidden_dim, action_dim, learning_rate, gamma,
                 epsilon, target_update, device,epsilon_min,epsilon_decay,capacity):
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.target_update = target_update
        self.device = device
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.capacity = capacity

    def get_state_dim(self):
        return self.state_dim

    def get_hidden_dim(self):
        return self.hidden_dim

    def get_action_dim(self):
        return self.action_dim

    def get_learning_rate(self):
        return self.learning_rate

    def get_gamma(self):
        return self.gamma

    def get_epsilon(self):
        return self.epsilon

    def get_target_update(self):
        return self.epsilon

    def get_device(self):
        return self.device

    def get_epsilon_decay(self):
        return self.epsilon_decay

    def get_epsilon_min(self):
        return self.epsilon_min

    def get_capacity(self):
        return self.capacity
class State:
    def __init__(self):
        self.power = None
        self.time = None
        self.area = None
        self.auv_point = None
        self.target_point = None


        self.max_power = None
        self.max_time = None
        self.max_point = None

    def turn_to_features(self):
        left_up_point = np.array(self.area.get_left_up_point())/self.max_point
        right_up_point = np.array(self.area.get_right_up_point())/self.max_point
        right_down_point = np.array(self.area.get_right_down_point())/self.max_point
        left_down_point = np.array(self.area.get_left_down_point())/self.max_point
        area_point = np.concatenate([left_up_point, right_up_point,right_down_point,left_down_point])#归一化处理
        auv_point = np.array(self.auv_point)/self.max_point
        target_point = np.array(self.target_point)/self.max_point
        features = np.concatenate([[self.power/self.max_power],area_point, auv_point, target_point])
        return features

    def next_state(self):
        next_state = self.turn_to_features()
        return next_state

    def reduce_power(self,value):
        self.power = self.power - value

    def reduce_time(self,reduce_time):
        self.time = self.time - reduce_time
    def set_max_power(self, power):
        self.max_power = power

    def set_max_time(self, time):
        self.max_time = time

    def set_power(self, power):
        self.power = power

    def set_time(self, time):
        self.time = time

    def set_area(self, area):
        self.area = area

    def set_auv_point(self, auv_point):
        self.auv_point = auv_point

    def set_target_point(self, target_point):
        self.target_point = target_point

    def set_max_point(self, max_point):
        self.max_point = max_point

    def get_power(self):
        return self.power

    def get_time(self):
        return self.time

    def get_area(self):
        return self.area

    def get_auv_point(self):
        return self.auv_point

    def get_target_point(self):
        return self.target_point

    def get_max_power(self):
        return self.max_power

    def get_max_time(self):
        return self.max_time

    def get_max_point(self):
        return self.max_point