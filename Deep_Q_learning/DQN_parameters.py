import numpy as np

class Init_Parameters():
    def __init__(self):
        self.init_start_point = [4,4]
        self.init_target_point = [80,70]
        self.init_power = 200
        self.init_time = 200
        self.x_xlim = 100
        self.y_ylim = 100
        self.radius = 5
        self.power_consumption_value = 8
        self.move_time = 1#每移动1m消耗的时间
        self.explore_time = 1

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
    def get_power_consumption_value(self):
        return self.power_consumption_value
    def get_move_time(self):
        return self.move_time
    def get_explore_time(self):
        return self.explore_time
class DQN_Parameter:
    def __init__(self,state_dim, hidden_dim, action_dim, learning_rate, gamma,
                 epsilon, target_update, device):
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.target_update = target_update
        self.device = device

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
        features = np.concatenate([[self.power/self.max_power, self.time/self.max_time],area_point, auv_point, target_point])
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