class Parameter_base_show():
    def __init__(self):
        self.base_show_path_points = None

    def set_base_show_path_points(self,path_points):
        self.base_show_path_points = path_points

    def get_base_show_path_points(self):
        if self.base_show_path_points is None:
            print("base_show_path_points未赋值")
        else:
            return self.base_show_path_points

class Parameter_graph():
    def __init__(self):
        self.x_xlim = None
        self.y_ylim = None
        self.griding_range = None
        self.reward_function = None

    def set_x_xlim(self,x_xlim):
        self.x_xlim = x_xlim

    def set_y_ylim(self,y_ylim):
        self.y_ylim = y_ylim

    def set_griding_range(self,griding_range):
        self.griding_range = griding_range

    def set_reward_function(self,reward_function):
        self.reward_function = reward_function


    def get_x_xlim(self):
        return self.x_xlim

    def get_y_ylim(self):
        return self.y_ylim

    def get_griding_range(self):
        return self.griding_range

    def get_reward_function(self):
        return self.reward_function