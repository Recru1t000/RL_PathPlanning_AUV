import math


class Environment():
    def __init__(self,electric,init_point,goal_point,radius,init_points):
        self.electric = electric
        self.init_point = init_point
        self.goal_point = goal_point
        self.radius = radius
        self.init_points = init_points
        self.init_points_queue = list()
        self.origin_environment = None

    def state(self):
        for point in self.init_points:
            self.init_points_queue.append(point)
        state_queue = list()
        for i in range(self.radius):
            if i<len(self.init_points):
                state_queue.append(self.init_points_queue[i])
        state = State(self.electric, self.init_point,state_queue)

    def reset(self):
        self.origin_environment = Environment(self.electric, self.init_point, self.goal_point, self.radius, self.init_points)
        self.electric = self.origin_environment.get_electric()
        self.init_point = self.origin_environment.get_init_point()
        self.goal_point = self.origin_environment.get_goal_point()
        self.radius = self.origin_environment.get_radius()
        self.init_points_queue = list()

    def step(self):
        next_state = []
        reward = 0
        done = False
        truncated = False

    def get_electric(self):
        return self.electric

    def get_init_point(self):
        return self.init_point

    def get_goal_point(self):
        return self.goal_point

    def get_radius(self):
        return self.radius

    def get_init_points(self):
        return self.init_points


class State():
    def __init__(self,electric,init_point,goal_point,init_points):
        self.electric = electric
        self.init_point = init_point
        self.goal_point = goal_point
        self.distance = 0
        self.init_points = init_points

    def count_distance(self):
        self.distance = math.sqrt((self.init_point[0]-self.goal_point[0])**2+(self.init_point[1]-self.goal_point[1])**2)

    def count_electric(self):
        return self.electric

    def count_init_points(self):
        return self.init_points
    def state_main(self):
        self.count_distance()
        self.count_electric()
        self.count_init_points()