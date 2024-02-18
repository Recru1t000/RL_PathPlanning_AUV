import math

from map.simulation_map.utility import Point_collision


class Environment():
    def __init__(self,electric,init_point,goal_point,radius,init_points,basemap):
        self.electric = electric
        self.init_point = init_point
        self.goal_point = goal_point
        self.radius = radius
        self.init_points = init_points
        self.init_points_queue = list()
        self.origin_environment = None
        self.basemap = basemap

    def state(self):
        #todo
        # 有问题需修改
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
        self.basemap.base_map_reset()

    def step(self,action):
        sequence_points = list()
        #action返回的是序列里面的第n个点
        action_point = self.init_points_queue[action]
        for i in range(action):#todo 目前无法判断返回的action的数值，此处需要修改
            sequence_points.append(self.init_points_queue.pop(0))
        #然后将这个点前面的点传入进行判断
        point_collision = Point_collision(self.init_point,sequence_points,action_point)
        explorer = point_collision.point_angle()
        other_points = point_collision.other_points()#返回的为相同的list，内容为True或False
        #todo 判断的点知道了还需要进行reward设置。以及路径点的移动。如果是没探索到的路径点则直接跳过即可。以及将explorer带入basemap判断是否探索到障碍。
        reward = self.reward()
        i = action
        while i==0:
            self.init_points_queue.pop(0)
            i-=1
        next_state = []

        done = False
        truncated = False

    def reward(self):


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

class Reward():
    def __init__(self,explorer_cost,movement_cost,sequence_points):
        self.explorer_cost = explorer_cost
        self.movement_cost = movement_cost
        self.sequence_points = sequence_points
        self.reward = 0

    def explore_cost(self,explorer):
        r = explorer.get_radiues()
        self.reward  = self.reward + r*self.explorer_cost

    def move_cost(self,init_point):
        for sequence_point in self.sequence_points:
            distance = math.sqrt(init_point[0]-sequence_point[0])**2+(init_point[1]-sequence_point[1])**2
            self.reward = self.reward+distance*self.movement_cost