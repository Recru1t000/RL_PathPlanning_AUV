import copy
import math

from AUV_action.APF import Artificial_Potential_Field
from map.simulation_map.utility import Point_collision


class Environment():
    def __init__(self, electric, init_point, goal_point, radius, init_points, basemap, base_parameters):
        self.electric = electric
        self.init_point = init_point
        self.goal_point = goal_point
        self.radius = radius
        self.init_points = init_points
        self.init_points_queue = list()
        self.origin_environment = None
        self.basemap = basemap
        self.base_parameters = base_parameters

    def set_init_points(self, init_points):
        self.init_points = init_points

    def state(self):
        # todo
        # 有问题需修改
        for point in self.init_points:
            self.init_points_queue.append(point)
        state_queue = list()
        for i in range(self.radius):
            if i < len(self.init_points):
                state_queue.append(self.init_points_queue[i])

        while len(state_queue)<self.radius:
            state_queue.append(state_queue[len(state_queue)-1])
        state = State(self.get_electric(), self.get_init_point(), self.goal_point, state_queue)
        # state.state_main()
        new_state = state.state_main()
        return new_state

    def reset(self):
        if(self.origin_environment is None):
            self.origin_environment = Environment(self.get_electric(), copy.deepcopy(self.get_init_point()), self.goal_point, self.radius,
                                              copy.deepcopy(self.init_points),self.basemap,self.base_parameters)
        self.set_electric(self.origin_environment.get_electric())
        self.set_init_point(self.origin_environment.get_init_point())
        self.goal_point = self.origin_environment.get_goal_point()
        self.radius = self.origin_environment.get_radius()
        self.basemap = self.origin_environment.get_basemap()
        self.base_parameters = self.origin_environment.get_base_parameters()
        self.init_points_queue = list()
        self.init_points = self.origin_environment.get_init_points()
        self.basemap.base_map_reset()
        reset_state = self.state()
        return reset_state

    def step(self, action):
        if len(self.init_points_queue) == 0:
            self.state()
        sequence_points = list()
        # action返回的是序列里面的第n个点
        print(action)
        action_point = self.init_points_queue[action]  # todo 如果仅剩几个点了怎么办
        for i in range(action):  # todo 目前无法判断返回的action的数值，此处需要修改，需要将action的点也放进去
            sequence_points.append(self.init_points_queue[i])
        # 然后将这个点前面的点传入进行判断
        point_collision = Point_collision(self.get_init_point(), sequence_points, action_point)
        explorer = point_collision.point_angle()
        self.electric_cost(explorer)
        other_points = point_collision.other_points()  # 返回的为相同的list，内容为True或False
        # todo 判断的点知道了还需要进行reward设置。以及路径点的移动。如果是没探索到的路径点则直接跳过即可。以及将explorer带入basemap判断是否探索到障碍。
        # 判断basemap
        explore_obstacle = False
        if self.basemap.set_explorer(explorer):#如果探索到新的障碍点
            self.set_init_points(self.artificial_potential_field())
            explore_obstacle = True
            #return False

        # reward
        reward = self.reward(explorer, self.get_init_point(), sequence_points, other_points, action_point, self.goal_point,
                             explore_obstacle)

        # move
        if explore_obstacle:
            self.init_points_queue = list()
            self.set_init_points(self.artificial_potential_field())
        else:
            move_points = self.AUV_move(action, sequence_points, other_points)
            for move_point in move_points:
                self.basemap.append_init_points(move_point)
            i = action
            while i == 0:
                self.init_points_queue.pop(0)
                i -= 1
            self.set_init_point(action_point)

        next_state = self.state()

        done = self.AUV_done()
        truncated = False

        return next_state, reward, done, truncated

    def reward(self, explorer, init_point, sequence_points, other_points, action_point, goal_point, explore_obstacle):
        r = Reward(self.base_parameters.get_explorer_cost(), self.base_parameters.get_movement_cost(),
                   self.base_parameters.get_distance_cost(), self.base_parameters.get_explore_obstacle_cost())
        r.explore_cost(explorer)
        r.point_reward(other_points)
        if explore_obstacle:
            r.explore_obstacle()
        else:
            r.move_cost(init_point, sequence_points)
            r.distance_reward(init_point, action_point, goal_point)
        return r.get_reward()

    def AUV_move(self, action, sequence_points, other_points):
        result_points = list()
        for i in range(action):  # 此处的action需要判断数值
            if other_points[i]:
                result_points.append(sequence_points[i])
        return result_points

    def artificial_potential_field(self):
        # self.basemap.init_points_reset()
        apf = Artificial_Potential_Field(self.basemap)
        apf.set_initial_point(self.get_init_point())
        for i in range(1000):
            # time.sleep(1)
            a = apf.move()
            if (math.sqrt((a[0] - self.goal_point[0]) ** 2 + (a[1] - 70) ** self.goal_point[1]) <= 1):
                break
            # base_map1.show()
            # print(a)
        return apf.get_init_points()

    def AUV_done(self):
        if (math.sqrt(
                (self.get_init_point()[0] - self.goal_point[0]) ** 2 + (self.get_init_point()[1] - 70) ** self.goal_point[1]) <= 1):
            return True
        else:
            return False

    def env_show(self):
        self.basemap.base_show()

    def electric_cost(self,explorer):
        electric = self.get_electric() - explorer.get_radiues()[0]*0.1
        self.set_electric(electric)

    def get_electric(self):
        return self.electric

    def set_electric(self,electric):
        self.electric = electric

    def get_init_point(self):
        return self.init_point

    def set_init_point(self,init_point):
        self.init_point = init_point

    def get_goal_point(self):
        return self.goal_point

    def get_radius(self):
        return self.radius

    def get_init_points(self):
        return self.init_points

    def get_basemap(self):
        return self.basemap

    def get_base_parameters(self):
        return self.base_parameters

class State():
    def __init__(self, electric, init_point, goal_point, init_points):
        self.electric = electric
        self.init_point = init_point
        self.goal_point = goal_point
        self.distance = 0
        self.init_points = init_points

    def count_distance(self):
        self.distance = math.sqrt(
            (self.init_point[0] - self.goal_point[0]) ** 2 + (self.init_point[1] - self.goal_point[1]) ** 2)

    def count_electric(self):
        return self.electric

    def count_init_points(self):
        return self.init_points

    def get_electric(self):
        return self.electric

    def get_distance(self):
        return self.distance

    def get_init_points(self):
        return self.init_points

    def state_main(self):
        self.count_distance()
        self.count_electric()
        self.count_init_points()
        result = list()
        result.append(self.get_electric())
        result.append(self.get_distance())
        for init_point in self.get_init_points():
            for i in init_point:
                result.append(i)
        return result


class Reward():
    def __init__(self, explorer_cost, movement_cost, distance_cost, explore_obstacle_cost):
        self.explorer_cost = explorer_cost
        self.movement_cost = movement_cost
        self.distance_cost = distance_cost
        self.explore_obstacle_cost = explore_obstacle_cost
        self.reward = 0

    def explore_cost(self, explorer):
        r = explorer.get_radiues()
        for i in r:
            self.reward = self.reward + i * self.explorer_cost

    def move_cost(self, init_point, sequence_points):
        for sequence_point in sequence_points:
            distance = math.sqrt((init_point[0] - sequence_point[0]) ** 2 + (init_point[1] - sequence_point[1]) ** 2)
            self.reward = self.reward + distance * self.movement_cost

    def point_reward(self, other_points):
        for point in other_points:
            if point is True:
                self.reward = self.reward + 1  # 加1后面可以改
            else:
                self.reward = self.reward - 1  # 减1后面可以改

    def distance_reward(self, init_point, action_point, goal_point):
        init_x = init_point[0]
        init_y = init_point[1]
        action_x = action_point[0]
        action_y = action_point[1]
        goal_x = goal_point[0]
        goal_y = goal_point[1]
        init_distance = math.sqrt((init_x - goal_x) ** 2 + (init_y - goal_y) ** 2)
        action_distance = math.sqrt((action_x - goal_x) ** 2 + (action_y - goal_y) ** 2)
        self.reward = self.reward + (action_distance - init_distance) * self.distance_cost

    def explore_obstacle(self):
        self.reward = self.reward + self.explore_obstacle_cost

    def get_reward(self):
        return self.reward


class Base_Parameters():
    def __init__(self, explorer_cost, movement_cost, distance_cost, explore_obstacle_cost):
        self.explorer_cost = explorer_cost
        self.movement_cost = movement_cost
        self.distance_cost = distance_cost
        self.explore_obstacle_cost = explore_obstacle_cost

    def get_explorer_cost(self):
        return self.explorer_cost

    def get_movement_cost(self):
        return self.movement_cost

    def get_distance_cost(self):
        return self.distance_cost

    def get_explore_obstacle_cost(self):
        return self.explore_obstacle_cost
