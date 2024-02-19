import numpy as np

from map.simulation_map.explorer import explorer
from map.simulation_map.base_map import base_map as bmp


# APF先做一个点来实验，等一个点完成后再做其他的点
class Artificial_Potential_Field():
    def __init__(self, base_map, att_force=1.0, rep_force=100.0, rep_range=2.0, step_size=1.0):
        self.explorer = explorer([1, 2, 3, 4, 5, 6, 7, 8], [5, 5, 5, 5, 5, 5, 5, 5], [10, 10])
        self.base_map = base_map
        self.att_force = att_force
        self.rep_force = rep_force
        self.rep_range = rep_range
        self.goal = np.array(self.base_map.get_goal_point()[0])
        self.initial_point = np.array(self.explorer.get_initial_point())
        # self.goals = [np.array(goal) for goal in base_map.set_goal_point()]
        self.step_size = step_size
        self.init_points = []

    def set_initial_point(self,initial_point):
        self.initial_point = initial_point

    def append_init_points(self,initial_point):
        self.init_points.append(initial_point)

    def get_init_points(self):
        return self.init_points

    def get_init_points(self):
        return self.init_points

    def attractive_force(self):
        # 下面的注释是多点的想法
        # 遍历算每个点的步数，确定一个最小步数，然后从这个点确定下个点的步数，直到没有点了。然后计算全部步数.
        # 以三个目标点为例子,[0,1],[0,2],[0,3],[1,2],[1,3],[2,3],计算1,4,6；1,5,6;1
        return self.att_force * (self.goal - self.initial_point)

    def repulsive_force(self):
        rep_force = np.zeros_like(self.initial_point)
        obstacles = self.base_map.get_obstacles()
        for obstacle in obstacles:
            up_points = obstacle.get_up_explored_points()
            if len(up_points) != 0 and abs(self.initial_point[1] - obstacle.get_obstacle()[2][1]) <= self.rep_range:
                for up_point in up_points:
                    if up_point[0][0] <= self.initial_point[0] <= up_point[len(up_point) - 1][0]:
                        dist = np.linalg.norm(
                            self.initial_point - np.array([self.initial_point[0], obstacle.get_obstacle()[2][1]]))
                        rep_force += self.rep_force * ((1 / dist - 1 / self.rep_range) / dist ** 2) * (
                                    self.initial_point - np.array(
                                [self.initial_point[0], obstacle.get_obstacle()[2][1]]))

            bottom_points = obstacle.get_bottom_explored_points()
            if len(bottom_points) != 0 and abs(self.initial_point[1] - obstacle.get_obstacle()[0][1]) <= self.rep_range:
                for bottom_point in bottom_points:
                    if bottom_point[0][0] <= self.initial_point[0] <= bottom_point[len(bottom_point) - 1][0]:
                        dist = np.linalg.norm(
                            self.initial_point - np.array([self.initial_point[0], obstacle.get_obstacle()[0][1]]))
                        rep_force += self.rep_force * ((1 / dist - 1 / self.rep_range) / dist ** 2) * (
                                    self.initial_point - np.array(
                                [self.initial_point[0], obstacle.get_obstacle()[0][1]]))
            left_points = obstacle.get_left_explored_points()
            if len(left_points) != 0 and abs(self.initial_point[0] - obstacle.get_obstacle()[0][0]) <= self.rep_range:
                for left_point in left_points:
                    if left_point[0][1] <= self.initial_point[1] <= left_point[len(left_point) - 1][1]:
                        dist = np.linalg.norm(
                            self.initial_point - np.array([obstacle.get_obstacle()[0][0], self.initial_point[0]]))
                        rep_force += self.rep_force * ((1 / dist - 1 / self.rep_range) / dist ** 2) * (
                                    self.initial_point - np.array(
                                [obstacle.get_obstacle()[0][0], self.initial_point[0]]))
            right_points = obstacle.get_right_explored_points()
            if len(right_points) != 0 and abs(self.initial_point[0] - obstacle.get_obstacle()[1][0]) <= self.rep_range:
                for right_point in right_points:
                    if right_point[0][1] <= self.initial_point[1] <= right_point[len(right_point) - 1][1]:
                        dist = np.linalg.norm(
                            self.initial_point - np.array([obstacle.get_obstacle()[2][0], self.initial_point[0]]))
                        rep_force += self.rep_force * ((1 / dist - 1 / self.rep_range) / dist ** 2) * (
                                    self.initial_point - np.array(
                                [obstacle.get_obstacle()[2][0], self.initial_point[0]]))
        return rep_force

    def calculate_total_force(self):
        att_force = self.attractive_force()
        rep_force = self.repulsive_force()
        total_force = att_force + rep_force
        return total_force

    def take_explorer(self):
        self.base_map.set_explorer(explorer([1, 2, 3, 4, 5, 6, 7, 8], [5, 5, 5, 5, 5, 5, 5, 5], self.initial_point))

    def move(self):
        #self.take_explorer()
        force = self.calculate_total_force()
        new_position = self.initial_point + self.step_size * force / np.linalg.norm(force)
        self.explorer.set_initial_point(new_position)
        self.initial_point = new_position
        if (self.goal[0]-self.initial_point[0])**2+(self.goal[1]-self.initial_point[1])**2 <= self.step_size**2:
            self.initial_point = self.goal
        #self.base_map.append_init_points(new_position)
        self.append_init_points(new_position)
        return new_position
