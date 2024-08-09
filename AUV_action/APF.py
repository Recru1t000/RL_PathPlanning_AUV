import math

import numpy as np

from map.simulation_map.explorer import explorer
from map.simulation_map.base_map import base_map as bmp


# APF先做一个点来实验，等一个点完成后再做其他的点
class Artificial_Potential_Field():
    def __init__(self, base_map, att_force=1.0, rep_force=100.0, rep_range=2.0, step_size=0.5):
        self.explorer = explorer([1, 2, 3, 4, 5, 6, 7, 8], [5, 5, 5, 5, 5, 5, 5, 5], [2, 2])
        self.base_map = base_map
        self.att_force = att_force
        self.rep_force = rep_force
        self.rep_range = rep_range
        self.goal = np.array(self.base_map.get_goal_point()[0])
        self.initial_point = np.array(self.explorer.get_initial_point())
        # self.goals = [np.array(goal) for goal in base_map.set_goal_point()]
        self.step_size = step_size
        self.init_points = []
        self.heng_or_shu = 0

    def set_explorer(self,explorer):
        self.explorer = explorer
        self.initial_point = np.array(self.explorer.get_initial_point())
    def set_initial_point(self,initial_point):
        self.initial_point = initial_point

    def append_init_points(self,initial_point):
        self.init_points.append(initial_point)

    def get_init_points(self):
        return self.init_points

    def attractive_force(self):
        # 下面的注释是多点的想法
        # 遍历算每个点的步数，确定一个最小步数，然后从这个点确定下个点的步数，直到没有点了。然后计算全部步数.
        # 以三个目标点为例子,[0,1],[0,2],[0,3],[1,2],[1,3],[2,3],计算1,4,6；1,5,6;1
        return self.att_force * (self.goal - self.initial_point)

    def repulsive_force(self):
        rep_force = np.zeros_like(self.initial_point, dtype=np.float64)
        obstacles = self.base_map.get_obstacles()
        for obstacle in obstacles:
            up_points = obstacle.get_up_explored_points()
            if len(up_points) != 0 and abs(self.initial_point[1] - obstacle.get_obstacle()[2][1]) <= self.rep_range:
                for up_point in up_points:
                    if up_point[0][0] <= self.initial_point[0] <= up_point[len(up_point) - 1][0]:
                        rep_force += self.line_segment_repulsive_force(self.initial_point, up_point[0],up_point[1])
                        dist = self.initial_point[1] - up_point[0][1]
                        if dist < self.rep_range:
                            rep_force = self.perturbation(rep_force,obstacle.get_obstacle()[0],obstacle.get_obstacle()[1],0)

            bottom_points = obstacle.get_bottom_explored_points()
            if len(bottom_points) != 0 and abs(self.initial_point[1] - obstacle.get_obstacle()[0][1]) <= self.rep_range:
                for bottom_point in bottom_points:
                    if bottom_point[0][0] <= self.initial_point[0] <= bottom_point[len(bottom_point) - 1][0]:
                        rep_force +=self.line_segment_repulsive_force(self.initial_point, bottom_point[0], bottom_point[1])
                        dist = self.initial_point[1] - bottom_point[0][1]
                        if dist < self.rep_range:
                            rep_force = self.perturbation(rep_force,obstacle.get_obstacle()[0],obstacle.get_obstacle()[1],0)
                        '''
                        dist = np.linalg.norm(
                            self.initial_point - np.array([self.initial_point[0], obstacle.get_obstacle()[0][1]]))
                        rep_force += self.rep_force * ((1 / dist - 1 / self.rep_range) / dist ** 2) * (
                                    self.initial_point - np.array(
                                [self.initial_point[0], obstacle.get_obstacle()[0][1]]))
                        '''
            left_points = obstacle.get_left_explored_points()
            if len(left_points) != 0 and abs(self.initial_point[0] - obstacle.get_obstacle()[0][0]) <= self.rep_range:
                for left_point in left_points:
                    if left_point[0][1] <= self.initial_point[1] <= left_point[len(left_point) - 1][1]:
                        rep_force += self.line_segment_repulsive_force(self.initial_point, left_point[0],left_point[1])
                        dist = abs(self.initial_point[0] - left_point[0][0])
                        if dist < self.rep_range:
                            rep_force = self.perturbation(rep_force,obstacle.get_obstacle()[0],obstacle.get_obstacle()[3],1)

            right_points = obstacle.get_right_explored_points()
            if len(right_points) != 0 and abs(self.initial_point[0] - obstacle.get_obstacle()[1][0]) <= self.rep_range:
                for right_point in right_points:
                    if right_point[0][1] <= self.initial_point[1] <= right_point[len(right_point) - 1][1]:
                        rep_force += self.line_segment_repulsive_force(self.initial_point, right_point[0],right_point[1])
                        dist = abs(self.initial_point[0] - right_point[0][0])
                        if dist < self.rep_range:
                            rep_force = self.perturbation(rep_force,obstacle.get_obstacle()[0],obstacle.get_obstacle()[3],1)


        return rep_force
    def bottom_or_up_perturbation(self,obstacle):
        if abs(self.initial_point[0]-obstacle.get_obstacle()[0][1])>0:
            return 1
    def line_segment_repulsive_force(self, robot_position, p1, p2, num_points=2,perturbation_strength=0.1):
        p1 = np.array(p1, dtype=np.float64)
        p2 = np.array(p2, dtype=np.float64)
        robot_position = np.array(robot_position, dtype=np.float64)
        rep_force_total = np.zeros_like(robot_position)
        for i in range(num_points):
            t = num_points
            point_on_segment = p1 + i * (p2 - p1)
            dist = np.linalg.norm(robot_position - point_on_segment)
            if dist < self.rep_range:
                repulsive_component = (1 / dist - 1 / self.rep_range) / dist ** 2
                rep_force_total += self.rep_force * repulsive_component * (robot_position - point_on_segment)

        # 添加随机扰动

        return rep_force_total

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

    def move_to_goal(self):
        AUV_point = self.initial_point
        path = [AUV_point]
        while math.sqrt((AUV_point[0]-self.goal[0])**2+(AUV_point[1]-self.goal[1])**2)>1:
            AUV_point = self.move()
            path.append(AUV_point)
        return path

    def set_heng_or_shu(self,heng_or_shu):
        self.heng_or_shu = heng_or_shu

    def perturbation(self,rep_force,point1,point2,du_or_lr):
        if du_or_lr==0:
            if self.heng_or_shu==2:
                if abs(self.initial_point[0]-point1[0])>abs(self.initial_point[0]-point2[0]):
                    rep_force[0] =rep_force[0]+100
                else:
                    rep_force[0] = rep_force[0]-100
        if du_or_lr==1:
            if self.heng_or_shu == 1:
                if abs(self.initial_point[1]-point1[1])>abs(self.initial_point[1]-point2[1]):
                    rep_force[1] = rep_force[1] + 100
                else:
                    rep_force[1] = rep_force[1] - 100
        return rep_force