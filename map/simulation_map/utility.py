import math

import numpy as np
from matplotlib import pyplot as plt

from map.simulation_map.explorer import explorer


class Collision():
    def __init__(self, x_init, y_init, obstacle, r, angles):
        """
        :param x_init: x的初始点
        :param y_init: y的初始点
        :param obstacle: 所有障碍的集合[(x1,y1),(x2,y1),(x2,y2),(x1,y2),(x1,y1)],默认为具体的点，而不是类
        :param r: 扇形的半径
        :return:
        """
        self.collision_points = []
        self.x_init = x_init
        self.y_init = y_init
        self.obstacle = obstacle
        self.x1 = obstacle.get_obstacle()[0][0]
        self.x2 = obstacle.get_obstacle()[1][0]
        self.y1 = obstacle.get_obstacle()[0][1]
        self.y2 = obstacle.get_obstacle()[2][1]
        self.r = r
        self.angle1 = angles[0]
        self.angle2 = angles[1]
        self.left_points = []
        self.bottom_points = []
        self.right_points = []
        self.up_points = []
        self.circle_up_bottom = True  # 因为arccos的y区间为[0,π],所以下半圆的需要360减θ
        if angles[0] >= 180:
            self.circle_up_bottom = False

    def left_side(self):
        if abs(self.x1 - self.x_init) > self.r:
            # print("左边不相交")
            return
        y_chazhi = math.sqrt(self.r ** 2 - (self.x1 - self.x_init) ** 2)  # 算出y的距离

        if self.circle_up_bottom:
            y = self.y_init + y_chazhi
            xita = math.acos(
                (self.x1 - self.x_init) / self.r) * 180 / math.pi  # 求出xita的弧度#x也就正负一说,self.x1 - self.x_init不能加绝对值
        else:
            y = self.y_init - y_chazhi
            xita = 360 - math.acos((self.x1 - self.x_init) / self.r) * 180 / math.pi  # 求出xita的弧度

        if (self.angle1 <= xita <= self.angle2 and self.y1 <= y <= self.y2):
            self.collision_points.append((self.x1, y))
            # self.obstacle.add_left_explored((self.x1,y))
            self.left_points.append((self.x1, y))
            # print("左边相交")
        # else:
        # print("左边不相交")

    def right_side(self):
        if abs(self.x2 - self.x_init) > self.r:
            # print("右边不相交")
            return
        y_chazhi = math.sqrt(self.r ** 2 - (self.x2 - self.x_init) ** 2)  # 算出y的距离
        if self.circle_up_bottom:
            y = self.y_init + y_chazhi
            xita = math.acos((self.x2 - self.x_init) / self.r) * 180 / math.pi  # 求出xita的弧度
        else:
            y = self.y_init - y_chazhi
            xita = 360 - math.acos((self.x2 - self.x_init) / self.r) * 180 / math.pi  # 求出xita的弧度

        if (self.angle1 <= xita <= self.angle2 and self.y1 <= y <= self.y2):
            self.collision_points.append((self.x2, y))
            # self.obstacle.add_right_explored((self.x2,y))
            self.right_points.append((self.x2, y))
            # print("右边相交")
        # else:
        # print("右边不相交")

    def up_side(self):
        if abs(self.y2 - self.y_init) > self.r:
            # print("上边不相交")
            return
        x_chazhi = math.sqrt(self.r ** 2 - (self.y2 - self.y_init) ** 2)  # 算出y的距离
        if self.circle_up_bottom:
            if 0 <= self.angle1 < 90:
                x = self.x_init + x_chazhi
                xita = math.asin((self.y2 - self.y_init) / self.r) * 180 / math.pi  # 求出xita的弧度
            else:
                x = self.x_init - x_chazhi
                xita = 180 - math.asin((self.y2 - self.y_init) / self.r) * 180 / math.pi  # 求出xita的弧度
        else:
            if 270 <= self.angle1 < 360:
                x = self.x_init + x_chazhi
                xita = 360 + math.asin((self.y2 - self.y_init) / self.r) * 180 / math.pi  # 求出xita的弧度
            else:
                x = self.x_init - x_chazhi
                xita = 180 - math.asin((self.y2 - self.y_init) / self.r) * 180 / math.pi  # 求出xita的弧度
                # print(math.asin((self.y2 - self.y_init) / self.r)*180/math.pi)
        if (self.angle1 <= xita <= self.angle2 and self.x1 <= x <= self.x2):
            self.collision_points.append((x, self.y2))
            # self.obstacle.add_up_explored((x,self.y2))
            self.up_points.append((x, self.y2))
            # print("上边相交")
        # else:
        # print("上边不相交")

    def bottom_side(self):
        if abs(self.y1 - self.y_init) > self.r:
            # print("下边不相交")
            return
        x_chazhi = math.sqrt(self.r ** 2 - (self.y1 - self.y_init) ** 2)  # 算出y的距离
        if self.circle_up_bottom:
            if 0 <= self.angle1 < 90:
                x = self.x_init + x_chazhi
                xita = math.asin((self.y1 - self.y_init) / self.r) * 180 / math.pi  # 求出xita的弧度
            else:
                x = self.x_init - x_chazhi
                xita = 180 - math.asin((self.y1 - self.y_init) / self.r) * 180 / math.pi  # 求出xita的弧度
        else:
            if 270 <= self.angle1 < 360:
                x = self.x_init + x_chazhi
                xita = 360 + math.asin((self.y1 - self.y_init) / self.r) * 180 / math.pi  # 求出xita的弧度
            else:
                x = self.x_init - x_chazhi
                xita = 180 - math.asin((self.y1 - self.y_init) / self.r) * 180 / math.pi  # 求出xita的弧度
                # print(math.asin((self.y1 - self.y_init) / self.r)*180/math.pi)
        if (self.angle1 <= xita <= self.angle2 and self.x1 <= x <= self.x2):
            self.collision_points.append((x, self.y1))
            # self.obstacle.add_bottom_explored((x,self.y1))
            self.bottom_points.append((x, self.y1))
            # print("下边相交")
        # else:
        # print("下边不相交")

    def straight_line(self, angel):
        if (angel == 0 or angel == 360 or angel == 180):
            canshu = 1
            if angel == 180:  # 如果为0或360则说明是向右找点，所以该点必然大于初始点，向左则说明，小于所以设定该参数判断
                canshu = -1
                # 扇形两条射线的终点是否在矩形中
                # 上述条件是否可以不判断
                # 我们展示出的图形仅需描述障碍的外观，无法探测障碍的内部，也就是说障碍的边与探测器的交汇就行
            if (
                    abs(self.x1 - self.x_init) <= self.r and canshu * self.x1 >= canshu * self.x_init and self.y1 <= self.y_init <= self.y2):
                self.collision_points.append((self.x1, self.y_init))
                # self.obstacle.add_left_explored((self.x1,self.y_init))
                self.left_points.append((self.x1, self.y_init))
            if (
                    abs(self.x2 - self.x_init) <= self.r and canshu * self.x2 >= canshu * self.x_init and self.y1 <= self.y_init <= self.y2):
                self.collision_points.append((self.x2, self.y_init))
                # self.obstacle.add_right_explored((self.x2,self.y_init))
                self.right_points.append((self.x2, self.y_init))

        if (angel == 45 or angel == 225):  # 如果是45则说明，x和y都大于初始点，如果是225则说明都小于初始点
            a = self.y_init - self.x_init
            canshu = 1
            if angel == 225:
                canshu = -1
            y_left = self.x1 + a
            y_right = self.x2 + a
            x_bottmo = self.y1 - a
            x_up = self.y2 - a
            # 以下需要判断三点，1.求出来的点是否在半径之内，2.该点是否在矩形的边界之内，3.判断是向45还是向225，如果是45则所有点大于init否则小于init
            if (y_left - self.y_init) ** 2 + (
                    self.x1 - self.x_init) ** 2 <= self.r ** 2 and self.y1 <= y_left <= self.y2 and canshu * self.x1 >= canshu * self.x_init:  # 不用小于等于是因为弧线上的节点已经在上面被包含了
                self.collision_points.append((self.x1, y_left))
                # self.obstacle.add_left_explored((self.x1, y_left))
                self.left_points.append((self.x1, y_left))
            if (y_right - self.y_init) ** 2 + (
                    self.x2 - self.x_init) ** 2 <= self.r ** 2 and self.y1 <= y_right <= self.y2 and canshu * self.x2 >= canshu * self.x_init:
                self.collision_points.append((self.x2, y_right))
                # self.obstacle.add_right_explored((self.x2, y_right))
                self.right_points.append((self.x2, y_right))
            if (x_bottmo - self.x_init) ** 2 + (
                    self.y1 - self.y_init) ** 2 <= self.r ** 2 and self.x1 <= x_bottmo <= self.x2 and canshu * self.y1 >= canshu * self.y_init:
                self.collision_points.append((x_bottmo, self.y1))
                # self.obstacle.add_bottom_explored((x_bottmo, self.y1))
                self.bottom_points.append((x_bottmo, self.y1))
            if (x_up - self.x_init) ** 2 + (
                    self.y2 - self.y_init) ** 2 <= self.r ** 2 and self.x1 <= x_up <= self.x2 and canshu * self.y2 >= canshu * self.y_init:
                self.collision_points.append((x_up, self.y2))
                # self.obstacle.add_up_explored((x_up, self.y2))
                self.up_points.append((x_up, self.y2))

        if (angel == 90 or angel == 270):
            canshu = 1
            if angel == 270:  # 如果为90则说明是向上找点，所以该点必然大于初始点，向下则说明，小于所以设定该参数判断
                canshu = -1
            if (
                    abs(self.y1 - self.y_init) <= self.r and canshu * self.y1 >= canshu * self.y_init and self.x1 <= self.x_init <= self.x2):
                self.collision_points.append((self.x_init, self.y1))
                # self.obstacle.add_bottom_explored((self.x_init,self.y1))
                self.bottom_points.append((self.x_init, self.y1))
            if (
                    abs(self.y2 - self.y_init) <= self.r and canshu * self.y2 >= canshu * self.y_init and self.x1 <= self.x_init <= self.x2):
                self.collision_points.append((self.x_init, self.y2))
                # self.obstacle.add_up_explored((self.x_init,self.y2))
                self.up_points.append((self.x_init, self.y2))

        if (angel == 135 or angel == 315):  # 如果是135则说明，x小于初始点,y大于初始点，如果是315则说明,x大于初始点,y小于初始点
            a = self.y_init + self.x_init
            canshu = 1
            if angel == 315:
                canshu = -1
            y_left = -self.x1 + a
            y_right = -self.x2 + a
            x_bottmo = -self.y1 + a
            x_up = -self.y2 + a
            # 以下需要判断三点，1.求出来的点是否在半径之内，2.该点是否在矩形的边界之内，3.判断是向135还是向315
            if (y_left - self.y_init) ** 2 + (
                    self.x1 - self.x_init) ** 2 <= self.r ** 2 and self.y1 <= y_left <= self.y2 and canshu * self.x1 <= canshu * self.x_init:  # 不用小于等于是因为弧线上的节点已经在上面被包含了
                self.collision_points.append((self.x1, y_left))
                # self.obstacle.add_left_explored((self.x1, y_left))
                self.left_points.append((self.x1, y_left))
            if (y_right - self.y_init) ** 2 + (
                    self.x2 - self.x_init) ** 2 <= self.r ** 2 and self.y1 <= y_right <= self.y2 and canshu * self.x2 <= canshu * self.x_init:
                self.collision_points.append((self.x2, y_right))
                # self.obstacle.add_right_explored((self.x2, y_right))
                self.right_points.append((self.x2, y_right))
            if (x_bottmo - self.x_init) ** 2 + (
                    self.y1 - self.y_init) ** 2 <= self.r ** 2 and self.x1 <= x_bottmo <= self.x2 and canshu * self.y1 >= canshu * self.y_init:
                self.collision_points.append((x_bottmo, self.y1))
                # self.obstacle.add_bottom_explored((x_bottmo, self.y1))
                self.bottom_points.append((x_bottmo, self.y1))
            if (x_up - self.x_init) ** 2 + (
                    self.y2 - self.y_init) ** 2 <= self.r ** 2 and self.x1 <= x_up <= self.x2 and canshu * self.y2 >= canshu * self.y_init:
                self.collision_points.append((x_up, self.y2))
                # self.obstacle.add_up_explored((x_up, self.y2))
                self.up_points.append((x_up, self.y2))

    # todo 两条线，和线的终点是否在障碍中

    def rectangle_point(self, obstacle):
        # 分不同的象限来做
        x1 = obstacle[0]
        y1 = obstacle[1]

        r = math.sqrt((self.x_init - x1) ** 2 + (self.y_init - y1) ** 2)
        x2 = x1 + r
        O = np.array([self.x_init, self.y_init])
        A = np.array([x1, y1])
        B = np.array([x2, self.y_init])

        # 计算向量 OA 和 OB
        OA = A - O
        OB = B - O
        denominator = np.linalg.norm(OA) * np.linalg.norm(OB)
        if denominator == 0:
            # print("分母为零，夹角无法计算。")
            return

        # 计算夹角的余弦值
        cos_theta = np.dot(OA, OB) / denominator

        # 计算弧度和角度
        theta_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))
        theta_degree = np.degrees(theta_rad)
        if y1 < self.y_init:
            theta_degree = 360 - theta_degree
        if r ** 2 <= self.r ** 2 and self.angle1 <= theta_degree <= self.angle2:
            self.collision_points.append((x1, y1))
            if x1 == self.x1 and y1 == self.y1:
                # self.obstacle.add_left_explored((x1,y1))
                self.left_points.append((x1, y1))
                # self.obstacle.add_bottom_explored((x1,y1))
                self.bottom_points.append((x1, y1))
            elif x1 == self.x1 and y1 == self.y2:
                # self.obstacle.add_left_explored((x1, y1))
                self.left_points.append((x1, y1))
                # self.obstacle.add_up_explored((x1, y1))
                self.up_points.append((x1, y1))
            elif x1 == self.x2 and y1 == self.y1:
                # self.obstacle.add_right_explored((x1,y1))
                self.right_points.append((x1, y1))
                # self.obstacle.add_bottom_explored((x1, y1))
                self.bottom_points.append((x1, y1))
            elif x1 == self.x2 and y1 == self.y2:
                # self.obstacle.add_right_explored((x1,y1))
                self.right_points.append((x1, y1))
                # self.obstacle.add_up_explored((x1, y1))
                self.up_points.append((x1, y1))
        # print(theta_degree)

    def get_collision_points(self):
        self.left_side()
        self.right_side()
        self.up_side()
        self.bottom_side()
        self.straight_line(self.angle1)
        self.straight_line(self.angle2)
        for obstacle in self.obstacle.get_obstacle():
            self.rectangle_point(obstacle)
        self.collision_points = list(set(self.collision_points))

        if len(self.collision_points) != 0:
            self.collision_points.append(self.collision_points[0])
        # else:
        # print("无碰撞点")
        return self.collision_points

class Point_collision():
    def __init__(self, init_point, points, action_point):
        self.init_point = init_point
        self.explorer = None
        self.action_point = action_point
        self.sequence_points = points#如果为空怎么办
        self.angle = None
        self.distance_between_points = None

    def point_angle(self):
        x1 = self.init_point[0]
        y1 = self.init_point[1]
        x2 = self.action_point[0]
        y2 = self.action_point[1]

        # 计算两个点之间的距离
        distance_between_points = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        self.distance_between_points = distance_between_points

        # 计算两个点之间的夹角（单位为弧度）
        angle_in_radians = math.atan2(y2 - y1, x2 - x1)

        # 将夹角转换为角度制
        angle_in_degrees = angle_in_radians * 180 / math.pi
        if angle_in_degrees < 0:
            angle_in_degrees = 360 + angle_in_degrees

        angle = self.judge_angel(angle_in_degrees)  # 判断对应探索器的角度
        self.angle = angle
        self.explorer = explorer([angle], [distance_between_points], self.init_point)

        return self.explorer

    def other_points(self):
        x1 = self.init_point[0]
        y1 = self.init_point[1]
        other_points_explored = list()
        for sequence_point in self.sequence_points:
            x2 = sequence_point[0]
            y2 = sequence_point[1]

            # 计算两个点之间的距离
            distance_between_points = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

            # 计算两个点之间的夹角（单位为弧度）
            angle_in_radians = math.atan2(y2 - y1, x2 - x1)

            # 将夹角转换为角度制
            angle_in_degrees = angle_in_radians * 180 / math.pi
            if angle_in_degrees < 0:
                angle_in_degrees = 360 + angle_in_degrees

            angle = self.judge_angel(angle_in_degrees)  # 判断对应探索器的角度

            if self.angle is not None and self.distance_between_points is not None:
                if angle == self.angle and distance_between_points == self.distance_between_points:
                    other_points_explored.append(True)
                else:
                    other_points_explored.append(False)
        return other_points_explored


    def judge_angel(self, angle):
        if 135 <= angle < 180:
            return 1
        elif 90 <= angle < 135:
            return 2
        elif 45 <= angle < 90:
            return 3
        elif 0 <= angle < 45:
            return 4
        elif 315 <= angle < 360:
            return 5
        elif 270 <= angle < 315:
            return 6
        elif 225 <= angle < 270:
            return 7
        elif 180 <= angle < 225:
            return 8


class FindIntersections():
    def find_intersections(self, points, grid_size, cell_size):
        def line_intersection(p1, p2, q1, q2):
            def on_segment(p, q, r):
                if min(p[0], q[0]) <= r[0] <= max(p[0], q[0]) and min(p[1], q[1]) <= r[1] <= max(p[1], q[1]):
                    return True
                return False

            def orientation(p, q, r):
                val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
                if val == 0:
                    return 0
                return 1 if val > 0 else 2

            o1 = orientation(p1, p2, q1)
            o2 = orientation(p1, p2, q2)
            o3 = orientation(q1, q2, p1)
            o4 = orientation(q1, q2, p2)

            if o1 != o2 and o3 != o4:
                denom = (p2[0] - p1[0]) * (q2[1] - q1[1]) - (p2[1] - p1[1]) * (q2[0] - q1[0])
                if denom == 0:
                    return None
                num1 = (p1[1] - q1[1]) * (q2[0] - q1[0]) - (p1[0] - q1[0]) * (q2[1] - q1[1])
                num2 = (p1[1] - q1[1]) * (p2[0] - p1[0]) - (p1[0] - q1[0]) * (p2[1] - p1[1])
                t1 = num1 / denom
                t2 = num2 / denom
                if 0 <= t1 <= 1 and 0 <= t2 <= 1:
                    return (p1[0] + t1 * (p2[0] - p1[0]), p1[1] + t1 * (p2[1] - p1[1]))
            return None

        intersections = list()  # 使用集合避免重复

        for i in range(len(points) - 1):
            p1, p2 = points[i], points[i + 1]

            x_min, x_max = min(p1[0], p2[0]), max(p1[0], p2[0])
            y_min, y_max = min(p1[1], p2[1]), max(p1[1], p2[1])

            for x in range(0, grid_size + 1, cell_size):
                for y in range(0, grid_size + 1, cell_size):
                    grid_lines = [
                        ((x, y), (x + cell_size, y)),
                        ((x, y), (x, y + cell_size)),
                        ((x + cell_size, y), (x + cell_size, y + cell_size)),
                        ((x, y + cell_size), (x + cell_size, y + cell_size))
                    ]

                    for q1, q2 in grid_lines:
                        intersection = line_intersection(p1, p2, q1, q2)
                        if intersection and x_min <= intersection[0] <= x_max and y_min <= intersection[1] <= y_max:
                            if intersection not in intersections:
                                intersections.append(intersection)

        return list(intersections)

    def plot_grid_and_segments(self,points, grid_size, cell_size, intersections):
        fig, ax = plt.subplots()

        # 绘制网格线段
        for x in range(0, grid_size + 1, cell_size):
            ax.plot([x, x], [0, grid_size], color='lightgrey', linestyle='--')
        for y in range(0, grid_size + 1, cell_size):
            ax.plot([0, grid_size], [y, y], color='lightgrey', linestyle='--')

        # 绘制线段
        for i in range(len(points) - 1):
            p1, p2 = points[i], points[i + 1]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='blue')

        # 绘制交点
        for inter in intersections:
            ax.plot(inter[0], inter[1], 'ro')  # 交点标记为红色

        ax.set_xlim(0, grid_size)
        ax.set_ylim(0, grid_size)
        ax.set_aspect('equal')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Line Segments and Grid Intersections')
        plt.grid(True)
        plt.show()