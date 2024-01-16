import math
from math import cos,sin,acos,asin

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Wedge
from map.simulation_map.utility import Collision

class base_map():
    def __init__(self,x_xlim,y_ylim,figure_value,explorer):
        self.xlim = x_xlim#x轴
        self.ylim = y_ylim#y轴
        self.figure_value = figure_value #默认为15
        self.x_datas = []#x数据点
        self.y_datas = []#y数据点
        self.obstacles = []
        self.explorer = explorer

    def random_datas(self):
        # 生成一些示例数据
        x_datas = []
        y_datas = []
        i = 0
        j = 0
        while i<10:
            i = i+0.0001
            self.x_datas.append(i)
            j = i**2+3
            self.y_datas.append(j)

    def set_datas(self,x_datas,y_datas):
        self.x_datas = x_datas
        self.y_datas = y_datas

    def set_obstacles(self,obstacles):
        self.obstacles = obstacles

    def collision(self):
        angles = self.explorer.get_angle(self.explorer.angles)
        for obstacle in self.obstacles:
            remove_repeat_obstacle_left_point = False
            for i in range(len(angles)):
                c = Collision(self.explorer.initial_point[0], self.explorer.initial_point[1], obstacle,
                              self.explorer.radiues[i], angles[i]).get_collision_points()
                #print(c)
                if len(c) !=0:
                    remove_repeat_obstacle_left_point = True
                t = obstacle
                print(t)
                #提取每个障碍的所有点，结合到一起按照下，右，上，左进行排列，最后加上最左边的点
            if remove_repeat_obstacle_left_point:
                obstacle.arrange_every_side()

        print(self.obstacles)


    def show(self):
        if self.x_datas==[]:
            self.random_datas()

        fig, ax = plt.subplots(figsize=(self.figure_value, self.figure_value))
        angles = self.explorer.get_angle(self.explorer.angles)
        # 生成八分之一圆形
        for radius,angle in zip(self.explorer.radiues,angles):
            first_angle = angle[0]
            second_angle = angle[1]
            wedge = Wedge((self.explorer.initial_point[0], self.explorer.initial_point[1]), radius, first_angle, second_angle, facecolor='blue')
            ax.add_patch(wedge)

        # 使用plot函数绘制连续轨迹
        plt.plot(self.x_datas, self.y_datas, label='Trajectory')

        # 添加黑色多边形
        for obstacle in self.obstacles:
            polygon_vertices = np.array(obstacle.get_obstacle())
            plt.fill(polygon_vertices[:, 0], polygon_vertices[:, 1], color='black', alpha=0.5)#polygon_vertices[:, 0]表示切片所有数组的第一个元素

        #todo 生成形状还是生成线段？？？
        for obstacle in self.obstacles:
            obstacle_points = obstacle.show_point()
            if len(obstacle_points) !=0:
                polygon_vertices = np.array(obstacle_points)
                plt.fill(polygon_vertices[:, 0], polygon_vertices[:, 1], color='red', alpha=0.5)#polygon_vertices[:, 0]表示切片所有数组的第一个元素


        plt.xlim(0, self.xlim)
        plt.ylim(0, self.ylim)

        # 添加标签和标题
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.title('Continuous Trajectory Plot')
        # 添加图例
        plt.legend()
        # 显示图形
        plt.show()

