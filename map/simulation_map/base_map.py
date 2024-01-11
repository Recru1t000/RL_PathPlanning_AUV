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
        c = Collision(self.explorer.initial_point[0],self.explorer.initial_point[1],self.obstacles[1],self.explorer.radiues[0],angles[0])
        print(c.get_collision_points())
        #print(cos(angles[0][0]*math.pi/180))
        x1 = self.explorer.initial_point[0]+self.explorer.radiues[0]*cos(angles[0][0]*math.pi/180)
        y1 = self.explorer.initial_point[1]+self.explorer.radiues[0]*sin(angles[0][0]*math.pi/180)

        x2 = self.explorer.initial_point[0]+self.explorer.radiues[0]*cos(angles[0][1]*math.pi/180)
        y2 = self.explorer.initial_point[1]+self.explorer.radiues[0]*sin(angles[0][1]*math.pi/180)

        ''' #左边
        t = (self.obstacles[1][0][0]-self.explorer.initial_point[0])/self.explorer.radiues[0]
        if t>=-1 and t<=1:
            angle = acos(t)
            #angle = angle*180/math.pi
            if(angle*180/math.pi>=angles[0][0] and angle*180/math.pi>=angles[0][1]):#先判断该角度是否在已有角度中
                print("不相交")
            y = self.explorer.initial_point[1]+self.explorer.radiues[0]*sin(angle)#如果在已有角度中则判断相交点是否在局限内
            #print(angle*180/math.pi)
            x = self.obstacles[1][0][0]
            #todo 还需添加判断两条直线
            print("y:")
            print(y)
            print("x:")
            print(x)
            if y>=self.obstacles[1][0][1] and y<=self.obstacles[1][3][1]:
                print("相交")
            print(angle)
        else:
            print("不相交")

        #下边
        t = (self.obstacles[1][0][1]-self.explorer.initial_point[1])/self.explorer.radiues[0]
        if t >= -1 and t <= 1:
            angle = asin(t)
            angle = angle*180/math.pi
            print(angle)
        else:
            print("不相交")

        #右边
        t = (self.obstacles[1][2][0]-self.explorer.initial_point[0])/self.explorer.radiues[0]
        if t >= -1 and t <= 1:
            angle = acos(t)
            angle = angle*180/math.pi
            print(angle)
        else:
            print("不相交")

        #上边
        t = (self.obstacles[1][2][1]-self.explorer.initial_point[1])/self.explorer.radiues[0]
        if t >= -1 and t <= 1:
            angle = asin(t)
            angle = angle*180/math.pi
            print(angle)
        else:
            print("不相交")
        '''

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
            polygon_vertices = np.array(obstacle)
            plt.fill(polygon_vertices[:, 0], polygon_vertices[:, 1], color='black', alpha=0.5)#polygon_vertices[:, 0]表示切片所有数组的第一个元素

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

