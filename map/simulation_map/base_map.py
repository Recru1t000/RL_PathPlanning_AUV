import math
from math import cos,sin,acos,asin
from obstacle import *
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Wedge
from map.simulation_map.utility import Collision

class base_map():
    def __init__(self,x_xlim,y_ylim,figure_value):
        self.xlim = x_xlim#x轴
        self.ylim = y_ylim#y轴
        self.figure_value = figure_value #默认为15
        self.obstacles = []
        self.explorer = None
        self.explorered = []#记录已探索的点，以方便画图
        self.starting_point = None
        self.goal_point = None #数据格式为[[1,0],[2,0],[3,0]]
        self.init_points = []
        self.line_rewards = {}


    def set_obstacles(self,obstacles):
        self.obstacles = obstacles

    def base_map_reset(self):
        self.obstacles.obstacles_reset()
        self.init_points = []
        self.explorer = None
        self.explorered = []


    def set_line_rewards(self,line_rewards):
        self.line_rewards = line_rewards
        print(line_rewards)

    def collision(self):
        angles = self.explorer.get_angle(self.explorer.angles)
        for obstacle in self.obstacles:
            remove_repeat_obstacle_left_point = False
            for i in range(len(angles)):
                c = Collision(self.explorer.initial_point[0], self.explorer.initial_point[1], obstacle,
                              self.explorer.radiues[i], angles[i])
                if len(c.get_collision_points()) !=0:
                    #remove_repeat_obstacle_left_point = True
                    if len(c.left_points) != 0 and len(c.left_points) != 1:
                        obstacle.add_left_explored(c.left_points)
                    if len(c.bottom_points) != 0 and len(c.bottom_points) != 1:
                        obstacle.add_bottom_explored(c.bottom_points)
                    if len(c.right_points) != 0 and len(c.right_points) != 1:
                        obstacle.add_right_explored(c.right_points)
                    if len(c.up_points) != 0 and len(c.up_points) != 1:
                        obstacle.add_up_explored(c.up_points)

                #提取每个障碍的所有点，结合到一起按照下，右，上，左进行排列，最后加上最左边的点
            #if remove_repeat_obstacle_left_point:
            #    obstacle.arrange_every_side()

        print(self.obstacles)

    def set_explorer(self,explorer):
        self.explorer = explorer
        self.explorered.append(explorer)
        self.collision()

    def set_goal_point(self,goal_point):
        self.goal_point = goal_point

    def get_goal_point(self):
        return self.goal_point

    def append_init_points(self,point):
        self.init_points.append(point)

    def get_init_points(self):
        return self.init_points

    def get_obstacles(self):
        return self.obstacles
    def show(self):


        fig, ax = plt.subplots(figsize=(self.figure_value, self.figure_value))
        for explorer in self.explorered:
            angles = explorer.get_angle(explorer.angles)
            # 生成八分之一圆形
            for radius,angle in zip(explorer.radiues,angles):
                first_angle = angle[0]
                second_angle = angle[1]
                wedge = Wedge((explorer.initial_point[0], explorer.initial_point[1]), radius, first_angle, second_angle, facecolor='blue')
                ax.add_patch(wedge)

        x_datas = [point[0] for point in self.init_points]
        y_datas = [point[1] for point in self.init_points]
        # 使用plot函数绘制连续轨迹
        plt.plot(x_datas, y_datas, label='Trajectory',color='black')

        # 添加黑色多边形
        for obstacle in self.obstacles:
            polygon_vertices = np.array(obstacle.get_obstacle())
            plt.fill(polygon_vertices[:, 0], polygon_vertices[:, 1], color='black', alpha=0.5)#polygon_vertices[:, 0]表示切片所有数组的第一个元素

        #todo 生成形状还是生成线段？？？
        '''
        for obstacle in self.obstacles:
            obstacle_points = obstacle.show_point()
            if len(obstacle_points) !=0:
                polygon_vertices = np.array(obstacle_points)
                plt.fill(polygon_vertices[:, 0], polygon_vertices[:, 1], color='red', alpha=0.5)#polygon_vertices[:, 0]表示切片所有数组的第一个元素
        '''
        for obstacle in self.obstacles:
            obstacle_points = obstacle.show_line()
            for lines in obstacle_points:
                for line in lines:
                    if len(line) != 0:
                        polygon_vertices = np.array(line)
                        plt.fill(polygon_vertices[:, 0], polygon_vertices[:, 1], color='red', alpha=0.5)#polygon_vertices[:, 0]表示切片所有数组的第一个元素

        for key, value in self.line_rewards.items():
            polygon_vertices = np.array(key)
            if value==0:
                #polygon_vertices = np.array(key)
                plt.fill(polygon_vertices[:, 0], polygon_vertices[:, 1], color='black', alpha=0.5)
            elif value==10:
                plt.fill(polygon_vertices[:, 0], polygon_vertices[:, 1], color='white', alpha=0.5,linewidth=1)
        #添加目标点
        x_values = [point[0] for point in self.get_goal_point()]
        y_values = [point[1] for point in self.get_goal_point()]
        plt.scatter(x_values, y_values, c='red', label='Red Points')

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
