import numpy as np


class Graph():
    def __init__(self, parameter_graph):
        self.x_xlim = parameter_graph.get_x_xlim()
        self.y_ylim = parameter_graph.get_y_ylim()
        self.griding_range = parameter_graph.get_griding_range()
        self.reward_function = parameter_graph.get_reward_function()
        self.array = np.zeros((int(self.x_xlim / self.griding_range), int(self.y_ylim / self.griding_range), 2))
        self.areas = Areas()
        self.line_reward = {}
        self.queue = []

    def get_line_rewards(self):
        return self.line_reward

    def generate_graph(self):
        self.areas.generate_areas(self.x_xlim, self.y_ylim, self.griding_range)

    def generate_edge_reward(self,path_points):
        self.areas.generate_edge_reward(path_points,self.reward_function)


# 划分完环境后的区域集合
class Areas():
    def __init__(self):
        self.area_list = []
        self.area_AUV_in = None
        self.distance = 0
        self.x_xlim = 0
        self.y_ylim = 0
    # 根据x和y的距离生成区域
    def generate_areas(self, x_xlim, y_ylim, griding_range):
        for i in range(int(x_xlim / griding_range)):
            for j in range(int(y_ylim / griding_range)):
                x = i * 5
                y = j * 5
                area = Area()
                area.set_left_down_point([x, y])
                area.set_right_down_point([x + griding_range, y])
                area.set_left_up_point([x, y + griding_range])
                area.set_right_up_point([x + griding_range, y + griding_range])
                self.area_list.append(area)
        self.add_edge_to_area(x_xlim, y_ylim, griding_range)
        self.distance = x_xlim**2+y_ylim**2
        self.x_xlim = x_xlim
        self.y_ylim = y_ylim

    # 根据边的坐标，将边放入对应的区域中
    # 每次生成该点的上和左的边
    def add_edge_to_area(self, x_xlim, y_ylim, griding_range):
        for i in range(int(x_xlim / griding_range)):
            for j in range(int(y_ylim / griding_range)):
                x = i * griding_range
                y = j * griding_range
                #横边
                edge_left_to_right = Edge()
                edge_left_to_right.set_point_0_x(x)
                edge_left_to_right.set_point_0_y(y)
                edge_left_to_right.set_point_1_x(x + griding_range)
                edge_left_to_right.set_point_1_y(y)
                #竖边
                edge_up_to_down = Edge()
                edge_up_to_down.set_point_0_x(x)
                edge_up_to_down.set_point_0_y(y + griding_range)
                edge_up_to_down.set_point_1_x(x)
                edge_up_to_down.set_point_1_y(y)

                for area in self.area_list:
                    # 下边
                    if [edge_left_to_right.get_point_0_x(), edge_left_to_right.get_point_0_y()] == area.get_left_down_point():
                        area.set_down_edge(edge_left_to_right)
                    # 上边
                    if [edge_left_to_right.get_point_0_x(),edge_left_to_right.get_point_0_y()] == area.get_left_up_point():
                        area.set_up_edge(edge_left_to_right)
                    # 左边
                    if [edge_up_to_down.get_point_0_x(), edge_up_to_down.get_point_0_y()] == area.get_left_up_point():
                        area.set_left_edge(edge_up_to_down)
                    # 右边
                    if [edge_left_to_right.get_point_0_x(), edge_left_to_right.get_point_0_y()] == area.get_right_up_point():
                        area.set_right_edge(edge_up_to_down)

    #此处传入的是数组
    #判断逻辑，如果此时为None，则碰到那个加入那个,如果在边或角上则判断距离终点最近的area
    #如果已有,则选择四个角的坐标距离最贴近auv和下一个到达边的点的最近的距离的区域
    def where_area_auv_in(self,auv_point,goal_point,next_edge_point):

        if self.area_AUV_in is None:
            for area in self.area_list:
                if area.is_point_in(auv_point):
                    if self.distance<area.distance_to_point(goal_point):
                        self.set_distance(area.distance_to_point(goal_point))
                        self.set_area_AUV_in(area)
        else:
            distance = 0
            for area in self.area_list:
                if area.is_point_in(auv_point):
                    area_distance = area.distance_to_point(auv_point)+area.distance_to_point(next_edge_point)
                    if distance == 0:
                        distance = area_distance
                    if distance > area_distance:
                        distance = area_distance
                        self.set_area_AUV_in(area)

    #path_points为数组
    #先判断第一个点在那个区域里
    #接下来下一个点的边在哪里，赋予权重
    #判断离下个点最近的边，为中权重
    #跳转到下一个区域，转化区域
    def generate_edge_reward(self,path_points,reward_function):
        present_point = path_points[0]
        present_area = None
        for point in path_points:
            if present_point == point:
                continue

            for area in self.area_list:
                if area.is_point_in(present_point) and area.is_point_in(point):#两个点必然在一个区域,此边为高价值边
                    present_area = area
                    edge = present_area.which_point_in_edge(point)
                    #高价值
                    reward_function.set_high_edge(present_point,point,edge)
                    #中价值
                    edge = present_area.which_middle_edge(point)
                    reward_function.set_middle_edge(present_point, point, edge)

                    break
            present_point = point

    def print_area_list(self):
        for area in self.area_list:
            area.print_points()

    def areas_reset(self):
        for area in self.area_list:
            area.area_reset()
        self.set_distance(self.x_xlim**2+self.y_ylim**2)

    def set_area_AUV_in(self,area_AUV_in):
        self.area_AUV_in = area_AUV_in

    def set_distance(self,distance):
        self.distance = distance
# 划分完环境后的区域
class Area():
    def __init__(self):
        self.up_edge = None
        self.left_edge = None
        self.down_edge = None
        self.right_edge = None
        self.AUV_in = False
        self.has_obstacle = False
        # 数组格式
        self.left_up_point = []
        self.right_up_point = []
        self.left_down_point = []
        self.right_down_point = []

    def set_AUV_in(self, boolean):
        self.AUV_in = boolean

    def set_has_obstacle(self, boolean):
        self.has_obstacle = boolean

    def set_left_edge(self, edge):
        self.left_edge = edge

    def set_right_edge(self, edge):
        self.right_edge = edge

    def set_up_edge(self, edge):
        self.up_edge = edge

    def set_down_edge(self, edge):
        self.down_edge = edge

    def set_left_up_point(self, left_up_point):
        self.left_up_point = left_up_point

    def set_right_up_point(self, right_up_point):
        self.right_up_point = right_up_point

    def set_left_down_point(self, left_down_point):
        self.left_down_point = left_down_point

    def set_right_down_point(self, right_down_point):
        self.right_down_point = right_down_point

    def get_up_edge(self):
        return self.up_edge

    def get_down_edge(self):
        return self.down_edge

    def get_left_edge(self):
        return self.left_edge

    def get_right_edge(self):
        return self.right_edge

    def get_left_up_point(self):
        return self.left_up_point

    def get_right_up_point(self):
        return self.right_up_point

    def get_left_down_point(self):
        return self.left_down_point

    def get_right_down_point(self):
        return self.right_down_point

    def print_points(self):
        print("left_down_point:" + str(self.left_down_point) + ",right_down_point:" + str(self.right_down_point) +
              ",left_up_point:" + str(self.left_up_point) + ",right_up_point:" + str(self.right_up_point))

    def area_reset(self):
        self.left_edge.edge_reset()
        self.right_edge.edge_reset()
        self.up_edge.edge_reset()
        self.down_edge.edge_reset()

    def is_point_in(self,point):
        x = point[0]
        y = point[1]
        #判断左上角
        if self.left_up_point[0]<=x and self.left_up_point[1]>=y:
            #判断右上角
            if self.right_up_point[0]>=x and  self.right_up_point[1]>=y:
                #判断左下角
                if self.left_down_point[0]<=x and self.left_down_point[1]<=y:
                    #判断右下角
                    if self.right_down_point[0]>=x and self.right_down_point[1]<=y:
                        return True
        return False

    def distance_to_point(self,point):
        x = point[0]
        y = point[1]
        distance_left_up_point = (self.left_up_point[0]-x)**2+(self.left_up_point[1]-y)**2
        distance_left_down_point = (self.left_down_point[0] - x) ** 2 + (self.left_down_point[1] - y) ** 2
        distance_right_up_point = (self.right_up_point[0] - x) ** 2 + (self.right_up_point[1] - y) ** 2
        distance_right_down_point = (self.right_down_point[0] - x) ** 2 + (self.right_down_point[1] - y) ** 2
        distance = distance_left_up_point+distance_left_down_point+distance_right_up_point+distance_right_down_point
        return distance

    def which_point_in_edge(self,point):
        if self.up_edge.point_in_edge(point):
            return self.up_edge
        if self.left_edge.point_in_edge(point):
            return self.left_edge
        if self.right_edge.point_in_edge(point):
            return self.right_edge
        if self.down_edge.point_in_edge(point):
            return self.down_edge

    #目前没考虑到以探索的边
    def which_middle_edge(self,point):
        edge = None
        edge_list = []
        edge_list.append([self.up_edge.distance_point_to_edge(point),self.up_edge])
        edge_list.append([self.left_edge.distance_point_to_edge(point),self.left_edge])
        edge_list.append([self.right_edge.distance_point_to_edge(point),self.right_edge])
        edge_list.append([self.down_edge.distance_point_to_edge(point),self.down_edge])
        sorted_edge_list = sorted(edge_list, key=lambda x: x[0])

        # 找到第二小的数组，并将其第二位赋值给 edge
        if len(sorted_edge_list) >= 2:
            edge = sorted_edge_list[1][1]
        return edge
# 区域中的每一条边
class Edge():
    # point0为最上或最左的点
    # point1为最下或最右的点
    # 数值格式
    def __init__(self):
        self.point_0_x = None
        self.point_0_y = None
        self.point_1_x = None
        self.point_1_y = None
        self.reward = 0

    def set_point_0_x(self, point_0_x):
        self.point_0_x = point_0_x

    def set_point_0_y(self, point_0_y):
        self.point_0_y = point_0_y

    def set_point_1_x(self, point_1_x):
        self.point_1_x = point_1_x

    def set_point_1_y(self, point_1_y):
        self.point_1_y = point_1_y

    def set_reward(self, reward):
        self.reward = reward

    def get_point_0_x(self):
        return self.point_0_x

    def get_point_0_y(self):
        return self.point_0_y

    def get_point_1_x(self):
        return self.point_1_x

    def get_point_1_y(self):
        return self.point_1_y

    def get_reward(self):
        return self.reward

    def judge_edge(self):
        if self.point_0_x <= self.point_1_x:
            print("False,left" + str(self.point_0_x) + "right" + str(self.point_1_x))
        if self.point_0_y >= self.point_1_y:
            print("False,up" + str(self.point_0_y) + "down" + str(self.point_1_y))

    def edge_reset(self):
        self.reward = 0

    def point_in_edge(self,point):
        x = point[0]
        y = point[1]
        if self.point_0_x<=x<=self.point_1_x and self.point_0_y>=y>=self.point_1_y:
            return True
        else:
            print("self.point_0_x:"+str(self.point_0_x)+",x:"+str(x)+",self.point_1_x:"+str(self.point_1_x))
            print(self.point_0_x<=x<=self.point_1_x)
            print("self.point_0_y:" + str(self.point_0_y) + ",y:" + str(y) + ",self.point_1_y:" + str(self.point_1_y))
            print(self.point_0_y<=y<=self.point_1_y)
            return False

    def distance_point_to_edge(self,point):
        x = point[0]
        y = point[1]
        distance = (self.point_0_x-x)**2+(self.point_0_y-y)**2+(self.point_1_x-x)**2+(self.point_1_y-y)**2
        return distance