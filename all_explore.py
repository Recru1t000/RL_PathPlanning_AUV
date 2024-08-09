import math

import numpy as np

from AUV_action.APF import Artificial_Potential_Field
from Deep_Q_learning.DQN_parameters import State, Init_Parameters
from map.simulation_map.base_map import base_map
from map.simulation_map.explorer import explorer
from map.simulation_map.obstacle import Obstacles
from parameter_type import Parameter_base_show


class AllExplore():
    def __init__(self,init_parameters):
        self.state = State()
        self.init_parameters = init_parameters
        self.basemap = None
        self.move_distance = 0
        self.path_points = None
        self.show_points = []
        self.to_target = False
        self.parameter_base_show = Parameter_base_show()
        self.heng_or_shu = init_parameters.get_heng_or_shu()

    def set_state(self):
        self.state.set_power(0)
        self.state.set_max_power(0)
        self.state.set_auv_point(self.init_parameters.get_init_start_point())
        self.state.set_target_point(self.init_parameters.get_init_target_point())

    def set_base_map(self):
        obstacles = Obstacles()  # 障碍应该按照左下，右下，右上，左上，左下的方式添加
        obstacles.add_obstacles([[(42.6, 42.5), (47.4, 42.5), (47.4, 47.5), (42.6, 47.5), (42.6, 42.5)]])
        obstacles.add_obstacles([[(22.6, 22.5), (27.4, 22.5), (27.4, 27.5), (22.6, 27.5), (22.6, 22.5)]])
        obstacles.add_obstacles([[(82.6, 82.5), (87.4, 82.5), (87.4, 87.5), (82.6, 87.5), (82.6, 82.5)]])
        obstacles.add_obstacles([[(57.6, 32.5), (62.4, 32.5), (62.4, 37.5), (57.6, 37.5), (57.6, 32.5)]])
        base_map1 = base_map(self.init_parameters.get_x_lim(), self.init_parameters.get_y_lim(), 10)
        base_map1.set_obstacles(obstacles.get_obstacles())
        base_map1.set_goal_point([self.init_parameters.get_init_target_point()])
        self.basemap = base_map1

    def all_explore_main(self):
        self.set_path_points()
        while not self.to_target:
            self.explore_all_explore()
            self.get_round_and_line_point()
        return self.state.get_power(),self.get_move_distance(),self.show_points

    def set_path_points(self):
        apf = Artificial_Potential_Field(self.basemap)
        apf.set_explorer(explorer([1], [1], self.state.get_auv_point()))
        apf.set_heng_or_shu(self.heng_or_shu)
        path = apf.move_to_goal()
        path.append(np.array(self.init_parameters.get_init_target_point()))
        path.append(np.array(self.init_parameters.get_init_target_point()))
        self.path_points = path

    def get_round_and_line_point(self):
        get_point = False
        while get_point == False:
            point0 = self.path_points.pop(0)
            point1 = self.path_points[0]
            self.show_points.append(point0)
            if point0[0] == point1[0] and point0[1] == point1[1]:
                self.state.set_auv_point(point0)
                self.to_target = True
                return
            get_point = self.find_round_and_line_intersection(point0[0], point0[1], point1[0],point1[1], self.state.get_auv_point()[0],self.state.get_auv_point()[1], self.init_parameters.get_explore_radius())
            if get_point !=False:
                self.path_points.insert(0, (get_point[0][0], get_point[0][1]))
                self.state.set_auv_point(get_point[0])
                self.move_to_all_explore_point(point0, get_point[0])
                break
            self.move_to_all_explore_point(point0,point1)
    def explore_all_explore(self):
        radius = self.init_parameters.get_explore_radius()
        explorer_all = explorer([1,2,3,4,5,6,7,8],[radius,radius,radius,radius,radius,radius,radius,radius],self.state.get_auv_point())
        self.state.add_power(self.init_parameters.get_power_consumption_value())
        if self.basemap.set_explorer(explorer_all):
            self.set_path_points()

    def move_to_all_explore_point(self,point0,point1):
        self.move_distance = self.move_distance + math.sqrt((point0[0]-point1[0])**2 + (point0[1]-point1[1])**2)
    def find_round_and_line_intersection(self,x1, y1, x2, y2, h, k, r):
        # 计算A, B, C
        A = (x2 - x1) ** 2 + (y2 - y1) ** 2
        B = 2 * ((x2 - x1) * (x1 - h) + (y2 - y1) * (y1 - k))
        C = (x1 - h) ** 2 + (y1 - k) ** 2 - r ** 2

        # 计算判别式
        delta = B ** 2 - 4 * A * C

        if delta < 0:
            return False  # 没有实数解，线段与圆不相交

        # 计算 t1 和 t2
        t1 = (-B + math.sqrt(delta)) / (2 * A)
        t2 = (-B - math.sqrt(delta)) / (2 * A)

        intersections = []

        if 0 <= t1 <= 1:
            x_intersect1 = (1 - t1) * x1 + t1 * x2
            y_intersect1 = (1 - t1) * y1 + t1 * y2
            intersections.append((x_intersect1, y_intersect1))

        if 0 <= t2 <= 1:
            x_intersect2 = (1 - t2) * x1 + t2 * x2
            y_intersect2 = (1 - t2) * y1 + t2 * y2
            intersections.append((x_intersect2, y_intersect2))

        if intersections:
            return intersections
        else:
            return False  # t1 和 t2 都不在 [0, 1] 范围内，线段与圆不相交

    def get_move_distance(self):
        return self.move_distance

    def get_show_points(self):
        return self.show_points

    def show_all_explore(self):
        self.parameter_base_show.set_base_show_path_points(self.get_show_points())
        self.basemap.base_show(self.parameter_base_show)
# 示例：在圆的点上也可以实现
x1, y1 = 11, 2
x2, y2 = 1, 12
h, k = 2, 2
r = 10


init_parameters = Init_Parameters()
all_explore = AllExplore(init_parameters)
all_explore.set_state()
all_explore.set_base_map()
print(all_explore.find_round_and_line_intersection(x1, y1, x2, y2, h, k, r))
print(math.sqrt((21-93)**2 + (22-80)**2))

down_up = [
            [[21, 6],[23, 70]],
            [[31, 16],[33, 80]],
            [[37,11],[39, 75]],
            [[53, 21],[51, 85]],
            [[66, 26],[68, 90]],
            [[72, 31],[74, 95]],
            [[83,6],[81, 70]],
            [[11,11],[14, 75]],
        ]
# left-right
left_right = [
            [[16, 37],[80, 39]],
            [[11, 27],[75, 29]],
            [[21,52],[85, 54]],
            [[16, 14],[80, 12]],
            [[31, 66],[95, 68]],
            [[6, 78],[70, 76]],
            [[16,82],[80, 84]],
            [[11,64],[75, 62]],
]
ld_ru = [
            [[16, 14],[47, 70]],
            [[31, 21],[58, 80]],
            [[48, 28], [64, 90]],
            [[9,8], [46,60]],
            [[53, 12], [79, 70]],
            [[16, 31], [71, 65]],
            [[21, 36], [76, 70]],
            [[16, 7],[26, 70]],
            [[26, 16],[83, 40]]
        ]
for d_u in left_right :
    init_parameters.set_init_start_point(d_u[0])
    init_parameters.set_init_target_point(d_u[1])
    init_parameters.set_heng_or_shu(1)
    all_explore = AllExplore(init_parameters)
    all_explore.set_state()
    all_explore.set_base_map()
    p,d,po = all_explore.all_explore_main()
    all_explore.show_all_explore()
    print(p)