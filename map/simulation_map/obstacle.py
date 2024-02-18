class Obstacles():
    # todo APF每次更新后的obstacle的已探索边与APF相关联，但不直接调用obstacle中的边
    # todo 每次更新后已探索边后，在obstacle中调用APF
    def __init__(self):
        self.obstacles = []

    def add_obstacles(self, obstacles):
        for obstacle in obstacles:
            class_obstacle = Obstacle(obstacle)
            self.obstacles.append(class_obstacle)

    def get_obstacles(self):  # 直接获取全部的Obstacle类
        return self.obstacles

    def get_class_obstacle_from_coordinate(self, obstacle):  # 根据Obstacle类的坐标获取对应的Obstacle类
        for class_obstacle in self.obstacles:
            if class_obstacle.get_class_obstacle(obstacle):
                return class_obstacle

    def obstacles_reset(self):
        for class_obstacle in self.obstacles:
            class_obstacle.obstacle_reset()

class Obstacle():
    def __init__(self, obstacle):
        self.obstacle = obstacle
        self.bottom_explored_points = []
        self.right_explored_points = []
        self.up_explored_points = []
        self.left_explored_points = []

    def obstacle_reset(self):
        self.bottom_explored_points = []
        self.right_explored_points = []
        self.up_explored_points = []
        self.left_explored_points = []

    def get_class_obstacle(self, obstacle):  # 通过坐标获取对应的类
        if (obstacle == self.obstacle):
            return True

    def get_obstacle(self):
        return self.obstacle

    def get_bottom_explored_points(self):
        return self.bottom_explored_points

    def get_right_explored_points(self):
        return self.right_explored_points

    def get_up_explored_points(self):
        return self.up_explored_points

    def get_left_explored_points(self):
        return self.left_explored_points

    def add_bottom_explored(self, point):
        self.bottom_explored_points.append(point)
        for bottom_explored_point in self.bottom_explored_points:
            bottom_explored_point.sort(key=lambda x: x[0])
        self.bottom_explored_points.sort(key=lambda x: x[0])

    def add_right_explored(self, point):
        self.right_explored_points.append(point)
        for right_explored_point in self.right_explored_points:
            right_explored_point.sort(key=lambda x: x[0])
        self.right_explored_points.sort(key=lambda x: x[0])

    def add_up_explored(self, point):
        self.up_explored_points.append(point)
        for up_explored_point in self.up_explored_points:
            up_explored_point.sort(key=lambda x: x[0])
        self.up_explored_points.sort(key=lambda x: x[0])

    def add_left_explored(self, point):
        self.left_explored_points.append(point)
        for left_explored_point in self.left_explored_points:
            left_explored_point.sort(key=lambda x: x[0])
        self.left_explored_points.sort(key=lambda x: x[0])

    def arrange_every_side(self):
        # self.bottom_explored_points = list(set(self.bottom_explored_points))
        for bottom_explored_point in self.bottom_explored_points:
            bottom_explored_point.sort(key=lambda x: x[0])
        self.bottom_explored_points.sort(key=lambda x: x[0][0])

        self.right_explored_points = list(set(self.right_explored_points))
        self.right_explored_points.sort(key=lambda x: x[1])

        self.up_explored_points = list(set(self.up_explored_points))
        self.up_explored_points.sort(key=lambda x: x[0], reverse=True)

        self.left_explored_points = list(set(self.left_explored_points))
        self.left_explored_points.sort(key=lambda x: x[1], reverse=True)

    def show_point(self):
        show_points = []
        for point in self.bottom_explored_points:
            show_points.append(point)
        for point in self.right_explored_points:
            show_points.append(point)
        for point in self.up_explored_points:
            show_points.append(point)
        for point in self.left_explored_points:
            show_points.append(point)
        return show_points

    def show_line(self):
        show_points = [self.bottom_explored_points, self.right_explored_points, self.up_explored_points,
                       self.left_explored_points]
        return show_points
