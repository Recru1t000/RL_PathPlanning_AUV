import math


class Time_Consumption():
    def __init__(self):
        self.time = None
        self.move_time = None
        self.explore_time = None

    def move_time_consumption(self,point1,point2):
        reduce_time = math.sqrt((point1[0]-point2[0])**2+(point1[1]-point2[1])**2)
        return reduce_time


    def set_move_time(self, move_time):
        self.move_time = move_time

    def set_explore_time(self, explore_time):
        self.explore_time = explore_time

    def get_move_time(self):
        return self.move_time

    def get_explore_time(self):
        return self.explore_time