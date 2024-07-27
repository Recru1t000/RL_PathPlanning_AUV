from math import sqrt


class Reward():
    def __init__(self,start_point,end_point):
        self.start_point = start_point
        self.end_point = end_point

    #还需要修改
    def set_high_edge(self,first_point,second_point,edge):
        high_edge_reward = sqrt((first_point[0]-second_point[0])**2 + (first_point[1]-second_point[1])**2)
        edge.set_reward(high_edge_reward)

    def set_middle_edge(self,first_point,second_point,edge):
        middle_edge_reward = sqrt((first_point[0]-second_point[0])**2 + (first_point[1]-second_point[1])**2)
        edge.set_reward(middle_edge_reward)