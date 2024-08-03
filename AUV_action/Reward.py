from math import sqrt


class Reward():
    def __init__(self,start_point,end_point):
        self.start_point = start_point
        self.end_point = end_point
        self.reward = 0

    #还需要修改
    def set_high_edge(self,first_point,second_point,edge):
        high_edge_reward = 2
        edge.set_reward(high_edge_reward)

    def set_middle_edge(self,first_point,second_point,edge):
        middle_edge_reward = 1
        edge.set_reward(middle_edge_reward)

    def get_edge_reward(self,edge_list,reduce_power,first_point,second_point,power_consumption_full_sensing):
        distance = sqrt((first_point[0]-second_point[0])**2 + (first_point[1]-second_point[1])**2)
        for edge in edge_list:
            if edge.get_reward()==0:#low-value
                self.reward = self.reward-reduce_power
            elif edge.get_reward()==1:#middle-value
                #todo middle未完成
                self.reward = self.reward+distance*(power_consumption_full_sensing-reduce_power)
            elif edge.get_reward()==2:#high-value
                self.reward = self.reward+distance*(power_consumption_full_sensing-reduce_power)
            else:
                print("line reward 设置错误"+"point0x:"+str(edge.get_point_0_x())+"point0y:"+str(edge.get_point_0_y())+"point1x:"+str(edge.get_point_1_x())+"point1y:"+str(edge.get_point_1_y()))

    def get_power_consumption_reward(self,reduce_power):#todo 可改进，剩余的探索能量与总能量的关系
        self.reward = self.reward-reduce_power

    def get_time_consumption_reward(self,time):#todo 可改进，剩余移动的时间能量和剩余的探索的时间与总能量的关系
        self.reward = self.reward-time

    def get_has_explored_reward(self,reduce_power):
        self.reward = self.reward-reduce_power

    def move_to_end_point(self):#todo 可改进
        self.reward = self.reward+400

    def get_reward(self):
        return self.reward