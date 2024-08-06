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

    def new_get_edge_reward(self,edges_reduce_power,explored_ob,re_explore_power,power_rate):
        #edges_reduce_power = [edge,reduce_power,this_edge_all_sense_power]
        for edge_reduce_power in edges_reduce_power:
            if edge_reduce_power[0].get_reward()==0:#low-value
                self.reward = self.reward-edge_reduce_power[1]*(len(edges_reduce_power))
            elif edge_reduce_power[0].get_reward()==1:#middle-value
                if explored_ob:#如果有障碍则进一步削弱减成
                    self.reward = re_explore_power-power_rate*edge_reduce_power[2]/2
                else:
                    self.reward = re_explore_power - power_rate * edge_reduce_power[2]
            elif edge_reduce_power[0].get_reward()==2:#high-value
                self.reward = self.reward+(edge_reduce_power[2]-edge_reduce_power[1])*(5-len(edges_reduce_power))
                #print((edge_reduce_power[2]-edge_reduce_power[1])*(5-len(edges_reduce_power)))
            else:
                print("line reward 设置错误"+"point0x:"+str(edge_reduce_power[0].get_point_0_x())+"point0y:"+str(edge_reduce_power[0].get_point_0_y())+"point1x:"+str(edge_reduce_power[0].get_point_1_x())+"point1y:"+str(edge_reduce_power[0].get_point_1_y()))
        #print(self.reward)

    def get_power_consumption_reward(self,reduce_power,power,max_power):#todo 可改进，剩余的探索能量与总能量的关系
        power_rate =  power/max_power
        if power<0:
            self.reward = self.reward-reduce_power*2+power
        else:
            self.reward = self.reward - reduce_power
        '''
        elif power_rate>=0.5:
            self.reward = self.reward - reduce_power*0.5
        elif power_rate >= 0.25:
            self.reward = self.reward - reduce_power
        elif power_rate >= 0:
            self.reward = self.reward - reduce_power*1.5
        else:
            self.reward = self.reward-reduce_power*2
        '''

    def get_time_consumption_reward(self,time):#todo 可改进，剩余移动的时间能量和剩余的探索的时间与总能量的关系
        self.reward = self.reward-time

    def get_has_explored_reward(self,reduce_power):
        self.reward = self.reward-reduce_power

    def move_to_end_point(self,power,power_rate):
        if power_rate>=0.5:
            power = power*2
        elif power_rate >= 0.25:
            power = power*1.5
        elif power_rate >=0:
            power = power*0.5
        #加上移动结束后剩余电量
        self.reward = self.reward+100+power

    def start_explored(self,start_explored_power):
        self.reward = self.reward-start_explored_power

    def get_reward(self):
        return self.reward

    def reset_reward(self):
        self.reward = 0