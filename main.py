import math
import time
from map.simulation_map.utility import Graph

from map.simulation_map.base_map import base_map
from map.simulation_map.explorer import explorer
from map.simulation_map.obstacle import Obstacles
from AUV_action.APF import Artificial_Potential_Field

obstacles = Obstacles()  # 障碍应该按照左下，右下，右上，左上，左下的方式添加
obstacles.add_obstacles([[(10, 3), (10, 5), (12, 5), (12, 3), (10, 3)]])
obstacles.add_obstacles([[(30, 35), (60, 35), (60, 60), (30, 60), (30, 35)]])
obstacles.add_obstacles([[(10, 20), (20, 20), (20, 25), (10, 25), (10, 20)]])

#explorers = explorer([1, 2,  4, 5, 6, 7, 8], [30, 25,  23, 25, 32, 12, 24], [40, 30])

base_map1 = base_map(100, 100, 10)
base_map1.set_obstacles(obstacles.get_obstacles())
#base_map1.set_explorer(explorers)
#explorers = explorer([2, 4, 6, 7], [10, 2, 11, 13], [10, 20])
#base_map1.set_explorer(explorers)
base_map1.set_goal_point([[80,70],[20,60],[90,90]])
#base_map1.collision()
#base_map1.show()

apf = Artificial_Potential_Field(base_map1)
for i in range(1000):
    #time.sleep(1)
    a = apf.move()
    if(math.sqrt((a[0]-80)**2+(a[1]-70)**2)<=1):
        break
    #base_map1.show()
    print(a)
g = Graph(100,100,5)
g.generate_graph()
g.generate_line_reward()
g.generate_line_reward_by_points(base_map1.get_init_points())

#输入
print(base_map1.get_init_points())
print("剩余电量")
print("当前距离目标点位置")
#输出
print("第n个点")
#reward
print("消耗的电量")
print("探索到探索点的多少")
print("距离目标点的距离")

base_map1.set_line_rewards(g.get_line_rewards())
base_map1.show()