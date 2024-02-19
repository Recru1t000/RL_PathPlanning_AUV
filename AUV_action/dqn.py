import math

from AUV_action.APF import Artificial_Potential_Field
from AUV_action.AUV_based import Environment, Base_Parameters
from map.simulation_map.base_map import base_map
from map.simulation_map.obstacle import Obstacles

obstacles = Obstacles()  # 障碍应该按照左下，右下，右上，左上，左下的方式添加
obstacles.add_obstacles([[(10, 3), (10, 5), (12, 5), (12, 3), (10, 3)]])
obstacles.add_obstacles([[(30, 35), (60, 35), (60, 60), (30, 60), (30, 35)]])
obstacles.add_obstacles([[(10, 20), (20, 20), (20, 25), (10, 25), (10, 20)]])

base_map1 = base_map(100, 100, 10)
base_map1.set_obstacles(obstacles.get_obstacles())

base_map1.set_goal_point([[80,70]])

apf = Artificial_Potential_Field(base_map1)
for i in range(1000):
    #time.sleep(1)
    a = apf.move()
    if(math.sqrt((a[0]-80)**2+(a[1]-70)**2)<=1):
        break
    #base_map1.show()
    print(a)

base_map1.get_init_points()


def APF_move(init_point,apf,goal_point):
    while math.sqrt((a[0]-goal_point[0])**2+(a[1]-goal_point[1])**2)<=1:
        apf.set_initial_point(init_point)
        a = apf.move()
        #print(a)

base_parameters = Base_Parameters(1,1,1,1)

electric = 100
init_point = [10,10]
goal_point = [80,70]
radius = 5
#APF_move(init_point,apf,goal_point)
env = Environment(electric,init_point,goal_point,radius,apf.get_init_points(),base_map1,base_parameters)
rrrrr = env.step(1)
if rrrrr is False:
    env.artificial_potential_field()
env.step(3)
env.env_show()

