from map.simulation_map.base_map import base_map
from map.simulation_map.explorer import explorer
from obstacle import Obstacle

obstacles = Obstacle()#障碍应该按照左下，右下，右上，左上，左下的方式添加
obstacles.add_obstacle([[(10, 3), (10, 5), (12,5),(12, 3), (10, 3)]])
obstacles.add_obstacle([[(35, 20),(50,20),(50,45), (35, 45), (35, 20)]])


explorers = explorer([3],[16],[40,30])



base_map1 = base_map(100,100,5,explorers)
base_map1.set_obstacles(obstacles.get_obstacle())
base_map1.collision()
base_map1.show()