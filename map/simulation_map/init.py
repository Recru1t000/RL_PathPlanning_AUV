from map.simulation_map.base_map import base_map
from map.simulation_map.explorer import explorer
from obstacle import Obstacles

obstacles = Obstacles()  # 障碍应该按照左下，右下，右上，左上，左下的方式添加
obstacles.add_obstacles([[(10, 3), (10, 5), (12, 5), (12, 3), (10, 3)]])
obstacles.add_obstacles([[(30, 35), (60, 35), (60, 60), (30, 60), (30, 35)]])
obstacles.add_obstacles([[(10, 20), (20, 20), (20, 25), (10, 25), (10, 20)]])

explorers = explorer([1, 2,  4, 5, 6, 7, 8], [30, 25,  23, 25, 32, 12, 24], [40, 30])

base_map1 = base_map(100, 100, 10)
base_map1.set_obstacles(obstacles.get_obstacles())
base_map1.set_explorer(explorers)
explorers = explorer([2, 4, 6, 7], [10, 2, 11, 13], [10, 20])
base_map1.set_explorer(explorers)
#base_map1.collision()
base_map1.base_show()
