import math
class DQN_Explorer():
    def __init__(self,radius):
        self.one = [135, 180]
        self.two = [90, 135]
        self.three = [45, 90]
        self.four = [0, 45]
        self.five = [315, 360]
        self.six = [270, 315]
        self.seven = [225, 270]
        self.eight = [180, 225]
        self.radius = radius

    #返回字典数组[[1,radius],[2,radius]]
    def explore_edge(self, edge, auv_point):
        s = Sector()
        edge_point = [[edge.get_point_0_x(),edge.get_point_0_y()],[edge.get_point_1_x(),edge.get_point_1_y()]]
        result = s.sectors_crossed(auv_point[0], auv_point[1], edge_point)
        return result

    def old_explorer_to_new(self,name):
        if name == "one":
            return 1
        elif name == "two":
            return 2
        elif name == "three":
            return 3
        elif name == "four":
            return 4
        elif name == "five":
            return 5
        elif name == "six":
            return 6
        elif name == "seven":
            return 7
        elif name == "eight":
            return 8
class Sector:
    def __init__(self):
        self.one = [135, 180]
        self.two = [90, 135]
        self.three = [45, 90]
        self.four = [0, 45]
        self.five = [315, 360]
        self.six = [270, 315]
        self.seven = [225, 270]
        self.eight = [180, 225]

    def get_angle_from_name(self,name):
        if name == 'one':
            return self.one
        elif name == 'two':
            return self.two
        elif name == 'three':
            return self.three
        elif name == 'four':
            return self.four
        elif name == 'five':
            return self.five
        elif name == 'six':
            return self.six
        elif name == 'seven':
            return self.seven
        elif name == 'eight':
            return self.eight

    def angle_from_center(self,x0, y0, x, y):
        """
        计算从圆心 (x0, y0) 到点 (x, y) 的角度，并转换为 0 到 360 度之间。
        """
        angle = math.degrees(math.atan2(y - y0, x - x0))
        if angle < 0:
            angle += 360
        return angle


    def is_within_sector(self,angles, sector):
        """
        检查一个角度是否在给定扇形的范围内。
        """
        #目前还有一种情况没有考虑到就是如果正好在顶点上怎么办
        start, end = sector
        #如果两个角正好相等，则说明可以
        if angles[0]==start and angles[1]==end:
            return True
        #在两边的差大于180度时,不要考虑在本条边探索本条边的问题，前面会进行排除
        if angles[1] - angles[0] > 180:
            if 0 <=start< angles[0] or 0 <=end< angles[0]:
                return True
            elif angles[1] <start<= 360 or angles[1] <end<= 360:
                return True

        #在两边的差小于180度时
        elif angles[0] <start< angles[1] or angles[0] <end< angles[1]:
            return True

        return False



    def sectors_crossed(self,x0, y0, edge):
        """
        判断边是否穿过任何扇形，并返回穿过的扇形列表。
        """
        x1, y1 = edge[0]
        x2, y2 = edge[1]
        sector_angles = Sector()
        sectors_crossed = []

        # 计算边的两端点相对于圆心的角度
        angle1 = self.angle_from_center(x0, y0, x1, y1)
        angle2 = self.angle_from_center(x0, y0, x2, y2)

        if angle1 > angle2:
            angle1, angle2 = angle2, angle1
            x1,x2 = x2,x1
            y1,y2 = y2,y1
        angles = [angle1, angle2]
        # 遍历所有扇形，判断边是否穿过
        for sector_name, sector_angle in vars(sector_angles).items():
            if self.is_within_sector(angles, sector_angle):
                r = self.angles_radius([x0,y0],[angle1,angle2],sector_angles.get_angle_from_name(sector_name),[[x1,y1],[x2,y2]])
                sectors_crossed.append({sector_name:r})
        return sectors_crossed


    def sector_with_line_point(self,center, line_points, angle):
        x0, y0 = center
        (x1, y1), (x2, y2) = line_points
        theta1 = angle

        if x1 == x2:  # Line is vertical
            x_intersect = x1

            y_intersect1 = y0 + (x_intersect - x0) * math.tan(math.radians(theta1))
            distance = self.calculate_distance(center,[x_intersect,y_intersect1])
            return distance

        elif y1 == y2:  # Line is horizontal
            y_intersect = y1

            x_intersect1 = x0 + (y_intersect - y0) / math.tan(math.radians(theta1))
            distance = self.calculate_distance(center,[x_intersect1,y_intersect])

            return distance

        else:
            raise ValueError("Line must be parallel to x-axis or y-axis")
    #计算单个角度的半径
    def angles_radius(self,center,edge_angles,sector_angles,edge):
        #0.相等时直接传入比大小
        if edge_angles[0] == sector_angles[0] and edge_angles[1] == sector_angles[1]:
            return max(self.sector_with_line_point(center,edge,sector_angles[0]),self.sector_with_line_point(center,edge,sector_angles[1]))
        #1.判断start和end是否都在angle[0]和angle[1]中，如果在则直接计算即可,找到最大值即可。
        if edge_angles[1] - edge_angles[0] > 180:
            if not edge_angles[0] <sector_angles[0]< edge_angles[1] and not edge_angles[0] <sector_angles[1]< edge_angles[1]:
                return max(self.sector_with_line_point(center,edge,sector_angles[0]),self.sector_with_line_point(center,edge,sector_angles[1]))
            elif not edge_angles[0] <sector_angles[0]< edge_angles[1]:
                edge_near_point = self.judge_near_point(edge_angles,sector_angles[1],edge)
                return max(self.sector_with_line_point(center,edge,sector_angles[0]), self.calculate_distance(center,edge_near_point))
            elif not edge_angles[0] <sector_angles[1]< edge_angles[1]:
                edge_near_point = self.judge_near_point(edge_angles, sector_angles[0], edge)
                return max(self.sector_with_line_point(center, edge, sector_angles[1]),self.calculate_distance(center, edge_near_point))
        else:
            if edge_angles[0] <=sector_angles[0]<= edge_angles[1] and edge_angles[0] <=sector_angles[1]<= edge_angles[1]:
                return max(self.sector_with_line_point(center,edge,sector_angles[0]),self.sector_with_line_point(center,edge,sector_angles[1]))
            elif edge_angles[0] <=sector_angles[0]<= edge_angles[1]:
                edge_near_point = self.judge_near_point(edge_angles,sector_angles[1],edge)
                return max(self.sector_with_line_point(center,edge,sector_angles[0]), self.calculate_distance(center,edge_near_point))
            elif edge_angles[0] <sector_angles[1]< edge_angles[1]:
                edge_near_point = self.judge_near_point(edge_angles, sector_angles[0], edge)
                return max(self.sector_with_line_point(center, edge, sector_angles[1]),self.calculate_distance(center, edge_near_point))
        return 1
        #2.如果有一条边不在则计算不在的那一条边最靠近的那个顶点的距离，该值即为最大值

    def calculate_distance(self,center,edge):
        x1,y1 = center
        x2,y2 = edge
        return math.sqrt((x2-x1)**2 + (y2-y1)**2)
    def judge_near_point(self,edge_angles,angle,edge):
        if abs(angle-edge_angles[0]) < abs(angle-edge_angles[1]):
            return edge[0]
        else:
            return edge[1]


