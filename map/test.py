import pygame
import sys
import math

# 初始化 Pygame
pygame.init()

# 设置窗口尺寸
width, height = 800, 600
screen = pygame.display.set_mode((width, height))

# 定义 1/8 扇形类
class EighthPie:
    def __init__(self, x, y, radius, angle_range, color):
        self.x = x
        self.y = y
        self.radius = radius
        self.angle_range = angle_range
        self.color = color

    def draw(self):
        # 计算扇形的边界点
        points = [(self.x, self.y)]
        for angle in range(int(self.angle_range[0] * 180 / math.pi), int(self.angle_range[1] * 180 / math.pi) + 1):
            x = self.x + int(self.radius * math.cos(math.radians(angle)))
            y = self.y + int(self.radius * math.sin(math.radians(angle)))
            points.append((x, y))

        # 使用 polygon 绘制扇形
        pygame.draw.polygon(screen, self.color, points)

    def check_collision_with_polygon(self, polygon_points):
        # 检测圆心是否在多边形内
        if self.point_in_polygon(self.x, self.y, polygon_points):
            return True

        # 检测扇形的边界点是否在多边形内
        for angle in range(int(self.angle_range[0] * 180 / math.pi), int(self.angle_range[1] * 180 / math.pi) + 1):
            x = self.x + int(self.radius * math.cos(math.radians(angle)))
            y = self.y + int(self.radius * math.sin(math.radians(angle)))
            if self.point_in_polygon(x, y, polygon_points):
                return True

        return False

    def point_in_polygon(self, x, y, polygon_points):
        # 判断点是否在多边形内，使用射线法
        count = 0
        for i in range(len(polygon_points)):
            x1, y1 = polygon_points[i]
            x2, y2 = polygon_points[(i + 1) % len(polygon_points)]
            if ((y1 <= y and y < y2) or (y2 <= y and y < y1)) and \
               (x < x1 + (x2 - x1) * (y - y1) / (y2 - y1)):
                count += 1
        return count % 2 == 1

# 定义多边形类
class Polygon:
    def __init__(self, points, color):
        self.points = points
        self.color = color

    def draw(self):
        pygame.draw.polygon(screen, self.color, self.points)

# 创建一个 1/8 扇形对象和一个多边形对象
my_eighth_pie = EighthPie(400, 300, 33, (math.pi, 3*math.pi/2), (255, 0, 0))
my_polygon = Polygon([(300, 200), (400, 150), (500, 200), (450, 300)], (0, 0, 255))

# 游戏主循环
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    # 清空屏幕
    screen.fill((255, 255, 255))

    # 绘制 1/8 扇形和多边形
    my_eighth_pie.draw()
    my_polygon.draw()

    # 检测 1/8 扇形与多边形是否相交
    if my_eighth_pie.check_collision_with_polygon(my_polygon.points):
        print("1/8 扇形与多边形相交！")
    else:
        print("1/8 扇形与多边形不相交。")

    # 更新显示
    pygame.display.flip()

    # 控制帧速率
    pygame.time.Clock().tick(60)
