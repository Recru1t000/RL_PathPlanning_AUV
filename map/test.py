import math

x1 = 78.02276999
y1 = 66.31170447
x2 = 77.55029657
y2 = 65.43035955
# 定义两个点的坐标
point1 = (x1, y1)  # x1、y1为第一个点的坐标值
point2 = (x2, y2)  # x2、y2为第二个点的坐标值

# 计算两个点之间的距离
distance_between_points = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
print(distance_between_points)
# 计算两个点之间的夹角（单位为弧度）
angle_in_radians = math.atan2(y2 - y1, x2 - x1)

# 将夹角转换为角度制
angle_in_degrees = angle_in_radians * 180 / math.pi
if angle_in_degrees<0:
    angle_in_degrees = 360+angle_in_degrees

print("两个点之间的夹角为：", angle_in_degrees, "°")

for i in list():
    print(i)