import math

import numpy as np

x_i = 40
y_i = 30
x1 = 40
y1 = 20

r = math.sqrt((x_i-x1)**2+(y_i-y1)**2)
x2 = x1+r
O = np.array([x_i, y_i])
A = np.array([x1, y1])
B = np.array([x2, y_i])

# 计算向量 OA 和 OB
OA = A - O
OB = B - O
denominator = np.linalg.norm(OA) * np.linalg.norm(OB)
if denominator == 0:
    print("分母为零，夹角无法计算。")
else:
    # 计算夹角的余弦值
    cos_theta = np.dot(OA, OB) / denominator

    # 计算弧度和角度
    theta_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    theta_degree = np.degrees(theta_rad)
    print(theta_degree)