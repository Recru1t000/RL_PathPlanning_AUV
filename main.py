import matplotlib.pyplot as plt

# 生成一些示例数据
x_datas = []
y_datas = []
i = 0
j = 0
while i<10:
    i = i+0.0001
    x_datas.append(i)
    j = i**2+3
    y_datas.append(j)

print(x_datas)
# 使用plot函数绘制连续轨迹
plt.figure(figsize=(15, 15))
plt.plot(x_datas, y_datas, label='Trajectory')
plt.xlim(0, 100)
plt.ylim(0, 100)
# 添加标签和标题

plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Continuous Trajectory Plot')


# 添加图例
plt.legend()

# 显示图形
plt.show()