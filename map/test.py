import matplotlib.pyplot as plt


class Find_intersections():
    def find_intersections(self,points, grid_size, cell_size):
        def line_intersection(p1, p2, q1, q2):
            def on_segment(p, q, r):
                if min(p[0], q[0]) <= r[0] <= max(p[0], q[0]) and min(p[1], q[1]) <= r[1] <= max(p[1], q[1]):
                    return True
                return False

            def orientation(p, q, r):
                val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
                if val == 0:
                    return 0
                return 1 if val > 0 else 2

            o1 = orientation(p1, p2, q1)
            o2 = orientation(p1, p2, q2)
            o3 = orientation(q1, q2, p1)
            o4 = orientation(q1, q2, p2)

            if o1 != o2 and o3 != o4:
                denom = (p2[0] - p1[0]) * (q2[1] - q1[1]) - (p2[1] - p1[1]) * (q2[0] - q1[0])
                if denom == 0:
                    return None
                num1 = (p1[1] - q1[1]) * (q2[0] - q1[0]) - (p1[0] - q1[0]) * (q2[1] - q1[1])
                num2 = (p1[1] - q1[1]) * (p2[0] - p1[0]) - (p1[0] - q1[0]) * (p2[1] - p1[1])
                t1 = num1 / denom
                t2 = num2 / denom
                if 0 <= t1 <= 1 and 0 <= t2 <= 1:
                    return (p1[0] + t1 * (p2[0] - p1[0]), p1[1] + t1 * (p2[1] - p1[1]))
            return None

        intersections = set()  # 使用集合避免重复

        for i in range(len(points) - 1):
            p1, p2 = points[i], points[i + 1]

            x_min, x_max = min(p1[0], p2[0]), max(p1[0], p2[0])
            y_min, y_max = min(p1[1], p2[1]), max(p1[1], p2[1])

            for x in range(0, grid_size + 1, cell_size):
                for y in range(0, grid_size + 1, cell_size):
                    grid_lines = [
                        ((x, y), (x + cell_size, y)),
                        ((x, y), (x, y + cell_size)),
                        ((x + cell_size, y), (x + cell_size, y + cell_size)),
                        ((x, y + cell_size), (x + cell_size, y + cell_size))
                    ]

                    for q1, q2 in grid_lines:
                        intersection = line_intersection(p1, p2, q1, q2)
                        if intersection and x_min <= intersection[0] <= x_max and y_min <= intersection[1] <= y_max:
                            intersections.add(intersection)

        return list(intersections)

def plot_grid_and_segments(points, grid_size, cell_size, intersections):
    fig, ax = plt.subplots()

    # 绘制网格线段
    for x in range(0, grid_size + 1, cell_size):
        ax.plot([x, x], [0, grid_size], color='lightgrey', linestyle='--')
    for y in range(0, grid_size + 1, cell_size):
        ax.plot([0, grid_size], [y, y], color='lightgrey', linestyle='--')

    # 绘制线段
    for i in range(len(points) - 1):
        p1, p2 = points[i], points[i + 1]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='blue')

    # 绘制交点
    for inter in intersections:
        ax.plot(inter[0], inter[1], 'ro')  # 交点标记为红色

    ax.set_xlim(0, grid_size)
    ax.set_ylim(0, grid_size)
    ax.set_aspect('equal')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Line Segments and Grid Intersections')
    plt.grid(True)
    plt.show()

# 示例点数组
points = []


# 环境大小和网格线长度
grid_size = 100
cell_size = 5

# 找到交点
intersections = find_intersections(points, grid_size, cell_size)

print("交点:")
for point in sorted(intersections):
    print(point)

# 可视化
plot_grid_and_segments(points, grid_size, cell_size, intersections)
