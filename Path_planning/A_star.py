import matplotlib.pyplot as plt
import heapq
import math


class Node:
    def __init__(self, position, parent=None):
        self.position = position
        self.parent = parent
        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position

    def __lt__(self, other):
        return self.f < other.f


def heuristic(node, goal):
    return math.sqrt((node.position[0] - goal.position[0]) ** 2 + (node.position[1] - goal.position[1]) ** 2)


def get_neighbors(node, obstacles, step_size):
    neighbors = []
    for dx in [-step_size, 0, step_size]:
        for dy in [-step_size, 0, step_size]:
            if dx == 0 and dy == 0:
                continue
            new_position = (node.position[0] + dx, node.position[1] + dy)
            if not is_in_obstacle(new_position, obstacles):
                neighbors.append(Node(new_position, node))
    return neighbors


def is_in_obstacle(position, obstacles):
    for (ox, oy, width, height) in obstacles:
        if ox <= position[0] <= ox + width and oy <= position[1] <= oy + height:
            return True
    return False


def a_star_search(start, goal, obstacles, step_size):
    open_list = []
    closed_list = set()
    start_node = Node(start)
    goal_node = Node(goal)
    heapq.heappush(open_list, start_node)

    while open_list:
        current_node = heapq.heappop(open_list)
        closed_list.add(current_node.position)

        if heuristic(current_node, goal_node) <= step_size:
            path = []
            while current_node:
                path.append(current_node.position)
                current_node = current_node.parent
            path.append(goal)
            print(path)
            return path[::-1]

        for neighbor in get_neighbors(current_node, obstacles, step_size):
            if neighbor.position in closed_list:
                continue

            neighbor.g = current_node.g + step_size
            neighbor.h = heuristic(neighbor, goal_node)
            neighbor.f = neighbor.g + neighbor.h

            if any(open_node for open_node in open_list if neighbor == open_node and neighbor.g > open_node.g):
                continue

            heapq.heappush(open_list, neighbor)

    return None


def plot_path(start, goal, obstacles, path):
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    plt.grid(True)

    # Plot start and goal
    plt.plot(start[0], start[1], "go", label="Start")
    plt.plot(goal[0], goal[1], "ro", label="Goal")

    # Plot obstacles
    for (ox, oy, width, height) in obstacles:
        rect = plt.Rectangle((ox, oy), width, height, color='r', fill=True)
        ax.add_patch(rect)

    # Plot path
    if path:
        path_x, path_y = zip(*path)
        plt.plot(path_x, path_y, "b-", label="Path")

    plt.legend()
    plt.show()


# Example usage
start = (0, 0)
goal = (9.1, 8.7)
obstacles = [(4, 4, 2, 3)]  # 障碍物列表，形式为 (x, y, width, height)

path = a_star_search(start, goal, obstacles, step_size=0.1)
plot_path(start, goal, obstacles, path)
