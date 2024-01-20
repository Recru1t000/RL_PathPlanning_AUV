import numpy as np
import matplotlib.pyplot as plt


class ArtificialPotentialField:
    def __init__(self, start, goal, obstacles, att_force=1.0, rep_force=100.0, rep_range=5.0):
        self.start = np.array(start)
        self.goal = np.array(goal)
        self.obstacles = obstacles
        self.att_force = att_force
        self.rep_force = rep_force
        self.rep_range = rep_range

    def attractive_force(self, position):
        return self.att_force * (self.goal - position)

    def repulsive_force(self, position):
        rep_force = np.zeros_like(position)
        for obstacle in self.obstacles:
            dist = np.linalg.norm(position - obstacle)
            if dist < self.rep_range:
                rep_force += self.rep_force * ((1 / dist - 1 / self.rep_range) / dist ** 2) * (position - obstacle)
        return rep_force

    def calculate_total_force(self, position):
        att_force = self.attractive_force(position)
        rep_force = self.repulsive_force(position)
        total_force = att_force + rep_force
        return total_force

    def move_robot(self, position, step_size):
        force = self.calculate_total_force(position)
        new_position = position + step_size * force / np.linalg.norm(force)
        return new_position


def plot_environment(apf, robot_pos):
    plt.figure(figsize=(8, 8))

    # Plot obstacles
    for obstacle in apf.obstacles:
        plt.plot(obstacle[0], obstacle[1], 'ro', markersize=10)

    # Plot start and goal
    plt.plot(apf.start[0], apf.start[1], 'go', markersize=10)
    plt.plot(apf.goal[0], apf.goal[1], 'bo', markersize=10)

    # Plot robot position
    plt.plot(robot_pos[0], robot_pos[1], 'yo', markersize=10)

    plt.title('Artificial Potential Field')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.grid(True)
    plt.show()


def main():
    # Define start, goal, and obstacles
    start = [0, 0]
    goal = [10, 10]
    obstacles = [ [2, 7]]

    # Create artificial potential field
    apf = ArtificialPotentialField(start, goal, obstacles)

    # Initialize robot position
    robot_pos = np.array(start)

    # Plot initial environment
    plot_environment(apf, robot_pos)

    # Move the robot towards the goal
    for _ in range(50):
        robot_pos = apf.move_robot(robot_pos, step_size=0.5)
        plot_environment(apf, robot_pos)


if __name__ == "__main__":
    main()
