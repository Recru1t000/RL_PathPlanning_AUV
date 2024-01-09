class Obstacle():
    def __init__(self):
        self.obstacles = []

    def add_obstacle(self, obstacle):
        for i in obstacle:
            self.obstacles.append(i)

    def get_obstacle(self):
        return self.obstacles
