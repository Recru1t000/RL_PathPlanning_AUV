import matplotlib.pyplot as plt
from matplotlib.patches import Wedge

class explorer():
    def __init__(self,angles,radiues,initial_point):
        self.one = [135,180]
        self.two = [90,135]
        self.three = [45,90]
        self.four = [0,45]
        self.five = [315,360]
        self.six = [270,315]
        self.seven = [225,270]
        self.eight = [180,225]
        self.angles = angles
        self.radiues = radiues
        self.initial_point = initial_point

    def get_angle(self,angles):
        angle_result = []
        for angle in angles:
            if angle==1:
                angle_result.append(self.one)
            elif angle==2:
                angle_result.append(self.two)
            elif angle==3:
                angle_result.append(self.three)
            elif angle==4:
                angle_result.append(self.four)
            elif angle==5:
                angle_result.append(self.five)
            elif angle==6:
                angle_result.append(self.six)
            elif angle==7:
                angle_result.append(self.seven)
            elif angle==8:
                angle_result.append(self.eight)
        return angle_result

    def draw_eighth_circle(self,radiues,angles):
        fig, ax = plt.subplots()
        angles = self.get_angle(angles)
        # 生成八分之一圆形
        for radius,angle in zip(radiues,angles):
            first_angle = angle[0]
            second_angle = angle[1]
            wedge = Wedge((0, 0), radius, first_angle, second_angle, facecolor='blue')
            ax.add_patch(wedge)

        # 设置坐标轴
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(-radius, radius)
        ax.set_ylim(-radius, radius)

        # 显示图形
        plt.show()



