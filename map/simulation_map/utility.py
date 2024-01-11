import math
import sympy
from sympy import symbols,acos,pi,sin


class Collision():
    def __init__(self, x_init, y_init, obstacles, r, angles):
        """
        :param x_init: x的初始点
        :param y_init: y的初始点
        :param obstacles: 所有障碍的集合[(x1,y1),(x2,y1),(x2,y2),(x1,y2),(x1,y1)]
        :param r: 扇形的半径
        :return:
        """
        self.collision_points = []
        self.x_init = x_init
        self.y_init = y_init
        self.obstacles = obstacles
        self.x1 = obstacles[0][0]
        self.x2 = obstacles[1][0]
        self.y1 = obstacles[0][1]
        self.y2 = obstacles[2][1]
        self.r = r
        self.angle1 = angles[0]
        self.angle2 = angles[1]
        self.circle_up_bottom = True  # 因为arccos的y区间为[0,π],所以下半圆的需要360减θ
        if angles[0] >= 180:
            self.circle_up_bottom = False

    def left_side(self):
        if abs(self.x1-self.x_init)>self.r:
            print("左边不相交")
            return
        y_chazhi = math.sqrt(self.r**2-(self.x1-self.x_init)**2)#算出y的距离

        if self.circle_up_bottom:
            y = self.y_init+y_chazhi
            xita = math.acos((self.x1 - self.x_init) / self.r)*180/math.pi  # 求出xita的弧度#x也就正负一说,self.x1 - self.x_init不能加绝对值
        else:
            y = self.y_init-y_chazhi
            xita = 360-math.acos((self.x1 - self.x_init) / self.r)*180/math.pi  # 求出xita的弧度

        if(self.angle1 <= xita <= self.angle2 and self.y1<=y<=self.y2):
            self.collision_points.append((self.x1,y))
            print("左边相交")
        else:
            print("左边不相交")

    def right_side(self):
        if abs(self.x2-self.x_init)>self.r:
            print("右边不相交")
            return
        y_chazhi = math.sqrt(self.r**2-(self.x2-self.x_init)**2)#算出y的距离
        if self.circle_up_bottom:
            y = self.y_init+y_chazhi
            xita = math.acos((self.x2 - self.x_init) / self.r)*180/math.pi  # 求出xita的弧度
        else:
            y = self.y_init-y_chazhi
            xita = 360-math.acos((self.x2 - self.x_init) / self.r)*180/math.pi  # 求出xita的弧度

        if(self.angle1 <= xita <= self.angle2 and self.y1<=y<=self.y2):
            self.collision_points.append((self.x2,y))
            print("右边相交")
        else:
            print("右边不相交")

    def up_side(self):
        if abs(self.y2-self.y_init)>self.r:
            print("上边不相交")
            return
        x_chazhi = math.sqrt(self.r**2-(self.y2-self.y_init)**2)#算出y的距离
        if self.circle_up_bottom:
            if 0<=self.angle1<90:
                x = self.x_init+x_chazhi
                xita = math.asin((self.y2 - self.y_init) / self.r)*180/math.pi  # 求出xita的弧度
            else:
                x = self.x_init-x_chazhi
                xita = 180-math.asin((self.y2 - self.y_init) / self.r)*180/math.pi  # 求出xita的弧度
        else:
            if 270 <= self.angle1 < 360:
                x = self.x_init+x_chazhi
                xita = 360+math.asin((self.y2 - self.y_init) / self.r)*180/math.pi  # 求出xita的弧度
            else:
                x = self.x_init-x_chazhi
                xita = 180-math.asin((self.y2 - self.y_init) / self.r)*180/math.pi  # 求出xita的弧度
                print(math.asin((self.y2 - self.y_init) / self.r)*180/math.pi)
        if(self.angle1 <= xita <= self.angle2 and self.x1<=x<=self.x2):
            self.collision_points.append((x,self.y2))
            print("上边相交")
        else:
            print("上边不相交")

    def bottom_side(self):
        if abs(self.y1-self.y_init)>self.r:
            print("下边不相交")
            return
        x_chazhi = math.sqrt(self.r**2-(self.y1-self.y_init)**2)#算出y的距离
        if self.circle_up_bottom:
            if 0<=self.angle1<90:
                x = self.x_init+x_chazhi
                xita = math.asin((self.y1 - self.y_init) / self.r)*180/math.pi  # 求出xita的弧度
            else:
                x = self.x_init-x_chazhi
                xita = 180-math.asin((self.y1 - self.y_init) / self.r)*180/math.pi  # 求出xita的弧度
        else:
            if 270 <= self.angle1 < 360:
                x = self.x_init+x_chazhi
                xita = 360+math.asin((self.y1 - self.y_init) / self.r)*180/math.pi  # 求出xita的弧度
            else:
                x = self.x_init-x_chazhi
                xita = 180-math.asin((self.y1 - self.y_init) / self.r)*180/math.pi  # 求出xita的弧度
                print(math.asin((self.y1 - self.y_init) / self.r)*180/math.pi)
        if(self.angle1 <= xita <= self.angle2 and self.x1<=x<=self.x2):
            self.collision_points.append((x,self.y1))
            print("下边相交")
        else:
            print("下边不相交")

    def straight_line(self,angel):
        if(angel==0 or angel==360 or angel==180):
            canshu = 1
            if angel==180:#如果为0或360则说明是向右找点，所以该点必然大于初始点，向左则说明，小于所以设定该参数判断
                canshu=-1
            if(abs(self.x1-self.x_init)<=self.r and canshu*self.x1>=canshu*self.x_init):
                self.collision_points.append((self.x1,self.y_init))
            if(abs(self.x2-self.x_init)<=self.r and canshu*self.x2>=canshu*self.x_init):
                self.collision_points.append((self.x2,self.y_init))

        if(angel==45 or angel==225):#如果是45则说明，x和y都大于初始点，如果是225则说明都小于初始点
            a = self.y_init - self.x_init
            canshu = 1
            if angel==225:
                canshu = -1
            y_left = self.x1+a
            y_right = self.x2+a
            x_bottmo = self.y1-a
            x_up = self.y2-a
            #以下需要判断三点，1.求出来的点是否在半径之内，2.该点是否在矩形的边界之内，3.判断是向45还是向225，如果是45则所有点大于init否则小于init
            if (y_left-self.y_init)**2+(self.x1-self.x_init)**2<=self.r**2 and self.y1<=y_left<=self.y2 and canshu*self.x1>=canshu*self.x_init:#不用小于等于是因为弧线上的节点已经在上面被包含了
                self.collision_points.append((self.x1, y_left))
            if (y_right-self.y_init)**2+(self.x2-self.x_init)**2<=self.r**2 and self.y1<=y_right<=self.y2 and canshu*self.x2>=canshu*self.x_init:
                self.collision_points.append((self.x2, y_right))
            if (x_bottmo-self.x_init)**2+(self.y1-self.y_init)**2<=self.r**2 and self.x1<=x_bottmo<=self.x2 and canshu*self.y1>=canshu*self.y_init:
                self.collision_points.append((x_bottmo, self.y1))
            if (x_up-self.x_init)**2+(self.y2-self.y_init)**2<=self.r**2 and self.x1<=x_up<=self.x2 and canshu*self.y2>=canshu*self.y_init:
                self.collision_points.append((x_up, self.y2))

        if(angel==90 or angel==270):
            canshu = 1
            if angel==270:#如果为90则说明是向上找点，所以该点必然大于初始点，向下则说明，小于所以设定该参数判断
                canshu=-1
            if(self.y1-self.y_init<=self.r and canshu*self.y1>=canshu*self.y_init):
                self.collision_points.append((self.x_init,self.y1))
            if(self.y2-self.y_init<=self.r and canshu*self.y2>=canshu*self.y_init):
                self.collision_points.append((self.x_init,self.y2))

        if(angel==135 or angel==315):#如果是135则说明，x小于初始点,y大于初始点，如果是315则说明,x大于初始点,y小于初始点
            a = self.y_init - self.x_init
            canshu = 1
            if angel==315:
                canshu = -1
            y_left = -self.x1+a
            y_right = -self.x2+a
            x_bottmo = -self.y1+a
            x_up = -self.y2+a
            #以下需要判断三点，1.求出来的点是否在半径之内，2.该点是否在矩形的边界之内，3.判断是向135还是向315
            if (y_left-self.y_init)**2+(self.x1-self.x_init)**2<=self.r**2 and self.y1<=y_left<=self.y2 and canshu*self.x1<=canshu*self.x_init:#不用小于等于是因为弧线上的节点已经在上面被包含了
                self.collision_points.append((self.x1, y_left))
            if (y_right-self.y_init)**2+(self.x2-self.x_init)**2<=self.r**2 and self.y1<=y_right<=self.y2 and canshu*self.x2<=canshu*self.x_init:
                self.collision_points.append((self.x2, y_right))
            if (x_bottmo-self.x_init)**2+(self.y1-self.y_init)**2<=self.r**2 and self.x1<=x_bottmo<=self.x2 and canshu*self.y1>=canshu*self.y_init:
                self.collision_points.append((x_bottmo, self.y1))
            if (x_up-self.x_init)**2+(self.y2-self.y_init)**2<=self.r**2 and self.x1<=x_up<=self.x2 and canshu*self.y2>=canshu*self.y_init:
                self.collision_points.append((x_up, self.y2))
    #todo 两条线，和线的终点是否在障碍中

    def rectangle_point(self,obstacle):
        #分不同的象限来做
        x = obstacle[0]
        y = obstacle[1]
        x2 = (self.x_init-x)**2
        y2 = (self.y_init - y)**2
        theta = math.atan2(self.y_init - y, self.x_init - x)
        theta_degrees = math.degrees(theta)
        if x2+y2<=self.r**2 and self.angle1<=theta_degrees<=self.angle2:
            self.collision_points.append((x,y))

    def get_collision_points(self):
        self.left_side()
        self.right_side()
        self.up_side()
        self.bottom_side()
        self.straight_line(self.angle1)
        self.straight_line(self.angle2)
        self.collision_points = list(set(self.collision_points))
        self.collision_points.append(self.collision_points[0])
        for obstacle in self.obstacles:
            self.rectangle_point(obstacle)
        return self.collision_points
