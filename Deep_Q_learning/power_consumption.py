class Power_Consumption():
    def __init__(self,radius,power_consumption_value):
        self.radius = radius
        self.power_consumption_value = power_consumption_value
        self.power_consumption_full_sensing = self.radius*self.power_consumption_value

    def explorer_power_consumption(self,angle_and_radius):
        reduce_power = 0
        for key,value in angle_and_radius.items():
            reduce_power =reduce_power + self.power_consumption_value * value/self.radius
        return reduce_power

    def edge_explorer_power_consumption_full(self,edge_angle_and_radius):
        for angle_and_radius in edge_angle_and_radius:
            angle_and_radius[1] = angle_and_radius[1]*self.power_consumption_value/self.radius
            angle_and_radius[2] = angle_and_radius[2]
        return edge_angle_and_radius

    def set_power_consumption_value(self,power_consumption_value):
        self.power_consumption_value = power_consumption_value

    def get_power_consumption_full_sensing(self):
        return self.power_consumption_full_sensing

    def has_explored_edge(self,distance):
        reduce_power = self.power_consumption_value*distance/self.radius
        return reduce_power