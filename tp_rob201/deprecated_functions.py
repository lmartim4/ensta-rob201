########## MyRobotSlam ###############


# def control_tp1(self):
#     if not hasattr(self, "last_rotation"):
#         self.last_rotation = 0

#     if self.tick_count > 0:
#         self.tick_count -= 1

#         command = {"forward": 0.5, "rotation": self.last_rotation}
#     else:
#         command = reactive_obst_avoid(self.lidar())
#         if command["rotation"] != 0:
#             self.last_rotation = command["rotation"]
#             self.tick_count = 30

#     return command


# def choose_random_goal(self, pose, lidar):
#     angles = lidar.get_ray_angles()
#     distances = lidar.get_sensor_values()

#     random_index = random.randint(0, len(distances) - 1)
#     random_angle = angles[random_index]
#     random_distance = distances[random_index]

#     safe_distance = random_distance - 30.0

#     x0, y0, theta0 = pose

#     x = x0 + safe_distance * np.cos(theta0 + random_angle)
#     y = y0 + safe_distance * np.sin(theta0 + random_angle)

#     self.target = [x, y, 0]

######################################

