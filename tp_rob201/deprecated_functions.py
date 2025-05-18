########## MyRobotSlam.py ###############


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

############# Control.py #############

# def reactive_obst_avoid(lidar):
#     laser_dist = lidar.get_sensor_values()

#     threshold_distance = 30

#     if laser_dist[180] < threshold_distance:
#         rotation_angle = random.uniform(-1, 1)
#         speed = 0.0
#         rotation_speed = rotation_angle
#     else:
#         speed = 0.2
#         rotation_speed = 0.0

#     command = {"forward": speed, "rotation": rotation_speed}

#     return command

######################################