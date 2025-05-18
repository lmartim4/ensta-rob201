import numpy as np
from place_bot.entities.lidar import Lidar

def polar_to_cartesian(r, theta):
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y

def cartesian_to_polar(x, y):
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return r, theta

def pose_lidar_to_world_coords(pose, lidar : Lidar, rayoffset=0):
    return pose_radial_to_word(pose, lidar.get_sensor_values(), lidar.get_ray_angles(), ranges_offset=rayoffset)

def pose_radial_to_word(pose, ranges, angles, ranges_offset=0):
    ray_angles_world = pose[2] + angles
    x_world = pose[0] + (ranges + ranges_offset) * np.cos(ray_angles_world)
    y_world = pose[1] + (ranges + ranges_offset) * np.sin(ray_angles_world)
    return x_world, y_world