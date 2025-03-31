""" A set of robotics control functions """

import random
import numpy as np


def reactive_obst_avoid(lidar):
    """
    Simple obstacle avoidance
    lidar : placebot object with lidar data
    """
    # TODO for TP1
    
    laser_dist = lidar.get_sensor_values()
    laser_angles = lidar.get_ray_angles()

    #xo_ref = laser_dist * np.cos(laser_angles)
    #yo_ref = laser_dist * np.sin(laser_angles)

    #print(xo_ref[180], yo_ref[180])
    
    threshold_distance = 30  # Seuil de détection d'un obstacle (par exemple, 50 cm)

    if laser_dist[180] < threshold_distance:
        rotation_angle = random.uniform(-1, 1)
        speed = 0.0
        rotation_speed = rotation_angle
    else:
        speed = 0.2  # Avancer à une vitesse constante
        rotation_speed = 0.0  # Pas de rotation, on va tout droit

    command = {"forward": speed,
               "rotation": rotation_speed}

    return command


def potential_field_control(lidar, current_pose, goal_pose):
    """
    Control using potential field for goal reaching and obstacle avoidance
    lidar : placebot object with lidar data
    current_pose : [x, y, theta] nparray, current pose in odom or world frame
    goal_pose : [x, y, theta] nparray, target pose in odom or world frame
    Notes: As lidar and odom are local only data, goal and gradient will be defined either in
    robot (x,y) frame (centered on robot, x forward, y on left) or in odom (centered / aligned
    on initial pose, x forward, y on left)
    """
    # TODO for TP2

    command = {"forward": 0,
               "rotation": 0}

    return command
