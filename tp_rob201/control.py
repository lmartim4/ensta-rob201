"""A set of robotics control functions"""

import random

import numpy as np


def has_arrived(pose, goal, dlim=20):
    return np.linalg.norm(pose - goal) < dlim


def reactive_obst_avoid(lidar):
    """
    Simple obstacle avoidance
    lidar : placebot object with lidar data
    """
    laser_dist = lidar.get_sensor_values()

    threshold_distance = 30

    if laser_dist[180] < threshold_distance:
        rotation_angle = random.uniform(-1, 1)
        speed = 0.0
        rotation_speed = rotation_angle
    else:
        speed = 0.2
        rotation_speed = 0.0

    command = {"forward": speed, "rotation": rotation_speed}

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

    grad_atractive = calculate_atractive_grad(
        current_pose, goal_pose, d_lim=10, K_goal=0.5
    )

    grad_repulsive = calculate_repulsive_grad(
        lidar, current_pose, k_obs=100, d_safe=30)

    grad_r = grad_atractive - grad_repulsive

    # print(grad_r, " ", grad_atractive, " ", grad_repulsive)

    forward_speed = np.linalg.norm(grad_r)

    if forward_speed < 0.001:
        rotation_speed = 0
    else:
        rotation_speed = calculate_rotation_speed(grad_r, current_pose, Kv=0.2)

    if rotation_speed > 0.03:
        forward_speed = 0

    if abs(grad_atractive[0]) < 0.001 and abs(rotation_speed) < 0.001:
        return {"forward": 0, "rotation": 0}

    return {
        "forward": np.clip(forward_speed, a_min=-0.4, a_max=0.4),
        "rotation": np.clip(rotation_speed, a_min=-0.4, a_max=0.4),
    }


def calculate_atractive_grad(current_pose, goal_pose, d_lim, K_goal):
    diff = goal_pose - current_pose
    dist = np.linalg.norm(diff)

    if dist <= d_lim:
        grad_f = K_goal * diff / d_lim
        return np.array([0, 0, 0])
    else:
        grad_f = K_goal * (diff) / dist

    return grad_f


def calculate_rotation_speed(grad_r, current_pose, Kv):
    target_angle = np.atan2(grad_r[1], grad_r[0])
    angle_error = target_angle - current_pose[2]
    angle_error = np.arctan2(np.sin(angle_error), np.cos(angle_error))

    # print("angle_error: ", angle_error)

    if np.abs(angle_error) * 180 / np.pi < 5:
        return 0
    else:
        return Kv * (angle_error)


def calculate_repulsive_grad(lidar, current_pose, k_obs, d_safe):
    distances = np.array(lidar.get_sensor_values())
    angles = np.array(lidar.get_ray_angles())

    mask = distances < d_safe

    filtered_distances = distances[mask]
    filtered_angles = angles[mask]

    x_positions = filtered_distances * np.cos(filtered_angles)
    y_positions = filtered_distances * np.sin(filtered_angles)

    q_obs = np.column_stack(
        (x_positions, y_positions, np.zeros_like(filtered_distances))
    )

    scalar_factors = (k_obs / (filtered_distances**3)) * (
        1 / filtered_distances - 1 / d_safe
    )

    gradient_components = q_obs * scalar_factors[:, np.newaxis]

    gradient = np.sum(gradient_components, axis=0)

    return gradient
