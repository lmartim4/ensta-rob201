"""A set of robotics control functions"""

from place_bot.entities.odometer import normalize_angle 
import utils
import numpy as np


def has_arrived(pose, goal, dlim=20):
    return np.linalg.norm(pose - goal) < dlim


def potential_field_control(lidar, current_pose, goal_pose):
    grad_atractive = calculate_atractive_grad(
        current_pose, goal_pose, d_lim=20, K_goal=0.4
    )

    grad_repulsive = calculate_repulsive_grad(
        lidar, k_obs=10, d_safe=50
    )

    grad_r = grad_atractive - grad_repulsive
    forward_speed = np.linalg.norm(grad_r)

    if forward_speed < 0.001:
        rotation_speed = 0
    else:
        rotation_speed = calculate_rotation_speed(grad_r, current_pose, Kv=0.8)

    if abs(rotation_speed) > 0.001:
        forward_speed = 0

    if forward_speed < 0.001 and abs(rotation_speed) < 0.001:
        return {
            "forward": 0,
            "rotation": 0
        }

    return {
        "forward": np.clip(forward_speed, a_min=-0.4, a_max=0.4),
        "rotation": np.clip(rotation_speed, a_min=-0.4, a_max=0.4),
    }


def calculate_atractive_grad(current_pose, goal_pose, d_lim, K_goal):
    diff = goal_pose - current_pose
    dist = np.linalg.norm(diff)

    if dist <= d_lim:
        grad_f = K_goal * diff / d_lim
        return grad_f
    else:
        grad_f = K_goal * (diff) / dist

    return grad_f


def calculate_rotation_speed(grad_r, current_pose, Kv):
    target_angle = np.atan2(grad_r[1], grad_r[0])
    angle_error = target_angle - current_pose[2]
    angle_error = normalize_angle(angle_error)

    if np.abs(angle_error) * 180 / np.pi < 5:
        return 0
    else:
        return Kv * (angle_error)


def calculate_repulsive_grad(lidar, k_obs, d_safe):
    raw_dist = np.array(lidar.get_sensor_values())
    raw_angles = np.array(lidar.get_ray_angles())

    filt = raw_dist < d_safe

    filt_ranges = raw_dist[filt]
    filt_angles = raw_angles[filt]

    x_pos,y_pos = utils.polar_to_cartesian(r=filt_ranges,theta=filt_angles)

    q_obs = np.column_stack((x_pos, y_pos, np.zeros_like(filt_ranges)))

    scalar_factors = (k_obs / (filt_ranges**3)) * (
        1 / filt_ranges - 1 / d_safe
    )

    gradient_components = q_obs * scalar_factors[:, np.newaxis]

    gradient = np.sum(gradient_components, axis=0)

    return gradient
