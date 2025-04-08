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
    
    grad_atractive = calculate_atractive_grad(current_pose, goal_pose, d_lim=3, K_goal=0.1)
    grad_repulsive = calculate_repulsive_grad(lidar, current_pose, k_obs=1, d_safe=50)
    
    grad_r = grad_atractive  - grad_repulsive
    
    print(f"Front distance: {lidar.get_sensor_values()[180]}")
    print(f"Attractive force: {np.linalg.norm(grad_atractive)}")
    print(f"Repulsive force: {np.linalg.norm(grad_repulsive)}")    
    
    print("atrc: ", grad_atractive)
    print("repul: ", grad_repulsive)
    print("grad_r", grad_r)
    
    
    
    forward_speed = 0.1 #np.sqrt(grad_r[0]**2 + grad_r[1]**2)
    rotation_speed = calculate_rotation_speed(grad_r, current_pose, Kv=1)
    print(rotation_speed)
    
    return {"forward":  np.clip(forward_speed, a_min=-1, a_max=1),
            "rotation": np.clip(rotation_speed, a_min=-1, a_max=1)}

def calculate_atractive_grad(current_pose, goal_pose, d_lim, K_goal):
    goal_minus_current = goal_pose - current_pose
    dist = np.sqrt(goal_minus_current[0]**2 + goal_minus_current[1]**2) 
    
    print(f"Error: ({dist}) - {goal_minus_current}")
    
    if(dist <= d_lim):
        grad_f = K_goal*goal_minus_current / d_lim
        return np.array([0,0,0])
        #return grad_f
    else:
        grad_f = K_goal * (goal_minus_current) / dist
    
    return grad_f

def calculate_rotation_speed(grad_r, current_pose, Kv):
    target_angle = np.atan2(-grad_r[1], grad_r[0])
    angle_error = (target_angle - current_pose[2])
    
    print(f"target: {target_angle*180/np.pi} current = {current_pose[2]*180/np.pi} error= {angle_error*180/np.pi}")
    
    if(np.abs(angle_error)*180/np.pi < 2):
        return 0
    else:
        return -Kv*(angle_error)
      
def calculate_repulsive_grad(lidar, current_pose, k_obs, d_safe):
    if d_safe == 0:
        raise ValueError("d_safe should not be 0!")
    
    gradient = np.zeros(3)
    
    for (l, angle) in zip(lidar.get_sensor_values(), lidar.get_ray_angles()):
        if l == 0:
            print("Lidar received a 0 ???")
        
        if l < d_safe:
            q_obs = np.array([l * np.cos(angle), l * np.sin(angle), 0])
            gradient += k_obs * (1 / l - 1 / d_safe) * (q_obs - current_pose)
    
    return gradient

        