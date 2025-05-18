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

def lidar_to_world_coords(pose, lidar : Lidar, rayoffset=0):
    return pose_radial_to_word(pose, lidar.get_sensor_values(), lidar.get_ray_angles(), ranges_offset=rayoffset)

def pose_radial_to_word(pose, ranges, angles, ranges_offset=0):
    ray_angles_world = pose[2] + angles
    x_world = pose[0] + (ranges + ranges_offset) * np.cos(ray_angles_world)
    y_world = pose[1] + (ranges + ranges_offset) * np.sin(ray_angles_world)
    return x_world, y_world

def lidar_to_world_batch(lidar, poses):
    """
    Convert lidar readings to world coordinates for multiple poses at once
    
    Parameters:
    lidar : placebot object with lidar data
    poses : [N, 3] array of poses to evaluate
    
    Returns:
    x_world, y_world : tuple of [N, num_rays] arrays
    """
    sensor_values = lidar.get_sensor_values()
    ray_angles = lidar.get_ray_angles()
    
    # Filter for valid points
    max_lidar_range = lidar.max_range
    valid_mask = sensor_values < max_lidar_range
    distances = sensor_values[valid_mask]
    angles = ray_angles[valid_mask]
    
    # Expand dimensions for broadcasting
    distances = distances.reshape(1, -1)  # [1, num_valid_rays]
    angles = angles.reshape(1, -1)        # [1, num_valid_rays]
    
    # Extract pose components and reshape
    x = poses[:, 0].reshape(-1, 1)        # [N, 1]
    y = poses[:, 1].reshape(-1, 1)        # [N, 1]
    theta = poses[:, 2].reshape(-1, 1)    # [N, 1]
    
    # Compute world coordinates for all poses at once
    ray_angles_world = theta + angles     # [N, num_valid_rays]  
    x_world = x + distances * np.cos(ray_angles_world)  # [N, num_valid_rays]
    y_world = y + distances * np.sin(ray_angles_world)  # [N, num_valid_rays]
    
    return x_world, y_world