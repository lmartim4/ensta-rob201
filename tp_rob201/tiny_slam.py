""" A simple robotics navigation code including SLAM, exploration, planning"""

import cv2
import numpy as np
from occupancy_grid import OccupancyGrid
from place_bot.entities.lidar import Lidar


class TinySlam:
    """Simple occupancy grid SLAM"""
    
    def __init__(self, occupancy_grid: OccupancyGrid):
        self.grid = occupancy_grid
        self.max_grid_value = 40
        # Origin of the odom frame in the map frame
        self.odom_pose_ref = np.array([0, 0, 0])

    def _score(self, lidar: Lidar, pose):
        """
        Computes the sum of log probabilities of laser end points in the map
        lidar : placebot object with lidar data
        pose : [x, y, theta] nparray, position of the robot to evaluate, in world coordinates
        """
        
        sensor_values = lidar.get_sensor_values()
        ray_angles = lidar.get_ray_angles()
        max_lidar_range = lidar.max_range
        
        lidar_valid_points = sensor_values < max_lidar_range
        
        filtered_values = sensor_values[lidar_valid_points]
        filtered_angles = ray_angles[lidar_valid_points]
        
        x, y, theta = pose
        
        x_world = x + filtered_values * np.cos(theta + filtered_angles)
        y_world = y + filtered_values * np.sin(theta + filtered_angles)
        
        map_coords = self.grid.conv_world_to_map(x_world, y_world)
        log_probs = self.grid.occupancy_map[map_coords]
        
        score = np.sum(log_probs)
        
        return score

    def get_corrected_pose(self, odom_pose, odom_pose_ref=None):
        """
        Compute corrected pose in map frame from raw odom pose + odom frame pose,
        either given as second param or using the ref from the object
        odom : raw odometry position
        odom_pose_ref : optional, origin of the odom frame if given,
                        use self.odom_pose_ref if not given
        """
        # TODO for TP4
        if(odom_pose_ref == None):
            print("should not be none")
        
        ref_x, ref_y, ref_theta = odom_pose_ref
            
        corrected_pose = odom_pose

        return corrected_pose

    def localise(self, lidar, raw_odom_pose, N = 20):
        """
        Compute the robot position wrt the map, and updates the odometry reference
        lidar : placebot object with lidar data
        odom : [x, y, theta] nparray, raw odometry position
        """
        best_score = self._score(lidar, raw_odom_pose)
        best_pose = raw_odom_pose.copy()
        
        sigma = [10, 10, 0.02]
        
        iterations_without_improvement = 0
        
        while iterations_without_improvement < N:
            offset = np.random.normal(0, sigma, 3)
            
            new_pose = raw_odom_pose + offset
            new_score = self._score(lidar, new_pose)
            
            if new_score > best_score:
                best_score = new_score
                best_pose = new_pose.copy()
                iterations_without_improvement = 0
            else:
                iterations_without_improvement += 1
        
        self.odom_ref = best_pose
        return best_score
    
    def update_map(self, lidar, pose):
        """
        Bayesian map update with new observation
        lidar : placebot object with lidar data
        pose : [x, y, theta] nparray, corrected pose in world coordinates
        """
    
        x = pose[0] + lidar.get_sensor_values() * np.cos(pose[2]+lidar.get_ray_angles())
        y = pose[1] + lidar.get_sensor_values() * np.sin(pose[2]+lidar.get_ray_angles())
        
        self.grid.add_map_points(x, y, 10)
        
        x = pose[0] + (lidar.get_sensor_values() - 20.0) * np.cos(pose[2]+lidar.get_ray_angles())
        y = pose[1] + (lidar.get_sensor_values() - 20.0) * np.sin(pose[2]+lidar.get_ray_angles())
                
        for xi, yi in zip(x, y):
            self.grid.add_value_along_line(pose[0], pose[1], xi, yi, -1)
        
        self.grid.occupancy_map = np.clip(self.grid.occupancy_map, -self.max_grid_value, self.max_grid_value)
        
        
        
        