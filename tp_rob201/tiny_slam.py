"""A simple robotics navigation code including SLAM, exploration, planning"""

from numba import jit  # For JIT compilation
import utils
import numpy as np
from occupancy_grid import OccupancyGrid
from place_bot.entities.lidar import Lidar

@jit(nopython=True)
def _score_vectorized(occupancy_map, map_coords_x, map_coords_y, x_max_map, y_max_map):
    valid_mask = (
        (map_coords_x >= 0)
        & (map_coords_x < x_max_map)
        & (map_coords_y >= 0)
        & (map_coords_y < y_max_map)
    )
    valid_x = map_coords_x[valid_mask].astype(np.int32)
    valid_y = map_coords_y[valid_mask].astype(np.int32)
    
    # Replace advanced indexing with a loop
    total_score = 0.0
    for i in range(len(valid_x)):
        total_score += occupancy_map[valid_x[i], valid_y[i]]
    
    return total_score
    
    log_probs = occupancy_map[valid_x, valid_y]
    return np.sum(log_probs)
    
class TinySlam:
    """Simple occupancy grid SLAM"""

    def __init__(self, occupancy_grid: OccupancyGrid):
        self.grid = occupancy_grid
        self.odom_pose_ref = np.array([0.0, 0.0, 0.0])

    def _score(self, lidar: Lidar, pose):
        """
        Computes the sum of log probabilities of laser end points in the map
        lidar : placebot object with lidar data
        pose : [x, y, theta] nparray, position of
        the robot to evaluate, in world coordinates
        """
        sensor_values = lidar.get_sensor_values()
        ray_angles = lidar.get_ray_angles()
        max_lidar_range = lidar.max_range
        lidar_valid_points_mask = sensor_values < max_lidar_range
        dist_filtered = sensor_values[lidar_valid_points_mask]
        angles_filtered = ray_angles[lidar_valid_points_mask]
        x_world, y_world = utils.pose_radial_to_word(pose, dist_filtered, angles_filtered)
        map_coords = self.grid.conv_world_to_map(x_world, y_world)
        
        # Use the standalone function for scoring
        return _score_vectorized(
            self.grid.occupancy_map,
            map_coords[0],
            map_coords[1],
            self.grid.x_max_map, 
            self.grid.y_max_map
        )
    
    def get_corrected_pose(self, odom_pose, odom_pose_ref=None):
        if odom_pose_ref is None:
            odom_pose_ref = self.odom_pose_ref

        x0, y0, theta0 = odom_pose
        
        alpha0 = np.arctan2(y0, x0)
        d0 = np.sqrt(x0**2 + y0**2)

        x,y = utils.pose_radial_to_word(odom_pose_ref, d0, alpha0)
        theta = odom_pose_ref[2] + theta0

        corrected_pose = np.array([x, y, theta])
        return corrected_pose

    def localise(self, lidar, raw_odom_pose, N=150, debug=False):
        current_correction = self.get_corrected_pose(raw_odom_pose)
        best_score = self._score(lidar, current_correction)
        initial_score = best_score
        
        current_odom_pos_ref = self.odom_pose_ref
        sigma = [1.5, 1.5, 0.4 * (np.pi / 180.0)]

        iterations_without_improvement = 0
        iterations_count = 0

        while iterations_without_improvement < N:
            iterations_count += 1
            offset = np.random.normal(0, sigma, 3)

            ref_offset = current_odom_pos_ref + offset
            new_pose = self.get_corrected_pose(raw_odom_pose, ref_offset)
            new_score = self._score(lidar, new_pose)

            if new_score > best_score:
                best_score = new_score
                self.odom_pose_ref = ref_offset
                iterations_without_improvement = 0
            else:
                iterations_without_improvement += 1

        iterations_count -= N
        
        if debug:
            x, y, t = self.odom_pose_ref
            print(f"Score : {initial_score} -> {best_score}")
            print(f"odom_ref = [{x:.1f}, {y:.1f}, {t:.1f}] : {iterations_count}")

        return best_score

    def update_map(self, lidar, pose):
        x, y = utils.lidar_to_world_coords(pose, lidar)

        self.grid.add_map_points(x, y, 2)

        x, y = utils.lidar_to_world_coords(pose, lidar, -10.0)

        for xi, yi in zip(x, y):
            self.grid.add_value_along_line(pose[0], pose[1], xi, yi, -1)

        np.clip(self.grid.occupancy_map, -self.grid.max_value, self.grid.max_value, out=self.grid.occupancy_map)
        
    def localise_optimized(self, lidar, raw_odom_pose, N=300, batch_size=50, debug=False):
        """Optimized version that processes batches of poses at once"""
        current_correction = self.get_corrected_pose(raw_odom_pose)
        best_score = self._score(lidar, current_correction)
        initial_score = best_score
        current_odom_pos_ref = self.odom_pose_ref.copy()
        sigma = np.array([1.5, 1.5, 0.4 * (np.pi / 180.0)])
        
        iterations_without_improvement = 0
        iterations_count = 0
        
        # Pre-filter lidar data once
        sensor_values = lidar.get_sensor_values()
        ray_angles = lidar.get_ray_angles()
        max_lidar_range = lidar.max_range
        valid_mask = sensor_values < max_lidar_range
        distances = sensor_values[valid_mask]
        angles = ray_angles[valid_mask]
        
        while iterations_without_improvement < N:
            # Generate batch_size random poses at once
            offsets = np.random.normal(0, sigma, (batch_size, 3))
            ref_offsets = current_odom_pos_ref + offsets
            
            # Get corrected poses (vectorized)
            batch_poses = np.zeros((batch_size, 3))
            for i, ref in enumerate(ref_offsets):
                batch_poses[i] = self.get_corrected_pose(raw_odom_pose, ref)
            
            # Calculate scores for all poses at once
            x_worlds, y_worlds = utils.lidar_to_world_batch(lidar, batch_poses)
            scores = np.zeros(batch_size)
            
            for i in range(batch_size):
                map_coords = self.grid.conv_world_to_map(x_worlds[i], y_worlds[i])
                scores[i] = _score_vectorized(
                    self.grid.occupancy_map, 
                    map_coords[0], map_coords[1],
                    self.grid.x_max_map, self.grid.y_max_map
                )
            
            # Find best pose in batch
            best_in_batch = np.argmax(scores)
            if scores[best_in_batch] > best_score:
                best_score = scores[best_in_batch]
                self.odom_pose_ref = ref_offsets[best_in_batch]
                iterations_without_improvement = 0
            else:
                iterations_without_improvement += batch_size
            
            iterations_count += batch_size
        
        if debug:
            x, y, t = self.odom_pose_ref
            print(f"Score : {initial_score} -> {best_score}")
            print(f"odom_ref = [{x:.1f}, {y:.1f}, {t:.1f}] : {iterations_count}")
        
        return best_score