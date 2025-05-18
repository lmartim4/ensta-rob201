"""A simple robotics navigation code including SLAM, exploration, planning"""

import utils
import numpy as np
from occupancy_grid import OccupancyGrid
from place_bot.entities.lidar import Lidar
    
class TinySlam:
    """Simple occupancy grid SLAM"""

    def __init__(self, occupancy_grid: OccupancyGrid):
        self.grid = occupancy_grid
        self.odom_pose_ref = np.array([0.0, 0.0, 0.0])

    def _score(self, lidar: Lidar, pose):
        sensor_values = lidar.get_sensor_values()
        ray_angles = lidar.get_ray_angles()

        filter_mask = sensor_values < lidar.max_range

        ranges = sensor_values[filter_mask]
        angles = ray_angles[filter_mask]

        x_world,y_world = utils.pose_radial_to_word(pose, ranges, angles)
        map_coords = self.grid.conv_world_to_map(x_world, y_world)

        valid_mask = (
            (map_coords[0] >= 0)
            & (map_coords[0] < self.grid.x_max_map)
            & (map_coords[1] >= 0)
            & (map_coords[1] < self.grid.y_max_map)
        )

        valid_x = map_coords[0][valid_mask]
        valid_y = map_coords[1][valid_mask]

        valid_x = valid_x.astype(int)
        valid_y = valid_y.astype(int)

        log_probs = self.grid.occupancy_map[valid_x, valid_y]

        score = np.sum(log_probs)

        return score
    
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
        x, y = utils.pose_lidar_to_world_coords(pose, lidar)

        self.grid.add_map_points(x, y, 2)

        x, y = utils.pose_lidar_to_world_coords(pose, lidar, -10.0)

        for xi, yi in zip(x, y):
            self.grid.add_value_along_line(pose[0], pose[1], xi, yi, -1)

        np.clip(self.grid.occupancy_map, -self.grid.max_value, self.grid.max_value, out=self.grid.occupancy_map)