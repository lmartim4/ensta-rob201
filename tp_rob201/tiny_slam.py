"""A simple robotics navigation code including SLAM, exploration, planning"""

import numpy as np
from occupancy_grid import OccupancyGrid
from place_bot.entities.lidar import Lidar


class TinySlam:
    """Simple occupancy grid SLAM"""

    def __init__(self, occupancy_grid: OccupancyGrid):
        self.grid = occupancy_grid
        # Origin of the odom frame in the map frame
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

        dist_filtred = sensor_values[lidar_valid_points_mask]
        angles_filtres = ray_angles[lidar_valid_points_mask]

        x, y, theta = pose

        x_world = x + dist_filtred * np.cos(theta + angles_filtres)
        y_world = y + dist_filtred * np.sin(theta + angles_filtres)

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
        """
        Compute corrected pose in map frame
        from raw odom pose + odom frame pose,
        either given as second param or using the ref from the object
        odom : raw odometry position
        odom_pose_ref : optional, origin of the odom frame if given,
                        use self.odom_pose_ref if not given
        """
        if odom_pose_ref is None:
            odom_pose_ref = self.odom_pose_ref

        # Robot Odom
        x0, y0, theta0 = odom_pose
        x0ref, y0ref, theta0ref = odom_pose_ref
        # Robot Absolut
        alpha0 = np.arctan2(y0, x0)
        d0 = np.sqrt(x0**2 + y0**2)

        x = x0ref + d0 * np.cos(theta0ref + alpha0)
        y = y0ref + d0 * np.sin(theta0ref + alpha0)
        theta = theta0ref + theta0

        corrected_pose = np.array([x, y, theta])
        # print(f"corrected_pose() = {corrected_pose}")
        return corrected_pose

    # N number of iterations without improvement
    def localise(self, lidar, raw_odom_pose, N=150):
        """
        Compute the robot position wrt the map,
        and updates the odometry reference
        lidar : placebot object with lidar data
        odom : [x, y, theta] nparray, raw odometry position
        """
        current_correction = self.get_corrected_pose(raw_odom_pose)
        best_score = self._score(lidar, current_correction)
        initial_score = best_score

        current_odom_pos_ref = self.odom_pose_ref
        sigma = [1.5, 1.5, 0.3 * (np.pi / 180.0)]

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
        x, y, t = self.odom_pose_ref

        print(f"Score : {initial_score} -> {best_score}")
        print(f"odom_ref = [{x:.1f}, {y:.1f}, {t:.1f}] : {iterations_count}")

        return best_score

    def update_map(self, lidar, pose):
        """
        Bayesian map update with new observation
        lidar : placebot object with lidar data
        pose : [x, y, theta] nparray, corrected pose in world coordinates
        """

        x = pose[0] + lidar.get_sensor_values() * np.cos(
            pose[2] + lidar.get_ray_angles()
        )
        y = pose[1] + lidar.get_sensor_values() * np.sin(
            pose[2] + lidar.get_ray_angles()
        )

        self.grid.add_map_points(x, y, 2)

        x = pose[0] + (lidar.get_sensor_values() - 10.0) * np.cos(
            pose[2] + lidar.get_ray_angles()
        )
        y = pose[1] + (lidar.get_sensor_values() - 10.0) * np.sin(
            pose[2] + lidar.get_ray_angles()
        )

        for xi, yi in zip(x, y):
            self.grid.add_value_along_line(pose[0], pose[1], xi, yi, -1)

        self.grid.occupancy_map = np.clip(
            self.grid.occupancy_map, -self.grid.max_value, self.grid.max_value
        )
