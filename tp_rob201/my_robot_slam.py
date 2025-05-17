"""
Robot controller definition
Complete controller including SLAM, planning, path following
"""

import numpy as np
import random

from place_bot.entities.robot_abstract import RobotAbstract
from place_bot.entities.odometer import OdometerParams
from place_bot.entities.lidar import LidarParams

from tiny_slam import TinySlam

from control import potential_field_control, reactive_obst_avoid, has_arrived
from occupancy_grid import OccupancyGrid
from planner import Planner


# Definition of our robot controller
class MyRobotSlam(RobotAbstract):
    """A robot controller including SLAM, path planning and path following"""

    def __init__(
        self,
        lidar_params: LidarParams = LidarParams(),
        odometer_params: OdometerParams = OdometerParams(),
    ):
        # Passing parameter to parent class
        super().__init__(
            should_display_lidar=False,
            lidar_params=lidar_params,
            odometer_params=odometer_params,
        )

        # Init SLAM object
        # Here we cheat to get an occupancy grid size that's not too large,
        # by using the robot's starting position and the maximum map size
        # that we shouldn't know.

        size_area = (1400, 1000)
        robot_position = (0.0, 0)
        self.occupancy_grid = OccupancyGrid(
            x_min=-(size_area[0] / 2 + robot_position[0]),
            x_max=size_area[0] / 2 - robot_position[0],
            y_min=-(size_area[1] / 2 + robot_position[1]),
            y_max=size_area[1] / 2 - robot_position[1],
            resolution=2,
        )

        self.tiny_slam = TinySlam(self.occupancy_grid)
        self.planner = Planner(self.occupancy_grid)

        # storage for pose after localization
        self.corrected_pose_in_map = np.array([0, 0, 0])

        self.current_waypoint_index = 0
        self.waypoints = [
            (200.0, 180.0, 0.0),
            (300.0, 200.0, 0.0),
            (450.0, 270.0, 0.0),
            (450.0, -100.0, 0.0),
            (450.0, -300.0, 0.0),
            (300, -100, 0),
            (300, -70, 0),
            (0, -70, 0),
            (0, 180, 0),
            (-310, 180, 0),
            (-300, -70, 0),
            (-380, -60, 0),
            (-410, 100, 0),
            # After this point it should use A* to go somewhere else.
        ]

        # Set this target so control algorithms will try to reach it.
        self.target = self.waypoints[0]

        # Store path here to display on map later.
        self.current_path = None

        # Used to know if we are at the start of the simulation.
        # We should know to update the mapwith bad scores
        self.tick_count = 0

        # Legacy variable to TP1
        self.last_rotation = 0

    def slam_tick(self):
        odometer_pose = self.odometer_values()
        lidar = self.lidar()
        best_score = self.tiny_slam.localise(lidar, odometer_pose)

        self.corrected_pose_in_map = self.tiny_slam.get_corrected_pose(
            odometer_pose)

        if best_score > 5000 or self.tick_count < 50:
            self.tiny_slam.update_map(lidar, self.corrected_pose_in_map)

    def map_tick(self):
        self.tick_count += 1
        if self.tick_count % 10 == 0:
            trajectory = None
            if self.current_path is not None and len(self.current_path) > 0:
                path_points = np.array(self.current_path)
                world_points = np.array(
                    [
                        self.tiny_slam.grid.conv_map_to_world(
                            point[0], point[1])
                        for point in path_points
                    ]
                )
                trajectory = np.array(
                    [
                        world_points[:, 0],  # All x coordinates
                        world_points[:, 1],  # All y coordinates
                    ]
                )

            self.tiny_slam.grid.display_cv(
                self.corrected_pose_in_map, self.target, trajectory
            )

    def control(self):
        """
        Main control function executed at each time step
        """
        self.slam_tick()
        self.map_tick()

        command = self.control_tp2()
        return command

    def control_tp1(self):
        """
        Control function for TP1
        Control funtion with minimal random motion
        """

        if self.tick_count > 0:
            self.tick_count -= 1

            command = {"forward": 0.5, "rotation": self.last_rotation}
        else:
            command = reactive_obst_avoid(self.lidar())
            if command["rotation"] != 0:
                self.last_rotation = command["rotation"]
                self.tick_count = 30

        return command

    def control_tp2(self):
        pose = self.tiny_slam.get_corrected_pose(self.odometer_values())

        if has_arrived(pose, self.target):
            self.next_waypoint()

        return potential_field_control(self.lidar(), pose, self.target)

    def next_waypoint(self):
        self.current_waypoint_index = (self.current_waypoint_index + 1) % len(
            self.waypoints
        )
        self.target = self.waypoints[self.current_waypoint_index]

    def choose_random_goal(self, pose, lidar):
        """
        Choose a random goal based on one of the lidar readings
        """
        angles = lidar.get_ray_angles()
        distances = lidar.get_sensor_values()

        random_index = random.randint(0, len(distances) - 1)
        random_angle = angles[random_index]
        random_distance = distances[random_index]

        safe_distance = random_distance - 30.0

        x0, y0, theta0 = pose

        x = x0 + safe_distance * np.cos(theta0 + random_angle)
        y = y0 + safe_distance * np.sin(theta0 + random_angle)

        self.target = [x, y, 0]
