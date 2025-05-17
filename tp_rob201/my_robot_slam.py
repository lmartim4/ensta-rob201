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

        self.enable_slam = True

        # storage for pose after localization
        self.corrected_pose = np.array([0, 0, 0])

        self.goal_index = 0
        self.current_path = None
        self.target = [0, 0, 0]
        self.tick_count = 0
        self.last_rotation = 0

    def slam_tick(self):
        odometer = self.odometer_values()

        if self.enable_slam:
            lidar = self.lidar()
            best_score = self.tiny_slam.localise(lidar, odometer)
            # best_score = self.tiny_slam.score20(best_score)

            print(f"Final Score = {best_score:.1f}")

            self.corrected_pose = self.tiny_slam.get_corrected_pose(odometer)

            if best_score > 5000 or self.tick_count < 50:
                # print("Tick = ", self.tick_count)
                self.tiny_slam.update_map(lidar, self.corrected_pose)
        else:
            self.corrected_pose = odometer

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
                self.corrected_pose, self.target, trajectory)

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
        """
        Control function for TP2
        Main control function with full SLAM,
        random exploration and path planning
        """
        pose = self.odometer_values()
        pose = self.tiny_slam.get_corrected_pose(pose)
        lidar = self.lidar()

        command = potential_field_control(lidar, pose, self.target)

        if has_arrived(pose, self.target):
            self.next_waypoint()
            print(f"Moving to next waypoint: {self.target[:2]}")
        return command

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

    def next_waypoint(self):
        """
        Select the next waypoint from the predefined list
        """
        # If no waypoints list exists yet, initialize it
        if not hasattr(self, "waypoints") or not self.waypoints:
            # Define list of waypoints as (x, y, theta) coordinates
            self.waypoints = [
                (200.0, 200.0, 0.0),
                (300.0, 200.0, 0.0),
                (450.0, 270.0, 0.0),
                (450.0, -100.0, 0.0),
                (450.0, -300.0, 0.0),
                (250, -100, 0),
                (250, -70, 0),
                (0, -70, 0),
                (0, 180, 0),
                (-340, 170, 0),
                (-360, -80, 0),
                (-410, -80, 0),
                (-440, 120, 0),
                # After this point it should use A* to go somewhere else.
                # (0.0, 0.0, 0.0),  # Return to starting position
            ]
            self.current_waypoint_index = 0
        else:
            # Move to the next waypoint
            self.current_waypoint_index = (self.current_waypoint_index + 1) % len(
                self.waypoints
            )

        # Set the target to the next waypoint
        self.target = self.waypoints[self.current_waypoint_index]
