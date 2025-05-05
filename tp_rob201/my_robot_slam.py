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

from control import potential_field_control, reactive_obst_avoid
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

        # step counter to deal with init and display
        self.current_target = [0, -10, 0]
        self.counter = 0
        self.last_rotation = 0
        # Init SLAM object
        # Here we cheat to get an occupancy grid size that's not too large, by using the
        # robot's starting position and the maximum map size that we shouldn't know.
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
        self.corrected_pose = np.array([0, 0, 0])

    def control(self):
        """
        Main control function executed at each time step
        """
        odometer_data = self.odometer_values()

        best_score = self.tiny_slam.localise(self.lidar(), odometer_data)
        note_sur_20 = best_score * 20 / (self.occupancy_grid.max_grid_value * 360)
        print(f"Score = {note_sur_20:.1f}")

        self.corrected_pose = self.tiny_slam.get_corrected_pose(odometer_data)

        if best_score > 14 or self.counter < 20:
            self.tiny_slam.update_map(self.lidar(), self.corrected_pose)

        self.counter += 1

        if self.counter % 10 == 0:
            self.tiny_slam.grid.display_cv(self.corrected_pose, self.current_target)

        control = self.control_tp2()

        return control

    def control_tp1(self):
        """
        Control function for TP1
        Control funtion with minimal random motion
        """

        if self.counter > 0:
            self.counter -= 1

            command = {"forward": 0.5, "rotation": self.last_rotation}
        else:
            command = reactive_obst_avoid(self.lidar())
            if command["rotation"] != 0:
                self.last_rotation = command["rotation"]
                self.counter = 30

        return command

    def control_tp2(self):
        """
        Control function for TP2
        Main control function with full SLAM, random exploration and path planning
        """
        pose = self.odometer_values()
        lidar = self.lidar()

        command = potential_field_control(lidar, pose, self.current_target)

        if command["forward"] == 0.0 and command["rotation"] == 0.0:
            self.choose_random_goal(pose, lidar)
            print(f"Choosing new random goal: {self.current_target[:2]}")

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

        safe_distance = random_distance * 0.7

        x = pose[0] + safe_distance * np.cos(pose[2] + random_angle)
        y = pose[1] + safe_distance * np.sin(pose[2] + random_angle)

        self.current_target = [x, y, 0]
