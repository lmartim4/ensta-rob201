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
        lidar = self.lidar()
        best_score = self.tiny_slam.localise(lidar, odometer_data)

        print(f"Final Score = {self.tiny_slam.note_sur_20(best_score):.1f}")

        self.corrected_pose = self.tiny_slam.get_corrected_pose(odometer_data)

        if best_score > 16 or self.counter < 20:
            print("Updating Map!")
            self.tiny_slam.update_map(lidar, self.corrected_pose)

        self.counter += 1

        if self.counter % 10 == 0:
            self.tiny_slam.grid.display_cv(self.corrected_pose, self.current_target)

        control = self.control_tp5()

        f, r = control["forward"], control["rotation"]
        print(f"F:{f:.2f} R:{r:.2f}")
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
        pose = self.tiny_slam.get_corrected_pose(pose)

        lidar = self.lidar()

        command = potential_field_control(lidar, pose, self.current_target)

        if command["forward"] == 0.0 and command["rotation"] == 0.0:

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

                self.current_target = [x, y, 0]

            self.choose_random_goal(pose, lidar)
            print(f"Choosing new random goal: {self.current_target[:2]}")

        return command

    def control_tp5(self):
        """
        Control function for TP5
        Main control function with full SLAM, predefined goals and A* path planning
        """
        # Define the list of hard-coded goals [x, y, theta]
        predefined_goals = [
            [100.0, 150.0, 0.0],
            [200.0, 50.0, 0.0],
            [50.0, 200.0, 0.0],
        ]

        # Initialize goal_index and path if they don't exist
        if not hasattr(self, "goal_index"):
            self.goal_index = 0
            self.current_target = predefined_goals[self.goal_index]
            self.current_path = None
            self.path_index = 0

        # Get current pose and update with SLAM correction
        pose = self.odometer_values()
        pose = self.tiny_slam.get_corrected_pose(pose)

        # Get lidar data for obstacle detection
        lidar = self.lidar()

        # Convert world coordinates to map coordinates
        current_pos_map = self.occupancy_grid.conv_world_to_map(pose[0], pose[1])
        target_pos_map_ = self.occupancy_grid.conv_world_to_map(
            self.current_target[0], self.current_target[1]
        )

        # Plan or replan path if needed
        if (
            self.current_path is None
            or len(self.current_path) == 0
            or self.path_index >= len(self.current_path)
        ):
            # Plan new path using A* planner
            print(f"Planning path to goal {self.goal_index}: {self.current_target[:2]}")
            self.current_path = self.planner.plan(current_pos_map, target_pos_map)
            self.path_index = 0

            # If no path is found, try the next goal
            if self.current_path is None or len(self.current_path) == 0:
                print(f"No path found to goal {self.goal_index}, trying next goal")
                self.goal_index = (self.goal_index + 1) % len(predefined_goals)
                self.current_target = predefined_goals[self.goal_index]
                # Rotate a bit to scan environment
                return {"forward": 0.0, "rotation": 0.1}

        # Get next waypoint in the path
        if self.path_index < len(self.current_path):
            next_waypoint_map = self.current_path[self.path_index]
            next_waypoint_world = self.occupancy_grid.conv_map_to_world(
                next_waypoint_map[0], next_waypoint_map[1]
            )

            # Create a temporary target for the next waypoint
            temp_target = [next_waypoint_world[0], next_waypoint_world[1], 0.0]

            # Use potential field control to move to the next waypoint
            command = potential_field_control(lidar, pose, temp_target)

            # Check if we've reached the current waypoint
            dist_to_waypoint = np.sqrt(
                (pose[0] - next_waypoint_world[0]) ** 2
                + (pose[1] - next_waypoint_world[1]) ** 2
            )

            if dist_to_waypoint < 10.0:  # Close enough to current waypoint
                self.path_index += 1
                print(f"Reached waypoint {self.path_index}/{len(self.current_path)}")

                # If we've reached the final waypoint (the goal)
                if self.path_index >= len(self.current_path):
                    print(f"Reached goal {self.goal_index}")
                    self.goal_index = (self.goal_index + 1) % len(predefined_goals)
                    self.current_target = predefined_goals[self.goal_index]
                    self.current_path = None

            # Safety check - if we're stuck, try to replan
            if command["forward"] < 0.00001 and abs(command["rotation"]) < 0.00001:
                print("Potential field control is stuck, replanning...")
                self.current_path = None
                # Rotate a bit to scan environment
                return {"forward": 0.0, "rotation": 0.1}

            return command

        # Fallback - should not reach here
        return {"forward": 0.0, "rotation": 0.1}
