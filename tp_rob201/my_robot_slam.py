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

from control import potential_field_control, reactive_obst_avoid, is_stopped
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

            print(f"Final Score = {self.tiny_slam.score20(best_score):.1f}")

            self.corrected_pose = self.tiny_slam.get_corrected_pose(odometer)

            if best_score > 16 or self.tick_count < 20:
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
                        self.tiny_slam.grid.conv_map_to_world(point[0], point[1])
                        for point in path_points
                    ]
                )
                trajectory = np.array(
                    [
                        world_points[:, 0],  # All x coordinates
                        world_points[:, 1],  # All y coordinates
                    ]
                )

            self.tiny_slam.grid.display_cv(self.corrected_pose, self.target, trajectory)

    def control(self):
        """
        Main control function executed at each time step
        """
        self.slam_tick()
        self.map_tick()

        control = self.control_tp5()
        print(f"F:{control['forward']:.2f} R:{control['rotation']:.2f}")

        return control

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

        if command["forward"] == 0.0 and command["rotation"] == 0.0:
            self.choose_random_goal(pose, lidar)
            print(f"Choosing new random goal: {self.target[:2]}")

        return command

    def control_tp5(self):
        """
        Control function for TP5
        Main control function with full SLAM,
        predefined goals and A* path planning
        """
        # Define the list of hard-coded goals [x, y, theta]
        predefined_goals = [
            [200.0, 50.0, 0.0],
        ]

        # Initialize goal_index and path if they don't exist
        self.target = predefined_goals[self.goal_index]
        X_t, Y_t, Theta_t = self.target
        self.current_path = None
        self.path_index = 0

        # Get current pose and update with SLAM correction
        pose = self.odometer_values()
        x_c, y_c, theta_c = self.tiny_slam.get_corrected_pose(pose)

        # Get lidar data for obstacle detection
        lidar = self.lidar()

        # Convert world coordinates to map coordinates
        current_pos_map = self.occupancy_grid.conv_world_to_map(x_c, y_c)
        target_pos_map = self.occupancy_grid.conv_world_to_map(X_t, Y_t)

        # Plan or replan path if needed
        if (
            self.current_path is None
            or len(self.current_path) == 0
            or self.path_index >= len(self.current_path)
        ):
            # Plan new path using A* planner
            print(f"New goal {self.goal_index}: {self.target[:2]}")
            self.current_path = self.planner.plan(current_pos_map, target_pos_map)
            self.path_index = 0

            # If no path is found, try the next goal
            if self.current_path is None or len(self.current_path) == 0:
                print(f"No path found to goal {self.goal_index}, trying next goal")
                self.goal_index = (self.goal_index + 1) % len(predefined_goals)
                self.target = predefined_goals[self.goal_index]
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
                    self.target = predefined_goals[self.goal_index]
                    self.current_path = None

            if is_stopped(command):
                print("Robot seems stuck")

            return command

        # Fallback - should not reach here
        return {"forward": 0.0, "rotation": 0.1}

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
