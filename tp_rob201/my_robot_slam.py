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
        self.enable_localisation = True  # Improvement of localise is clear
        self.planner = Planner(self.occupancy_grid)

        # storage for pose after localization
        self.pose_to_use = np.array([0, 0, 0])

        self.load_last_status = True
        
        # Use predefined goals (TP2 or A* TP5)
        self.has_completed_mapping = False
        
        self.current_waypoint_index = 0
        self.waypoints = [
            (0, 80, 0),
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
            (-270, -70, 0),
            (-380, -60, 0),
            (-410, 100, 0),
            # After this point it should use A* to go somewhere else.
        ]

        # Set this target so control algorithms will try to reach it.
        self.target = self.waypoints[self.current_waypoint_index]

        # Store path here to display on map later.
        self.current_path = None
        self.current_path_index = 0
        self.positions_only_path = None

        # Used to know if we are at the start of the simulation.
        self.tick_count = 0

        if self.load_last_status:
            self.occupancy_grid.load_state("last_grid")
            self.occupancy_grid.inflate_obstacles(20)
            self.start_A_star()
            self.occupancy_grid.load_state("last_grid")
        

    def update_map_tick(self, raw_odom, lidar):
        if self.enable_localisation:            
            best_score = self.tiny_slam.localise(lidar, raw_odom)

            self.pose_to_use = self.tiny_slam.get_corrected_pose(raw_odom)

            if best_score > 5000 or self.tick_count < 50:
                self.tiny_slam.update_map(lidar, self.pose_to_use)
        else:
            self.pose_to_use = raw_odom
            self.tiny_slam.update_map(lidar, raw_odom)

    def draw_map_tick(self):
        self.tick_count += 1
        if self.tick_count % 10 == 0:
            trajectory = None
            
            if self.positions_only_path is not None and len(self.positions_only_path) > 0:
                path_points = np.array(self.positions_only_path)
                trajectory = np.array(
                    [
                        path_points[:, 0],
                        path_points[:, 1],
                    ]
                )
            self.occupancy_grid.display_cv(
                self.pose_to_use, self.target, trajectory)
    
    def start_A_star(self):
        self.current_path_index = 0
        waypoint_index = np.random.randint(0, len(self.waypoints))
        
        self.current_path = self.planner.plan(self.pose_to_use, self.waypoints[waypoint_index])
        
        if(self.current_path == None):
            print("Failed to find path to waypoint: ", waypoint_index)
            return
            
        self.positions_only_path = [[point[0], point[1]] for point in self.current_path]
        
        
    def control(self):
        raw_odom = self.odometer_values()
        lidar = self.lidar()

        self.update_map_tick(raw_odom, lidar)
        self.draw_map_tick()

        if not self.has_completed_mapping and not self.load_last_status:
            command = self.control_tp2(lidar)
        else:
            command = self.control_tp5(lidar)

        return command

    def control_tp1(self):
        if not hasattr("last_rotation"):
            self.last_rotation = 0

        if self.tick_count > 0:
            self.tick_count -= 1

            command = {"forward": 0.5, "rotation": self.last_rotation}
        else:
            command = reactive_obst_avoid(self.lidar())
            if command["rotation"] != 0:
                self.last_rotation = command["rotation"]
                self.tick_count = 30

        return command

    def control_tp2(self, lidar):
        if has_arrived(self.pose_to_use, self.target):
            self.next_waypoint()

        return potential_field_control(lidar, self.pose_to_use, self.target)

    def control_tp5(self, lidar):
        if self.current_path is None:
            print("No path found to the waypoint!")
            return {"forward": 0, "rotation": 0}
        
        if self.current_path_index >= len(self.current_path):
            print("Already at final target!")
            return {"forward": 0, "rotation": 0}
        
        current_target = self.current_path[self.current_path_index]
        if has_arrived(self.pose_to_use, current_target):
            self.current_path_index += 10
            print(f"Evolving {self.current_path_index}/{len(self.current_path)}")
            
            if self.current_path_index >= len(self.current_path):
                print("Reached final target in path!")
                return {"forward": 0, "rotation": 0}
        
        return potential_field_control(lidar, self.pose_to_use, current_target)
    
    # This helps us use tp2 to complete map scanning and then
    # use tp5 A* to plan trajectory
    def next_waypoint(self):
        self.current_waypoint_index += 1

        if self.current_waypoint_index == len(self.waypoints):
            self.has_completed_mapping = True
            self.occupancy_grid.save_state("last_grid")
            self.tiny_slam.save_state("last_slam")

        if not self.has_completed_mapping:
            self.target = self.waypoints[self.current_waypoint_index]
        else:
            self.target = [0, 0, 0]

    def choose_random_goal(self, pose, lidar):
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