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

from control import potential_field_control, has_arrived
from occupancy_grid import OccupancyGrid
from planner import Planner

class MyRobotSlam(RobotAbstract):
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

        # Used to know if we are at the start of the simulation.
        self.tick_count = 0
        
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

        # CONFIGURATION
        
        # Whether we preload a map or starts from an empty grid
        self.preload_occupancy_map = False
        
        # This is the list of checkpoints the robot should follow in order to construct its map
        self.exploring_waypoints = [
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
        ]
        self.exploring_waypoints_index = 0
        
        # Control functions will try to reach the self.target
        self.target = self.exploring_waypoints[self.exploring_waypoints_index]
        
        # Current best pose estimation (can be raw_odom or localized. Depends on "self.enable_localisation")
        self.best_pose = np.array([0, 0, 0])
        self.enable_localisation = True # Wheter we will use localisation technology or not
                
        # Use predefined goals (TP2 or A* TP5)
        self.has_completed_mapping = False
        
        # Store path here to display on map later.
        self.path_2d = None # Use to draw in map
        self.current_path = None
        self.current_path_index = 0

        if self.preload_occupancy_map:
            self.occupancy_grid.load_state("last_grid")
            self.planAndStartTrajectoryToRandomWaypoint()
        

    def update_map_tick(self, raw_odom, lidar):
        if self.enable_localisation:
            best_score = self.tiny_slam.localise_optimized(lidar, raw_odom)
            self.best_pose = self.tiny_slam.get_corrected_pose(raw_odom)

            if best_score > 5000 or self.tick_count < 50:
                self.tiny_slam.update_map(lidar, self.best_pose)
        else:
            self.best_pose = raw_odom
            self.tiny_slam.update_map(lidar, raw_odom)

    def draw_map_tick(self):
        self.tick_count += 1
        
        if self.tick_count % 10 == 0:
            self.occupancy_grid.display_cv(self.best_pose, self.target, self.path_2d)
    
    def planAndStartTrajectoryToRandomWaypoint(self):   
        self.occupancy_grid.occupancy_map, grid_backup = self.occupancy_grid.get_inflated_map(radius=30, threshold=15)
        
        self.current_path = None
        while(self.current_path == None):
            waypoint_index = np.random.randint(0, len(self.exploring_waypoints))
            print(f"Searching planner for Waypoint({waypoint_index})")
            self.current_path = self.planner.plan(self.best_pose, self.exploring_waypoints[waypoint_index])
        
        self.occupancy_grid.occupancy_map = grid_backup
        self.current_path_index = 0     
        self.target = self.current_path[self.current_path_index]
            
        # Create 2d path for map drawing
        self.path_2d = [[point[0], point[1]] for point in self.current_path]
        self.path_2d = np.array(self.path_2d)
        self.path_2d  = np.array(
            [
                self.path_2d [:, 0],
                self.path_2d [:, 1],
            ]
        )
    
        
    def control(self):
        raw_odom = self.odometer_values()
        lidar = self.lidar()

        self.update_map_tick(raw_odom, lidar)
        self.draw_map_tick()

        if not self.has_completed_mapping and not self.preload_occupancy_map:
            command = self.control_tp2(lidar)
        else:
            command = self.control_tp5(lidar)

        return command



    def control_tp2(self, lidar):
        if has_arrived(self.best_pose, self.target):
            self.next_waypoint()

        return potential_field_control(lidar, self.best_pose, self.target)

    def control_tp5(self, lidar):
        if self.current_path is None:
            print("current_path is empty! Do not use control_tp5 without it")
            return {"forward": 0, "rotation": 0}
        
        if has_arrived(self.best_pose, self.target):
            self.current_path_index += 10
            
            if self.current_path_index >= len(self.current_path):
                print("Reached final target in path!")
                self.planAndStartTrajectoryToRandomWaypoint()
                return {"forward": 0, "rotation": 0}
            else:
                print(f"Evolving {self.current_path_index}/{len(self.current_path)}")
                self.target = self.current_path[self.current_path_index]
                
        return potential_field_control(lidar, self.best_pose, self.target)
    
    # This helps us use tp2 to complete map scanning and then
    # use tp5 A* to plan trajectory
    def next_waypoint(self):
        self.exploring_waypoints_index += 1

        if self.exploring_waypoints_index == len(self.exploring_waypoints):
            self.has_completed_mapping = True
            self.occupancy_grid.save_state("last_grid")

        if not self.has_completed_mapping:
            self.target = self.exploring_waypoints[self.exploring_waypoints_index]
        else:
            self.target = [0, 0, 0]