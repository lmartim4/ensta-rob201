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
    
    LOCALIZATION_SCORE_THRESHOLD = 5000
    INITIAL_MAP_UPDATE_TICKS = 50
    MAP_DISPLAY_FREQUENCY = 10
    PATH_SKIP_STEPS = 15
    
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
        self.preload_occupancy_map = False #False
        
        # This is the list of checkpoints the robot should follow in order to construct its map
        self.exploring_waypoints = [
            (0, 80, 0),
            (200, 180, 0),
            (300, 200, 0),
            (450, 270, 0),
            (450, -100, 0), 
            (450, -300.0, 0),
            (0, -300, 0),
            (100, -300, 0),
            (300, -100, 0),
            (300, -70, 0),
            (0, -70, 0),
            (0, 180, 0),
            (-310, 180, 0),
            (-310, 120, 0),
            (-200, 0, 0),
            (-200, 80, 0),
            (-310, 80, 0),
            (-310, -70, 0),
            (-310, -180, 0),
            (-280, -180, 0),
            (-310, -60, 0),
            (-410, -60, 0),
            (-450, 100, 0),
            (-450, 0, 0),
        ]
        self.exploring_waypoints_index = 0 #11
        
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
            self.occupancy_grid.load("last_grid")
            self.planAndStartTrajectoryToRandomWaypoint()
        

    def update_map_tick(self, raw_odom, lidar):
        if self.enable_localisation:
            best_score = self.tiny_slam.localise(lidar, raw_odom)
            self.best_pose = self.tiny_slam.get_corrected_pose(raw_odom)

            if best_score > self.LOCALIZATION_SCORE_THRESHOLD or self.tick_count < self.INITIAL_MAP_UPDATE_TICKS:
                self.tiny_slam.update_map(lidar, self.best_pose)
        else:
            self.best_pose = raw_odom
            self.tiny_slam.update_map(lidar, raw_odom)

    
    def planAndStartTrajectoryToRandomWaypoint(self):   
        #self.occupancy_grid.occupancy_map, grid_backup = self.occupancy_grid.get_inflated_map(radius= self.MAP_INFLATION_RADIUS, threshold=self.OBSTACLE_THRESHOLD)
        
        self.current_path = None
        while(self.current_path == None):
            waypoint_index = np.random.randint(0, len(self.exploring_waypoints))
            print(f"Searching planner for Waypoint({waypoint_index})")
            self.current_path = self.planner.plan(self.best_pose, self.exploring_waypoints[waypoint_index])
        
        self.current_path_index = 0     
        self.update_target(self.current_path[self.current_path_index], "A")
        self.create_2d_path()
    
    def create_2d_path(self):
        print("Indexing trajectory to map")
        
        self.path_2d = [[point[0], point[1]] for point in self.current_path]
        self.path_2d = np.array(self.path_2d)
        self.path_2d  = np.array(
            [
                self.path_2d [:, 0],
                self.path_2d [:, 1],
            ]
        )
    
    def next_waypoint(self):
        self.exploring_waypoints_index += 1

        if self.exploring_waypoints_index == len(self.exploring_waypoints):
            self.has_completed_mapping = True
            self.occupancy_grid.save("last_grid")
            self.planAndStartTrajectoryToRandomWaypoint()
            
        if not self.has_completed_mapping:
            self.update_target(self.exploring_waypoints[self.exploring_waypoints_index], f"Waypoint {self.exploring_waypoints_index}")
    
    def update_target(self, new_target, target_name = "N/A"):
        self.target = new_target
        print(f"Target Update : {target_name} - ({self.target})")
    
    def control(self):
        self.tick_count += 1
        
        raw_odom = self.odometer_values()
        lidar = self.lidar()

        self.update_map_tick(raw_odom, lidar)
        
        if self.tick_count % self.MAP_DISPLAY_FREQUENCY == 0:
            self.occupancy_grid.display_cv(self.best_pose, self.target, self.path_2d)
        
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
            self.current_path_index += self.PATH_SKIP_STEPS
            
            if self.current_path_index >= len(self.current_path):
                print("Reached final target in path!")
                self.planAndStartTrajectoryToRandomWaypoint()
            else:
                self.update_target(self.current_path[self.current_path_index], f"Reached Checkpoint: {self.current_path_index}/{len(self.current_path)}")
                
        return potential_field_control(lidar, self.best_pose, self.target)