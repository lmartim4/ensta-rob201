"""
Planner class
Implementation of A*
"""

import numpy as np
import heapq
from occupancy_grid import OccupancyGrid


class Planner:
    """Simple occupancy grid Planner"""

    def __init__(self, occupancy_grid: OccupancyGrid):
        self.grid = occupancy_grid
        self.odom_pose_ref = np.array([0, 0, 0])

        self.directions = np.array(
            [
                [-1, -1],  # Upper-left
                [-1, 0],  # Up
                [-1, 1],  # Upper-right
                [0, -1],  # Left
                [0, 1],  # Right
                [1, -1],  # Lower-left
                [1, 0],  # Down
                [1, 1],  # Lower-right
            ]
        )

    def reconstruct_path(self, came_from, current):
        total_path = [current]
        while current in came_from:
            current = came_from[current]
            total_path.insert(0, current)
        return total_path

    def heuristic(self, cell1, cell2):
        return np.linalg.norm(np.array(cell1) - np.array(cell2))

    def get_neighbors(self, current_cell):
        x, y = current_cell
        neighbors = self.directions + np.array([x, y])
        valid_mask = (
            (neighbors[:, 0] >= 0)
            & (neighbors[:, 0] < self.grid.x_max_map)
            & (neighbors[:, 1] >= 0)
            & (neighbors[:, 1] < self.grid.y_max_map)
        )
        return neighbors[valid_mask]

    def movement_cost(self, current, neighbor):
        dx = abs(current[0] - neighbor[0])
        dy = abs(current[1] - neighbor[1])

        if dx == 1 and dy == 1:
            return 1.414
        return 1.0

    def plan(self, A, B):
        start = tuple(self.grid.conv_world_to_map(A[0], A[1]))
        goal = tuple(self.grid.conv_world_to_map(B[0], B[1]))

        open_set = []
        came_from = {}

        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}

        heapq.heappush(open_set, (f_score[start], start))
        in_open_set = set([start])
        
        # Store the direction we came from for each node (used to calculate angles)
        direction_from = {}
        
        while open_set:
            _, current = heapq.heappop(open_set)
            in_open_set.remove(current)
            
            if current == goal:
                return self.reconstruct_path_with_angles(came_from, direction_from, current, A[2])
            
            for neighbor in self.get_neighbors(current):
                neighbor = tuple(neighbor)
                d = self.movement_cost(current, neighbor)
                test_g_score = g_score[current] + d

                if self.grid.occupancy_map[neighbor[0], neighbor[1]] > 15:
                    continue
                
                if test_g_score < g_score.get(neighbor, float("infinity")):
                    direction = (neighbor[0] - current[0], neighbor[1] - current[1])
                    direction_from[neighbor] = direction
                    came_from[neighbor] = current
                    g_score[neighbor] = test_g_score
                    f_score[neighbor] = test_g_score + self.heuristic(neighbor, goal)

                    if neighbor not in in_open_set:
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
                        in_open_set.add(neighbor)
                                
        return None
    
    def reconstruct_path_with_angles(self, came_from, direction_from, current, start_angle):
        path = []
        
        while current in came_from:
            x_world, y_world = self.grid.conv_map_to_world(current[0], current[1])
            
            if current in direction_from:
                dx, dy = direction_from[current]
                
                angle = np.arctan2(dy, dx)
                
                path.append([x_world, y_world, angle])
            else:
                if path:
                    angle = path[-1][2]
                else:
                    prev = came_from[current]
                    dx = current[0] - prev[0]
                    dy = current[1] - prev[1]
                    angle = np.arctan2(dy, dx)
                
                path.append([x_world, y_world, angle])
            
            current = came_from[current]
        
        x_world, y_world = self.grid.conv_map_to_world(current[0], current[1])
        
        path.append([x_world, y_world, start_angle])
        path.reverse()
        
        return path