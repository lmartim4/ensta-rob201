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


        # Origin of the odom frame in the map frame
        self.odom_pose_ref = np.array([0, 0, 0])

    def explore_frontiers(self):
        """ Frontier based exploration """
        goal = np.array([0, 0, 0])  # frontier to reach for exploration
        return goal

    def heuristic(self, cell1, cell2):
        """
        Calcule la distance euclidienne entre deux cellules.
        """
        return np.linalg.norm(np.array(cell1) - np.array(cell2))


    def get_neighbors(self, current_cell):
        """
        Returns a list of the 8 neighbors of the current cell using NumPy.
        Args:
        current_cell: tuple (x, y) representing the current cell in map coordinates
        Returns:
            numpy array of shape (n, 2) representing valid neighboring cells
        """
        x, y = current_cell
        directions = np.array([
            [-1, -1],  # Upper-left
            [-1,  0],  # Up 
            [-1,  1],  # Upper-right
            [ 0, -1],  # Left
            [ 0,  1],  # Right
            [ 1, -1],  # Lower-left
            [ 1,  0],  # Down
            [ 1,  1]   # Lower-right
        ])
        neighbors = directions + np.array([x, y])
        valid_mask = (
            (neighbors[:, 0] >= 0) & 
            (neighbors[:, 0] < self.occupancy_grid.x_max_map) &
            (neighbors[:, 1] >= 0) & 
            (neighbors[:, 1] < self.occupancy_grid.y_max_map)
        )
        return neighbors[valid_mask]


    def plan(self, start, goal):
        return [start, goal]
