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

    # def heuristic(self, cell1, cell2):
    #    return np.linalg.norm(np.array(cell1) - np.array(cell2))

    def heuristic(self, cell_1, cell_2):
        x1, y1 = cell_1
        x2, y2 = cell_2
        return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

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
        start = tuple(A)
        goal = tuple(B)

        open_set = []

        in_open_set = set([start])

        came_from = {}

        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}

        heapq.heappush(open_set, (f_score[start], start))

        while open_set:
            _, current = heapq.heappop(open_set)
            in_open_set.remove(current)

            if current == goal:
                return self.reconstruct_path(came_from, current)

            for neighbor in self.get_neighbors(current):
                neighbor = tuple(neighbor)

                if self.grid.occupancy_map[neighbor] > 20:
                    continue

                movement_cost = self.movement_cost(current, neighbor)
                test_g_score = g_score[current] + movement_cost

                if test_g_score < g_score.get(neighbor, float("infinity")):
                    came_from[neighbor] = current
                    g_score[neighbor] = test_g_score
                    f_score[neighbor] = test_g_score + self.heuristic(neighbor, goal)

                    if neighbor not in in_open_set:
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
                        in_open_set.add(neighbor)

        return None
