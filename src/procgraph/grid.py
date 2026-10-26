"""Shortest paths over an occupancy grid (0 = free, 1 = occupied).

Paths move between 8-connected cells; a diagonal step costs sqrt(2) cells.
"""
import math

import numpy as np
import skimage.graph

OBSTACLE_THRESHOLD = 0.5


def _path_finder(occupancy_grid):
    step_costs = np.ones(occupancy_grid.shape)
    step_costs[occupancy_grid >= OBSTACLE_THRESHOLD] = -1  # impassable
    return skimage.graph.MCP_Geometric(step_costs)


def distance_grid(occupancy_grid, start):
    """Shortest-path distance, in cells, from start to every cell (inf if unreachable)."""
    return _path_finder(occupancy_grid).find_costs(starts=[start])[0]


def shortest_path(occupancy_grid, start, goal):
    """Cells on a shortest path from start to goal as a 2xN array, or None if unreachable."""
    path_finder = _path_finder(occupancy_grid)
    distances = path_finder.find_costs(starts=[start], ends=[goal])[0]
    if math.isinf(distances[tuple(goal)]):
        return None
    return np.array(path_finder.traceback(goal)).T


def path_length(path):
    """Euclidean length of a 2xN path, in cells."""
    return float(np.sum(np.linalg.norm(np.diff(path, axis=1), axis=0)))
