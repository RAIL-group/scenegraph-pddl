import math

import numpy as np
import pytest

from procgraph import grid


@pytest.fixture
def wall_with_gap():
    """10x10 free grid with a wall across row 5, open only at column 9."""
    occupancy_grid = np.zeros((10, 10), dtype=int)
    occupancy_grid[5, :9] = 1
    return occupancy_grid


def test_distance_grid_straight_line():
    distances = grid.distance_grid(np.zeros((5, 5), dtype=int), (0, 0))
    assert distances[0, 4] == pytest.approx(4)
    assert distances[4, 4] == pytest.approx(4 * math.sqrt(2))


def test_distance_grid_goes_around_walls(wall_with_gap):
    distances = grid.distance_grid(wall_with_gap, (0, 0))
    assert distances[9, 0] > 9  # must detour through the gap
    assert math.isinf(distances[5, 0])  # wall cells are unreachable


def test_shortest_path(wall_with_gap):
    path = grid.shortest_path(wall_with_gap, (0, 0), (9, 0))
    assert tuple(path[:, 0]) == (0, 0) and tuple(path[:, -1]) == (9, 0)
    assert all(wall_with_gap[tuple(cell)] == 0 for cell in path.T)
    distances = grid.distance_grid(wall_with_gap, (0, 0))
    assert grid.path_length(path) == pytest.approx(distances[9, 0])


def test_shortest_path_unreachable(wall_with_gap):
    wall_with_gap[5, 9] = 1
    assert grid.shortest_path(wall_with_gap, (0, 0), (9, 0)) is None
