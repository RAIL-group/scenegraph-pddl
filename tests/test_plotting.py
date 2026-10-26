"""Plan plotting on a small hand-made house (no ProcTHOR data needed)."""
from types import SimpleNamespace

import numpy as np
import pytest

from scenegraph_pddl import plotting
from scenegraph_pddl.pddl import problem


@pytest.fixture
def house():
    occupancy_grid = np.zeros((10, 10), dtype=int)
    occupancy_grid[4, 2:8] = 1  # a wall the route must go around
    return SimpleNamespace(occupancy_grid=occupancy_grid, robot_position=(1, 5),
                           top_down_image=None, top_down_extent=None)


@pytest.fixture
def graph():
    nodes = [
        {'id': 'Apartment|0', 'name': 'apartment', 'type': 'apartment', 'position': None},
        {'id': 'room|2', 'name': 'kitchen', 'type': 'room', 'position': (5, 5)},
        {'id': 'Fridge|2|1', 'name': 'fridge', 'type': 'container', 'position': (8, 5)},
        {'id': 'Apple|2|2', 'name': 'apple', 'type': 'object', 'position': (8, 5)},
    ]
    return {'nodes': nodes, 'edges': [(0, 1), (1, 2), (2, 3)], 'containers': [2], 'objects': [3]}


ACTIONS = [('move', 'robot1', 'base', 'fridge_2'),
           ('pick', 'robot1', 'apple_3', 'fridge_2', 'hand1'),
           ('move', 'robot1', 'fridge_2', 'base')]


def test_plan_route(house, graph):
    points, costs, steps = plotting.plan_route(house, graph, ACTIONS)
    motion_costs = problem.compute_motion_costs(house, graph)
    move_cost = motion_costs[('base', 'fridge_2')] + motion_costs[('fridge_2', 'base')]
    # The route starts and ends at base, and goes around the wall
    assert tuple(points[0]) == tuple(points[-1]) == house.robot_position
    assert all(house.occupancy_grid[tuple(point)] == 0 for point in points)
    assert costs[-1] == pytest.approx(move_cost + plotting.PICK_PLACE_COST, abs=1)
    assert [location for _, location, _ in steps] == ['fridge_2', 'fridge_2', 'base']


def test_save_plan_plot(house, graph, tmp_path):
    path = tmp_path / 'plan.png'
    plotting.save_plan_plot(house, graph, ACTIONS, [('apple_3', 'base')], 40, str(path))
    assert path.stat().st_size > 0
