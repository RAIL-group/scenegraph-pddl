"""End-to-end tests on a real ProcTHOR-10k house (needs `make download`)."""
import os

import pytest

from scenegraph_pddl import config
from scenegraph_pddl.pddl import fast_downward, problem

pytestmark = pytest.mark.skipif(not os.path.exists(config.PROCTHOR_DATA_PATH),
                                reason='ProcTHOR data not downloaded (make download)')


@pytest.fixture(scope='module')
def scene():
    from scenegraph_pddl.scripts.run_pipeline import load_scene
    house, graph = load_scene(seed=0)
    return house, graph, problem.compute_motion_costs(house, graph)


def test_house(scene):
    house, _, _ = scene
    assert house.occupancy_grid[house.robot_position] == 0
    assert house.top_down_image.shape[0] == house.top_down_image.shape[1]
    for container in house.containers:
        assert house.occupancy_grid[container['position']] == 0
        assert 0 < len(container['children']) <= 7


def test_scene_graph(scene):
    _, graph, _ = scene
    assert graph['containers'] and graph['objects']
    assert len(graph['node_features']) == len(graph['nodes'])
    # Every object sits in exactly one container
    assert len(problem.object_locations(graph)) == len(graph['objects'])


def test_motion_costs(scene):
    house, graph, motion_costs = scene
    locations = list(problem.container_names(graph)) + [problem.BASE]
    for start in locations:
        for goal in locations:
            if start != goal:
                assert motion_costs[(start, goal)] > 0
                # Distances are symmetric (up to rounding)
                assert abs(motion_costs[(start, goal)] - motion_costs[(goal, start)]) <= 1
    cells = problem.location_cells(house, graph)
    start, goal = cells[locations[0]], cells[problem.BASE]
    assert house.path_cost(start, goal) == pytest.approx(motion_costs[(locations[0], problem.BASE)], abs=0.5)


def test_solve(scene, tmp_path):
    _, graph, motion_costs = scene
    goals = problem.random_goals(graph, num_goals=2, seed=0)
    problem_path = tmp_path / 'problem.pddl'
    problem_path.write_text(problem.build_problem(graph, goals, motion_costs))

    actions, cost = fast_downward.solve(problem.DOMAIN_PATH, str(problem_path),
                                        plan_path=str(tmp_path / 'plan.txt'))
    assert actions is not None
    move_cost = sum(motion_costs[(action[2], action[3])] for action in actions if action[0] == 'move')
    assert cost == 20 * len(goals) + move_cost
    assert set(goals) <= {(action[2], action[3]) for action in actions if action[0] == 'place'}
    assert actions[-1][-1] == problem.BASE
