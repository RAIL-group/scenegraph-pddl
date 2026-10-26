"""Problem generation and solving on a small hand-made scene graph (no ProcTHOR data needed)."""
import os

import pytest

from scenegraph_pddl import config
from scenegraph_pddl.pddl import fast_downward, problem


@pytest.fixture
def graph():
    nodes = [
        {'id': 'Apartment|0', 'name': 'apartment', 'type': 'apartment', 'position': None},
        {'id': 'room|2', 'name': 'kitchen', 'type': 'room', 'position': (5, 5)},
        {'id': 'Fridge|2|1', 'name': 'fridge', 'type': 'container', 'position': (1, 1)},
        {'id': 'CounterTop|2|2', 'name': 'countertop', 'type': 'container', 'position': (8, 8)},
        {'id': 'Apple|2|3', 'name': 'apple', 'type': 'object', 'position': (8, 8)},
        {'id': 'Egg|2|4', 'name': 'egg', 'type': 'object', 'position': (1, 1)},
    ]
    edges = [(0, 1), (1, 2), (1, 3), (3, 4), (2, 5)]
    return {'nodes': nodes, 'edges': edges, 'containers': [2, 3], 'objects': [4, 5]}


@pytest.fixture
def motion_costs():
    costs = {('base', 'fridge_2'): 5, ('base', 'countertop_3'): 7, ('fridge_2', 'countertop_3'): 9}
    return {**costs, **{(b, a): cost for (a, b), cost in costs.items()}}


def test_names_and_locations(graph):
    assert list(problem.container_names(graph)) == ['fridge_2', 'countertop_3']
    assert problem.object_locations(graph) == {'apple_4': 'countertop_3', 'egg_5': 'fridge_2'}


def test_resolve_goals(graph):
    assert problem.resolve_goals(graph, [('apple', 'fridge')]) == [('apple_4', 'fridge_2')]
    assert problem.resolve_goals(graph, [('apple_4', 'fridge_2')]) == [('apple_4', 'fridge_2')]
    with pytest.raises(ValueError):
        problem.resolve_goals(graph, [('banana', 'fridge')])


def test_random_goals_move_objects(graph):
    locations = problem.object_locations(graph)
    for obj, container in problem.random_goals(graph, num_goals=2, seed=0):
        assert locations[obj] != container


def test_build_problem(graph, motion_costs):
    pddl = problem.build_problem(graph, [('apple', 'fridge')], motion_costs)
    for fact in ['(In apple_4 countertop_3)', '(In egg_5 fridge_2)', '(ContainerUnsafe base)',
                 '(= (MotionCost base fridge_2) 5)', '(In apple_4 fridge_2)']:
        assert fact in pddl


@pytest.mark.skipif(not os.path.exists(os.path.join(config.FAST_DOWNWARD_DIR, 'builds')),
                    reason='Fast Downward not built (make fast-downward)')
def test_solve(graph, motion_costs, tmp_path):
    problem_path = tmp_path / 'problem.pddl'
    problem_path.write_text(problem.build_problem(graph, [('apple', 'fridge')], motion_costs))
    actions, cost = fast_downward.solve(problem.DOMAIN_PATH, str(problem_path),
                                        plan_path=str(tmp_path / 'plan.txt'))
    assert [action[0] for action in actions] == ['move', 'pick', 'move', 'place', 'move']
    # base -> countertop (7) -> fridge (9) -> base (5), plus pick and place (10 each)
    assert cost == 7 + 9 + 5 + 20
