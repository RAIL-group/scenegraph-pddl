"""Build a PDDL problem for domain.pddl from a scene graph.

Every container is a location the robot can move to, and every object starts
In the container it is attached to in the graph. The robot starts at (and must
return to) 'base', its start position, where nothing may be placed. Moving
between two locations costs their shortest-path distance over the house's
occupancy grid, in grid cells.
"""
import math
import os
import random
import re

from procgraph import grid

DOMAIN_PATH = os.path.join(os.path.dirname(__file__), 'domain.pddl')

ROBOT = 'robot1'
HANDS = ('hand1', 'hand2')
BASE = 'base'


def pddl_name(graph, node):
    """Unique PDDL name of a graph node, e.g. 'fridge_12'."""
    name = re.sub(r'[^a-z0-9_]', '_', graph['nodes'][node]['name'].lower())
    return f'{name}_{node}'


def container_names(graph):
    """PDDL name -> node index, for every container."""
    return {pddl_name(graph, node): node for node in graph['containers']}


def object_names(graph):
    """PDDL name -> node index, for every object."""
    return {pddl_name(graph, node): node for node in graph['objects']}


def object_locations(graph):
    """Object PDDL name -> PDDL name of the container holding it."""
    containers = set(graph['containers'])
    objects = set(graph['objects'])
    return {pddl_name(graph, child): pddl_name(graph, parent)
            for parent, child in graph['edges']
            if parent in containers and child in objects}


def location_cells(house, graph):
    """Grid cell of every location: each container, then the robot's start (base)."""
    cells = {name: tuple(graph['nodes'][node]['position'])
             for name, node in container_names(graph).items()}
    cells[BASE] = tuple(house.robot_position)
    return cells


def compute_motion_costs(house, graph):
    """(from, to) -> shortest-path distance in grid cells between every pair of
    locations, rounded to an integer (Fast Downward only supports integer action
    costs). Pairs with no path between them are left out."""
    cells = location_cells(house, graph)
    occupancy_grid = house.occupancy_grid.copy()
    for cell in cells.values():
        occupancy_grid[cell] = 0

    motion_costs = {}
    for start_name, start in cells.items():
        distances = grid.distance_grid(occupancy_grid, start)
        for goal_name, goal in cells.items():
            distance = distances[goal]
            if start_name != goal_name and not math.isinf(distance):
                motion_costs[(start_name, goal_name)] = int(round(distance))
    return motion_costs


def resolve_goals(graph, goals):
    """Map each (object, container) goal to PDDL names. Names may already be
    PDDL names ('apple_23') or generic names ('apple'); a generic name resolves
    to the first node with that name."""
    containers = container_names(graph)
    objects = object_names(graph)
    return [(_resolve_name(graph, obj, objects, 'object'),
             _resolve_name(graph, container, containers, 'container'))
            for obj, container in goals]


def random_goals(graph, num_goals=1, seed=None):
    """Move num_goals random objects to random containers they are not already in."""
    rng = random.Random(seed)
    locations = object_locations(graph)
    containers = list(container_names(graph))
    goals = []
    for obj in rng.sample(sorted(locations), num_goals):
        goals.append((obj, rng.choice([c for c in containers if c != locations[obj]])))
    return goals


def build_problem(graph, goals, motion_costs, problem_name='pick-place'):
    """problem.pddl content for putting each (object, container) in goals."""
    containers = list(container_names(graph))
    locations = object_locations(graph)
    goals = resolve_goals(graph, goals)

    init = [f'(Robot {ROBOT})', f'(At {ROBOT} {BASE})']
    for hand in HANDS:
        init += [f'(Hand {hand})', f'(HandEmpty {ROBOT} {hand})']
    for obj, container in locations.items():
        init += [f'(Object {obj})', f'(In {obj} {container})']
    init.append(f'(ContainerUnsafe {BASE})')
    for (start, goal), cost in motion_costs.items():
        init += [f'(Motion {start} {goal})', f'(= (MotionCost {start} {goal}) {cost})']
    init.append('(= (total-cost) 0)')

    goal = [f'(At {ROBOT} {BASE})'] + [f'(HandEmpty {ROBOT} {hand})' for hand in HANDS]
    goal += [f'(In {obj} {container})' for obj, container in goals]

    objects = [ROBOT, *HANDS, BASE, *containers, *locations]
    return _format_problem('procthor', problem_name, objects, init, goal)


def _resolve_name(graph, name, named_nodes, kind):
    if name in named_nodes:
        return name
    for candidate, node in named_nodes.items():
        if graph['nodes'][node]['name'] == name.lower():
            return candidate
    raise ValueError(f"No {kind} named '{name}'. Available: {sorted(named_nodes)}")


def _format_problem(domain_name, problem_name, objects, init, goal):
    lines = [f'(define (problem {problem_name})',
             f'    (:domain {domain_name})',
             '    (:objects']
    lines += [f'        {obj}' for obj in objects]
    lines += ['    )', '    (:init']
    lines += [f'        {fact}' for fact in init]
    lines += ['    )', '    (:goal', '        (and']
    lines += [f'            {fact}' for fact in goal]
    lines += ['        )', '    )',
              '    (:metric minimize (total-cost))',
              ')']
    return '\n'.join(lines) + '\n'
