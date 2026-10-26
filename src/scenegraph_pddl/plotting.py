"""Plot a plan on the house: the top-down view beside the occupancy grid, with
the robot's route coloured by the cost spent so far, and the list of actions.

The route follows the same shortest paths that problem.compute_motion_costs
measures, so its colour at the end matches the plan's cost.
"""
import numpy as np
from matplotlib import patheffects
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, Normalize
from matplotlib.figure import Figure

from scenegraph_pddl import grid
from scenegraph_pddl.pddl import problem

PICK_PLACE_COST = 10  # as in domain.pddl
# Light to dark red, skipping the near-white start of 'Reds'
ROUTE_CMAP = LinearSegmentedColormap.from_list(
    'route', ['#fcbba1', '#fb6a4a', '#cb181d', '#67000d'])
GRID_CMAP = ListedColormap(['white', '#c8c8c8'])  # free, occupied


def plan_route(house, graph, actions):
    """Replay a plan over the occupancy grid.

    Returns (points, costs, steps): points is an Nx2 array of (x, z) grid cells
    along the route, costs the plan cost spent on reaching each point, and
    steps a list of (action, location, cost when the action starts).
    """
    cells = problem.location_cells(house, graph)
    occupancy_grid = house.occupancy_grid.copy()
    for cell in cells.values():
        occupancy_grid[cell] = 0

    points, costs, steps = [np.array(cells[problem.BASE])], [0.0], []
    cost = 0.0
    for action in actions:
        if action[0] == 'move':
            start, goal = action[2], action[3]
            steps.append((action, goal, cost))
            path = grid.shortest_path(occupancy_grid, cells[start], cells[goal]).T
            lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
            points.extend(path[1:])
            costs.extend(cost + np.cumsum(lengths))
            cost += lengths.sum()
        else:  # pick or place: robot, object, location, hand
            steps.append((action, action[3], cost))
            cost += PICK_PLACE_COST
    return np.array(points), np.array(costs), steps


def render_plan(house, graph, actions, goals, cost):
    """Draw the plan as a matplotlib Figure (see the module docstring)."""
    points, point_costs, steps = plan_route(house, graph, actions)
    cells = problem.location_cells(house, graph)
    norm = Normalize(0, cost)

    figure = Figure(figsize=(17, 6.5))
    FigureCanvasAgg(figure)
    view_ax, map_ax, colorbar_ax, text_ax = figure.subplots(
        1, 4, gridspec_kw={'width_ratios': [1, 1, 0.03, 0.7], 'wspace': 0.2})

    # The top-down view and the grid share coordinates, so they line up
    if house.top_down_image is not None:
        view_ax.imshow(house.top_down_image, extent=house.top_down_extent)
        view_ax.set_title('Top-down view')
        view_ax.axis('off')
    else:
        view_ax.remove()

    map_ax.imshow(house.occupancy_grid.T, cmap=GRID_CMAP, interpolation='nearest')
    map_ax.contour(house.occupancy_grid.T, levels=[0.5], colors='black', linewidths=1)
    if house.top_down_extent is not None:
        left, right, bottom, top = house.top_down_extent
        for ax in (view_ax, map_ax):
            ax.set_xlim(left, right)
            ax.set_ylim(bottom, top)

    segments = np.stack([points[:-1], points[1:]], axis=1)
    route = LineCollection(segments, cmap=ROUTE_CMAP, norm=norm, linewidths=4,
                           capstyle='round', joinstyle='round')
    route.set_array(point_costs[1:])
    map_ax.add_collection(route)

    visited = [problem.BASE] + [location for _, location, _ in steps]
    for location in dict.fromkeys(visited):
        x, z = cells[location]
        map_ax.plot(x, z, 'o', color='black', markersize=6)
        map_ax.annotate(location, (x, z), xytext=(5, -10), textcoords='offset points',
                        fontsize=9, color='black',
                        path_effects=[patheffects.withStroke(linewidth=3, foreground='white')])
    goal_text = ', '.join(f'{obj} → {container}' for obj, container in goals)
    map_ax.set_title(f'Goal: {goal_text}   (cost = {cost:g})')
    map_ax.set_xlabel('x (grid cells)')
    map_ax.set_ylabel('z (grid cells)')

    figure.colorbar(route, cax=colorbar_ax).set_label('plan cost so far')

    # Action list, each marked with the route colour when that action starts
    text_ax.axis('off')
    text_ax.set_xlim(0, 1)
    text_ax.set_ylim(max(len(steps), 12), 0)
    for index, (action, _, start_cost) in enumerate(steps):
        text_ax.plot(0.03, index + 0.5, 'o', markersize=8, color=ROUTE_CMAP(norm(start_cost)))
        text_ax.text(0.08, index + 0.5, f'{index + 1}. {" ".join(action)}',
                     fontsize=9, family='monospace', va='center')
    return figure


def save_plan_plot(house, graph, actions, goals, cost, path):
    """Write the plan plot to path (e.g. plan.png)."""
    render_plan(house, graph, actions, goals, cost).savefig(path, dpi=150, bbox_inches='tight')
