"""Plot a plan on the house, following the conventions of railroad's ProcTHOR
plots (https://github.com/RAIL-group/railroad): the robot's route is drawn
over the top-down view of the house, beside a sidebar with the plan cost, the
actions and the goal.

The route follows the same shortest paths that problem.compute_motion_costs
measures, so it ends at the plan's cost.
"""
import numpy as np
from matplotlib import colormaps, patheffects
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter
from skimage.morphology import erosion

from scenegraph_pddl import grid
from scenegraph_pddl.pddl import problem

PICK_PLACE_COST = 10  # as in domain.pddl
# railroad's colour for its first robot: 'Reds', skipping the near-white start
ROUTE_CMAP = LinearSegmentedColormap.from_list(
    'Reds_t', colormaps['Reds'](np.linspace(0.25, 1.0, 256)))
ROBOT_COLOR = ROUTE_CMAP(0.85)
TRAIL_POINTS = 2000
TRAIL_SIZE_START, TRAIL_SIZE_END = 25.0, 2.0  # marker areas; the trail tapers
TRAIL_OUTLINE_WIDTH = 1.4  # points of white halo, so the trail reads over the image
UNTRAVERSABLE_SHADE = 0.25  # black wash over cells the robot cannot stand on
BACKGROUND_GRAY = 0.75  # occupied cells when there is no top-down image
SKYBOX_LUMINANCE = 250  # the white sky around the house in the top-down image


def plan_route(house, graph, actions):
    """Replay a plan over the occupancy grid.

    Returns (points, costs, steps): points is an Nx2 array of (x, z) grid cells
    along the route, costs the plan cost spent on reaching each point (pick and
    place hold the robot still for their cost), and steps a list of (action,
    location, cost when the action starts).
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
            points.append(points[-1])
            costs.append(cost)
    return np.array(points), np.array(costs), steps


def render_plan(house, graph, actions, goals, cost):
    """Draw the plan as a matplotlib Figure (see the module docstring)."""
    points, point_costs, steps = plan_route(house, graph, actions)

    figure = Figure(figsize=(12.8, 7.2))
    FigureCanvasAgg(figure)
    layout = GridSpec(1, 2, figure=figure, width_ratios=[3, 1], wspace=0.1,
                      left=0.05, right=0.97, top=0.95, bottom=0.05)
    map_ax = figure.add_subplot(layout[0, 0])
    sidebar_ax = figure.add_subplot(layout[0, 1])
    sidebar_ax.set_axis_off()

    image_extent = _draw_house(map_ax, house)
    trail = _draw_route(map_ax, points, point_costs, cost)
    markers = _draw_locations(map_ax, problem.location_cells(house, graph), steps)
    _frame(map_ax, house, image_extent, np.concatenate([trail, markers]))
    map_ax.set_title(f'{problem.ROBOT} plan  (cost = {cost:.1f})', fontfamily='monospace', fontsize=10)

    _draw_sidebar(sidebar_ax, steps, problem.goal_facts(goals), cost)
    return figure


def save_plan_plot(house, graph, actions, goals, cost, path):
    """Write the plan plot to path (e.g. plan.png)."""
    render_plan(house, graph, actions, goals, cost).savefig(path, dpi=150, bbox_inches='tight')


def _draw_house(ax, house):
    """Draw the top-down view with untraversable cells darkened, or, without
    a view, the occupancy grid. Returns the extent of the house in the view."""
    occupancy_grid = house.occupancy_grid
    if house.top_down_image is None:
        ax.imshow(_grid_rgb(occupancy_grid), zorder=0)
        return None

    # The view is framed in grid cells, so it lines up with the grid
    ax.imshow(house.top_down_image, extent=house.top_down_extent, zorder=-1)
    image_extent = _image_content_extent(house.top_down_image, house.top_down_extent)
    # From above, furniture looks much like floor: darken what the robot cannot
    # stand on, out to the edge of the house in the view
    shade, shade_extent = _untraversable_shade(occupancy_grid, image_extent)
    ax.imshow(shade, extent=shade_extent, zorder=0.25)
    return image_extent


def _grid_rgb(occupancy_grid):
    """Free cells white, occupied cells gray with a black outline, transposed for imshow."""
    occupied = occupancy_grid.T >= 0.5
    outline = occupied ^ erosion(occupied, footprint=np.ones((3, 3)))
    rgb = np.full((*occupied.shape, 3), BACKGROUND_GRAY)
    rgb[~occupied] = 1
    rgb[outline] = 0
    return rgb


def _untraversable_shade(occupancy_grid, cover):
    """Black wash over occupied cells, padded out to the imshow extent cover
    (the view reaches past the grid, which spans only reachable cells).
    Returns the RGBA overlay, transposed for imshow, and its extent."""
    n_x, n_z = occupancy_grid.shape
    left, right, bottom, top = -0.5, n_x - 0.5, n_z - 0.5, -0.5
    pad = [int(np.ceil(max(0.0, gap))) for gap in
           (left - cover[0], cover[1] - right, top - cover[3], cover[2] - bottom)]
    padded = np.pad(occupancy_grid, ((pad[0], pad[1]), (pad[2], pad[3])), constant_values=1)
    rgba = np.zeros((*padded.shape, 4))
    rgba[padded >= 0.5, 3] = UNTRAVERSABLE_SHADE
    extent = (left - pad[0], right + pad[1], bottom + pad[3], top - pad[2])
    return rgba.transpose(1, 0, 2), extent


def _image_content_extent(image, extent):
    """The imshow extent of the image's non-sky pixels, i.e. of the house.

    The top-down camera is square, so a non-square house is padded with sky.
    """
    left, right, bottom, top = extent
    content = np.any(image[:, :, :3] < SKYBOX_LUMINANCE, axis=2)
    rows = np.flatnonzero(content.any(axis=1))
    columns = np.flatnonzero(content.any(axis=0))
    if rows.size == 0:
        return extent
    height, width = content.shape
    # Row 0 sits at top, as imshow draws it
    return (left + (right - left) * columns[0] / width,
            left + (right - left) * (columns[-1] + 1) / width,
            top + (bottom - top) * (rows[-1] + 1) / height,
            top + (bottom - top) * rows[0] / height)


def _draw_route(ax, points, point_costs, cost):
    """Draw the route as a trail of dots, coloured by the plan cost spent so far
    and shrinking along the way. Returns the trail's points."""
    trail_costs = np.linspace(0, cost, TRAIL_POINTS)
    trail = np.column_stack([np.interp(trail_costs, point_costs, points[:, 0]),
                             np.interp(trail_costs, point_costs, points[:, 1])])
    sizes = np.linspace(TRAIL_SIZE_START, TRAIL_SIZE_END, TRAIL_POINTS)
    # One white layer under the whole trail: a halo of constant width
    halo_sizes = (np.sqrt(sizes) + 2 * TRAIL_OUTLINE_WIDTH) ** 2
    ax.scatter(trail[:, 0], trail[:, 1], s=halo_sizes, c='white', linewidths=0, zorder=4.5)
    ax.scatter(trail[:, 0], trail[:, 1], s=sizes, c=ROUTE_CMAP(trail_costs / cost), zorder=5)
    return trail


def _draw_locations(ax, cells, steps):
    """Mark and label the robot's start and every location it visits. Returns their cells."""
    visited = [problem.BASE] + [location for _, location, _ in steps]
    markers = np.array([cells[location] for location in dict.fromkeys(visited)])
    ax.scatter(markers[:, 0], markers[:, 1], s=20, color='black', zorder=6)
    # Over the trail, with a white outline so they read over the view
    label_style = dict(textcoords='offset points', color='brown', zorder=7,
                       path_effects=[patheffects.withStroke(linewidth=1.5, foreground='white')])
    ax.annotate(problem.ROBOT, cells[problem.BASE], xytext=(3, -9), fontsize=7,
                fontweight='bold', **label_style)
    for location in dict.fromkeys(visited):
        ax.annotate(location, cells[location], xytext=(3, 3), fontsize=5, **label_style)
    return markers


def _frame(ax, house, image_extent, points):
    """Frame the grid, widened to the house in the view and to points, with
    the axes labelled in metres (the data stays in grid cells)."""
    n_x, n_z = house.occupancy_grid.shape
    left, right, bottom, top = -0.5, n_x - 0.5, n_z - 0.5, -0.5
    if image_extent is not None:
        left, right = min(left, image_extent[0]), max(right, image_extent[1])
        bottom, top = max(bottom, image_extent[2]), min(top, image_extent[3])
    left, right = min(left, points[:, 0].min()), max(right, points[:, 0].max())
    bottom, top = max(bottom, points[:, 1].max()), min(top, points[:, 1].min())
    ax.set_xlim(left, right)
    ax.set_ylim(bottom, top)  # z increases downwards
    ax.set_aspect('equal', adjustable='box')

    to_metres = FuncFormatter(lambda cells, _: f'{cells * house.grid_resolution:g}')
    ax.xaxis.set_major_formatter(to_metres)
    ax.yaxis.set_major_formatter(to_metres)
    ax.set_xlabel('x (m)', fontsize=8)
    ax.set_ylabel('z (m)', fontsize=8)


def _draw_sidebar(ax, steps, goal_facts, cost):
    """The route's colour as a cost scale, with each action placed at the cost
    it starts at, and the goal (all of it reached by the plan) below."""
    strip_left, strip_width = 0.05, 0.06
    goal_lines = ['AND('] + [f'  {fact}' for fact in goal_facts] + [')']
    goal_line_spacing = 0.035
    goal_top = 0.03 + (len(goal_lines) + 1) * goal_line_spacing + 0.04
    strip_bottom, strip_top = goal_top + 0.04, 0.88

    strip_ax = ax.inset_axes((strip_left, strip_bottom, strip_width, strip_top - strip_bottom))
    strip_ax.imshow(np.linspace(0, 1, 256).reshape(-1, 1), aspect='auto', cmap=ROUTE_CMAP,
                    origin='lower', extent=(0, 1, 0, cost))
    strip_ax.set_xlim(0, 1)
    strip_ax.set_ylim(cost, 0)
    strip_ax.set_xticks([])
    strip_ax.set_ylabel('cost', fontsize=6)
    strip_ax.tick_params(axis='y', labelsize=5)
    middle = strip_left + strip_width / 2
    ax.plot(middle, strip_top + 0.04, 'o', color=ROBOT_COLOR, markeredgecolor='black',
            markeredgewidth=0.8, markersize=8, transform=ax.transAxes, clip_on=False)
    ax.text(middle, strip_top + 0.08, problem.ROBOT, fontsize=5, fontfamily='monospace',
            fontweight='bold', ha='center', va='bottom', transform=ax.transAxes)

    # Actions beside the strip at their start cost, at least min_gap apart
    actions_left, min_gap = strip_left + strip_width + 0.08, 0.025
    y = strip_top + min_gap
    for index, (action, _, start_cost) in enumerate(steps):
        y = min(strip_top - start_cost / cost * (strip_top - strip_bottom), y - min_gap)
        ax.text(actions_left, y, f'{index + 1}. {" ".join(action)}', fontsize=5,
                fontfamily='monospace', ha='left', va='top', transform=ax.transAxes)

    ax.text(strip_left, goal_top - 0.01, 'Goal:', fontsize=7, fontfamily='monospace',
            fontweight='bold', ha='left', va='top', transform=ax.transAxes)
    for index, line in enumerate(goal_lines):
        color = 'green' if line.strip().startswith('(') else 'gray'
        ax.text(strip_left, goal_top - 0.01 - (index + 1) * goal_line_spacing, line, fontsize=6,
                fontfamily='monospace', color=color, ha='left', va='top', transform=ax.transAxes)
