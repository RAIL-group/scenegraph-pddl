"""Load a ProcTHOR-10k house into AI2-THOR and build its occupancy grid."""
import json
import os

import numpy as np
from ai2thor.controller import Controller
from shapely import geometry

from scenegraph_pddl import config, grid

GRID_RESOLUTION = 0.05  # meters per occupancy-grid cell
TOP_DOWN_MARGIN = 0.5  # meters of slack around the rooms in the top-down view
# Square size, in pixels, of the simulator's frames and so of the top-down view.
# As in railroad: the plot is saved at 300 dpi, which puts about 2000 pixels
# across the map, so a smaller view is upscaled and looks blurry.
TOP_DOWN_RENDER_PX = int(os.environ.get('PROCTHOR_TOP_DOWN_PX', 2048))
MAX_OBJECTS_PER_CONTAINER = 7
IGNORED_ASSETS = ['BaseballBat', 'BasketBall', 'Boots', 'DeskLamp',
                  'TennisRacket', 'Faucet', 'HousePlant', 'Dumbbell',
                  'FloorLamp', 'RoomDecor', 'ShowerCurtain',
                  'ShowerHead', 'Television', 'VacuumCleaner', 'Photo',
                  'Painting']


def load_house_record(seed, path=config.PROCTHOR_DATA_PATH):
    """The ProcTHOR-10k house on line `seed` of the dataset, as a dict."""
    with open(path) as f:
        for index, line in enumerate(f):
            if index == seed:
                return json.loads(line)
    raise IndexError(f'{path} has no house {seed}')


def asset_name(asset_id):
    """Generic name of a ProcTHOR asset id, e.g. 'Fridge|2|1' -> 'fridge'."""
    return asset_id.split('|')[0].lower()


def is_ignored(name):
    return any(asset.lower() in name for asset in IGNORED_ASSETS)


class ProcTHORHouse:
    """A ProcTHOR-10k house with an occupancy grid built from AI2-THOR.

    Attributes:
        rooms, doors: the house's room and door records. Each room gets a
            'position': the grid cell nearest its centre.
        containers: container records (furniture, appliances, ...), each with
            a grid 'position' and its 'children' (the objects it holds).
        occupancy_grid: 0 for cells the agent can reach, 1 otherwise.
        robot_position: grid cell of the agent's start position.
        grid_to_world: maps each free grid cell to its AI2-THOR position.
        top_down_image, top_down_extent: set by render_top_down().
        controller: the AI2-THOR controller; call close() once done.
    """

    def __init__(self, seed, preprocess=True):
        self.seed = seed
        self.grid_resolution = GRID_RESOLUTION
        self.record = load_house_record(seed)
        self.rooms = self.record['rooms']
        self.doors = self.record['doors']
        self.containers = self.record['objects']
        self.controller = Controller(scene=self.record,
                                     gridSize=self.grid_resolution,
                                     width=TOP_DOWN_RENDER_PX, height=TOP_DOWN_RENDER_PX)
        self.occupancy_grid = self._build_occupancy_grid()
        if preprocess:
            self.containers = self._preprocess_containers(self.containers)

        agent_position = self.record['metadata']['agent']['position']
        self.robot_position = self.to_grid((agent_position['x'], agent_position['z']))
        self.top_down_image = None
        self.top_down_extent = None

    def close(self):
        """Stop the AI2-THOR simulator."""
        self.controller.stop()

    def render_top_down(self):
        """Render an orthographic top-down image of the house (needs the simulator).

        Sets top_down_image, with rows running along +z, and top_down_extent,
        its (left, right, bottom, top) edges in grid cells for imshow, so it
        lines up with the occupancy grid plotted as occupancy_grid.T.
        """
        # Frame the room floor polygons, square because the frame is square
        corners = [(vertex['x'], vertex['z']) for room in self.rooms for vertex in room['floorPolygon']]
        xs, zs = zip(*corners)
        center_x, center_z = (min(xs) + max(xs)) / 2, (min(zs) + max(zs)) / 2
        half = max(max(xs) - min(xs), max(zs) - min(zs)) / 2 + TOP_DOWN_MARGIN

        event = self.controller.step(action='GetMapViewCameraProperties', raise_for_failure=True)
        pose = event.metadata['actionReturn']
        bounds = event.metadata['sceneBounds']['size']
        event = self.controller.step(
            action='AddThirdPartyCamera',
            position={'x': center_x, 'y': pose['position']['y'] + 1.1 * max(bounds['x'], bounds['z']),
                      'z': center_z},
            rotation={'x': 90, 'y': 0, 'z': 0},  # straight down, +x right and +z up
            fieldOfView=50, orthographic=True, orthographicSize=half,
            farClippingPlane=50, skyboxColor='white', raise_for_failure=True)
        self.top_down_image = event.third_party_camera_frames[-1][::-1]

        def to_cells(x, z):
            # Cell centres sit at integer coordinates, so edges are half a cell out
            return ((x - self.grid_offset[0]) / self.grid_resolution - 0.5,
                    (z - self.grid_offset[1]) / self.grid_resolution - 0.5)
        left, bottom = to_cells(center_x - half, center_z - half)
        right, top = to_cells(center_x + half, center_z + half)
        # imshow with origin='upper': the first row (smallest z) goes at the top
        self.top_down_extent = (left, right, top, bottom)
        return self.top_down_image

    def to_grid(self, point):
        """Grid cell of a world (x, z) point."""
        return (round((point[0] - self.grid_offset[0]) / self.grid_resolution),
                round((point[1] - self.grid_offset[1]) / self.grid_resolution))

    def path_cost(self, start, goal):
        """Length, in cells, of the shortest path between two grid cells (inf if none)."""
        occupancy_grid = self.occupancy_grid.copy()
        occupancy_grid[start] = 0
        occupancy_grid[goal] = 0
        path = grid.shortest_path(occupancy_grid, start, goal)
        return float('inf') if path is None else grid.path_length(path)

    def _build_occupancy_grid(self):
        event = self.controller.step(action="GetReachablePositions")
        reachable_positions = event.metadata["actionReturn"]
        xs = [position["x"] for position in reachable_positions]
        zs = [position["z"] for position in reachable_positions]

        min_x, max_x = min(xs), max(xs)
        min_z, max_z = min(zs), max(zs)
        x_offset = min_x - self.grid_resolution if min_x < 0 else 0
        z_offset = min_z - self.grid_resolution if min_z < 0 else 0
        self.grid_offset = np.array([x_offset, z_offset])

        points = list(zip(xs, zs))
        self.grid_to_world = {self.to_grid(point): reachable_positions[idx]
                              for idx, point in enumerate(points)}
        rows, cols = self.to_grid((max_x, max_z))
        occupancy_grid = np.ones((rows + 2, cols + 2), dtype=int)
        for cell in self.grid_to_world:
            occupancy_grid[cell] = 0

        # Place containers (and the objects in them) on the nearest free cell
        for container in self.containers:
            position = container['position']
            nearest = _nearest_point((position['x'], position['z']), points)
            container['position'] = self.to_grid(nearest)
            for child in container.get('children') or []:
                child['position'] = container['position']

        for room in self.rooms:
            centroid = geometry.Polygon(
                [(point['x'], point['z']) for point in room['floorPolygon']]).centroid
            room['position'] = self.to_grid(_nearest_point((centroid.x, centroid.y), points))

        return occupancy_grid

    def _preprocess_containers(self, containers):
        """Drop ignored and empty containers, give repeated container names a
        number ('unique_name'), keep each object name once across the house and
        at most MAX_OBJECTS_PER_CONTAINER objects per container."""
        containers = [c for c in containers if not is_ignored(asset_name(c['id']))]
        containers = sorted(containers, key=lambda c: len(c['children']) if 'children' in c else 0)

        container_names = set()
        object_names = set()
        num_duplicates = 0
        kept_containers = []
        for container in containers:
            for key in ('kinematic', 'rotation', 'layer', 'material'):
                container.pop(key, None)

            name = asset_name(container['id'])
            if name in container_names:
                num_duplicates += 1
                container['unique_name'] = name + str(num_duplicates)
            else:
                container_names.add(name)
                container['unique_name'] = name

            children = container.get('children')
            if children is None:
                continue

            kept_children = []
            for child in children:
                child_name = asset_name(child['id'])
                if is_ignored(child_name) or child_name in container_names or child_name in object_names:
                    continue
                object_names.add(child_name)
                child['unique_name'] = child_name
                kept_children.append(child)
                if len(kept_children) == MAX_OBJECTS_PER_CONTAINER:
                    break

            container['children'] = kept_children
            kept_containers.append(container)
        return kept_containers


def _nearest_point(point, points):
    """The point in `points` closest to `point` (the first one on ties)."""
    squared_distances = [(x - point[0])**2 + (z - point[1])**2 for x, z in points]
    return points[int(np.argmin(squared_distances))]
