"""Scene graph of a ProcTHOR house: apartment -> rooms -> containers -> objects.

A scene graph is a dict:
    nodes:      list of {'id', 'name', 'type', 'position'}; a node's index is its id
                in edges. 'type' is one of NODE_TYPES; 'position' is a grid cell.
    edges:      list of (parent, child) node indices, plus (room, room) for rooms
                joined by a door.
    containers: indices of container nodes.
    objects:    indices of object nodes.
add_node_features() adds 'node_features': one vector per node.
"""
import functools
import io
import json
import os

import networkx as nx
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from PIL import Image

from procgraph import config
from procgraph.house import asset_name

NODE_TYPES = ('apartment', 'room', 'container', 'object')
NODE_COLORS = {'apartment': 'red', 'room': 'blue', 'container': 'green', 'object': 'orange'}


def build_scene_graph(house):
    """Build the scene graph of a ProcTHORHouse."""
    nodes = [{'id': 'Apartment|0', 'name': 'apartment', 'type': 'apartment', 'position': None}]
    edges = []

    for room in house.rooms:
        edges.append((0, len(nodes)))
        nodes.append({'id': room['id'], 'name': room['roomType'].lower(),
                      'type': 'room', 'position': room['position']})

    room_nodes = range(1, len(nodes))
    edges.extend(sorted(
        (i, j) for i in room_nodes for j in room_nodes
        if i < j and _rooms_connected(house, nodes[i]['id'], nodes[j]['id'])))

    # Containers attach to the room whose number appears in their id
    room_node_by_number = {}
    for idx, node in enumerate(nodes):
        room_node_by_number.setdefault(_room_number(node['id']), idx)

    containers = []
    container_node_by_id = {}
    for container in house.containers:
        edges.append((room_node_by_number[_room_number(container['id'])], len(nodes)))
        container_node_by_id[container['id']] = len(nodes)
        containers.append(len(nodes))
        nodes.append({'id': container['id'], 'name': asset_name(container['id']),
                      'type': 'container', 'position': container['position']})

    objects = []
    for container in house.containers:
        for child in container.get('children') or []:
            edges.append((container_node_by_id[container['id']], len(nodes)))
            objects.append(len(nodes))
            nodes.append({'id': child['id'], 'name': asset_name(child['id']),
                          'type': 'object', 'position': child['position']})

    return {'nodes': nodes, 'edges': edges, 'containers': containers, 'objects': objects}


def add_node_features(graph):
    """Give every node a feature vector: the sentence-BERT embedding of its name
    followed by a one-hot encoding of its type."""
    graph['node_features'] = [
        np.concatenate((embed_text(node['name']), _one_hot_type(node['type'])))
        for node in graph['nodes']]
    return graph


def render_scene_graph(graph):
    """Draw the graph (nodes labelled by name, coloured by type) as a PIL image."""
    nx_graph = nx.Graph()
    nx_graph.add_nodes_from(range(len(graph['nodes'])))
    nx_graph.add_edges_from(graph['edges'])

    figure = Figure()
    FigureCanvasAgg(figure)
    nx.draw(nx_graph, nx.spring_layout(nx_graph, seed=0), ax=figure.add_subplot(),
            with_labels=True, node_size=150, font_size=8, edge_color='black',
            node_color=[NODE_COLORS[node['type']] for node in graph['nodes']],
            labels={idx: node['name'] for idx, node in enumerate(graph['nodes'])})
    buffer = io.BytesIO()
    figure.savefig(buffer, format='png')
    buffer.seek(0)
    return Image.open(buffer)


def save_scene_graph(graph, output_dir):
    """Write scene_graph.json (without node features) and scene_graph.png."""
    with open(os.path.join(output_dir, 'scene_graph.json'), 'w') as f:
        json.dump({key: graph[key] for key in ('nodes', 'edges', 'containers', 'objects')},
                  f, indent=2)
    render_scene_graph(graph).save(os.path.join(output_dir, 'scene_graph.png'))


@functools.lru_cache(maxsize=None)
def embed_text(text):
    return _sentence_model().encode([text])[0]


@functools.lru_cache(maxsize=1)
def _sentence_model():
    # Imported lazily: loading sentence-transformers (and torch) is slow
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(config.SENTENCE_MODEL_DIR)


def _one_hot_type(node_type):
    encoding = np.zeros(len(NODE_TYPES))
    encoding[NODE_TYPES.index(node_type)] = 1
    return encoding


def _room_number(node_id):
    """Room number in an id, e.g. 'room|3' -> 3 and 'Fridge|3|1' -> 3."""
    return int(node_id.split('|')[1])


def _rooms_connected(house, room_a, room_b):
    return any({door['room0'], door['room1']} == {room_a, room_b} for door in house.doors)
