import random
from collections import deque

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - optional plotting dependency
    plt = None

try:
    import networkx as nx
except Exception:  # pragma: no cover - optional graph dependency
    nx = None

try:
    import numpy as np
except Exception:  # pragma: no cover - optional numeric dependency
    np = None


class GraphFallback:
    class NodeView:
        def __init__(self, graph):
            self.graph = graph

        def __call__(self):
            return list(self.graph._adjacency.keys())

    class EdgeView:
        def __init__(self, graph):
            self.graph = graph

        def __call__(self, node=None):
            if node is None:
                return [
                    (u, v, dict(attrs))
                    for u, edges in self.graph._adjacency.items()
                    for v, attrs in edges.items()
                ]
            return list(self.graph._adjacency.get(node, {}).keys())

        def data(self):
            return [
                (u, v, dict(attrs))
                for u, edges in self.graph._adjacency.items()
                for v, attrs in edges.items()
            ]

    def __init__(self):
        self._adjacency = {}
        self.nodes = self.NodeView(self)
        self.edges = self.EdgeView(self)

    def add_node(self, node):
        self._adjacency.setdefault(node, {})

    def add_edge(self, first, second, **attrs):
        self.add_node(first)
        self.add_node(second)
        self._adjacency[first][second] = dict(attrs)

    def has_path(self, start, end):
        if start == end:
            return True

        queue = deque([start])
        seen = {start}

        while queue:
            current = queue.popleft()
            for neighbor in self._adjacency.get(current, {}):
                if neighbor == end:
                    return True
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append(neighbor)

        return False

    def dijkstra_path_length(self, start, end):
        if start == end:
            return 0

        distances = {start: 0}
        queue = deque([start])

        while queue:
            current = queue.popleft()
            for neighbor in self._adjacency.get(current, {}):
                next_distance = distances[current] + 1
                if next_distance < distances.get(neighbor, float("inf")):
                    distances[neighbor] = next_distance
                    queue.append(neighbor)

        return distances.get(end, float("inf"))


class Building(object):
    def __init__(self, floors: int, width: int, length: int, difficulty: int = 3) -> None:
        assert isinstance(floors, int) and floors > 0, "floors must be positive"
        assert isinstance(width, int) and width > 0, "width must be positive"
        assert isinstance(length, int) and length > 0, "length must be positive"

        if not isinstance(difficulty, int) or difficulty not in range(1, 11):
            raise AssertionError("difficulty should be between 1 and 10")

        self.floors = floors
        self.width = width
        self.length = length

        self.start_pos = (self.length - 1, self.width - 1, self.floors - 1)
        self.exit_pos = (0, 0, 0)

        self.configure_difficulty(difficulty)

        graph = self.initialize_graph()
        path = self.generate_solution_path(
            graph=graph,
            start=self.start_pos,
            end=self.exit_pos,
            path=[],
            visited=set(),
            p_intra=self.p_intra_sol,
            p_inter=self.p_inter_sol,
            min_edges=self.min_edges,
            max_edges=self.max_edges,
        )
        if not path:
            raise RuntimeError("Failed to generate a valid escape path")

        self.path = list(map(tuple, path))

        graph = self.draw_path(graph=graph, path=self.path)
        self.display_graph(graph=graph)

        self.graph = self.populate_graph(
            graph=graph,
            path=self.path,
            non_solution_edges=self.non_solution_edges,
            p_intra_pop=self.p_intra_pop,
            p_inter_pop=self.p_inter_pop,
        )

        self.display_graph(graph=self.graph)

    def configure_difficulty(self, difficulty: int) -> None:
        self.min_edges = 5 * difficulty
        self.max_edges = self.min_edges + 10
        self.non_solution_edges = 50
        self.p_intra_sol = 0.9
        self.p_inter_sol = 0.1
        self.p_intra_pop = 0.99
        self.p_inter_pop = 0.01

    def initialize_graph(self):
        if nx is not None:
            graph = nx.grid_graph(dim=(self.floors, self.width, self.length))
            graph = nx.DiGraph(nx.create_empty_copy(graph))
            graph.has_path = lambda source, target: nx.has_path(graph, source, target)
            graph.dijkstra_path_length = lambda source, target: nx.dijkstra_path_length(graph, source, target)
            return graph

        graph = GraphFallback()
        for z in range(self.floors):
            for y in range(self.width):
                for x in range(self.length):
                    graph.add_node((x, y, z))
        return graph

    def _intra_neighbors(self, node):
        neighbors = []
        deltas = [(-1, 0), (0, -1), (0, 1), (1, 0)]
        for dx, dy in deltas:
            x = max(0, min(node[0] + dx, self.length - 1))
            y = max(0, min(node[1] + dy, self.width - 1))
            candidate = (x, y, node[2])
            if candidate != node:
                neighbors.append(candidate)
        return neighbors

    def _inter_neighbors(self, node):
        neighbors = []
        for z in range(self.floors):
            if z == node[2]:
                continue
            candidate = (node[0], node[1], z)
            if 0 <= candidate[0] < self.length and 0 <= candidate[1] < self.width:
                neighbors.append(candidate)
        return neighbors

    def generate_solution_path(
        self,
        graph,
        start,
        end,
        path,
        visited,
        min_edges: int = 2,
        max_edges: int = 10,
        p_intra: float = 0.9,
        p_inter: float = 0.1,
    ):
        if start == end:
            if min_edges <= len(path) <= max_edges:
                return path + [end]
            return []

        if len(path) >= max_edges:
            return []

        path = path + [start]
        visited.add(start)

        order = ["intra", "inter"]
        if random.random() < p_intra:
            order = ["intra", "inter"]
        else:
            order = ["inter", "intra"]

        for kind in order:
            if kind == "intra":
                neighbors = self._intra_neighbors(start)
                new_p_intra = p_intra * 0.9 if p_intra != 1 else self.p_intra_sol
                new_p_inter = (1 - new_p_intra) if p_inter != 0 else self.p_inter_sol
            else:
                neighbors = self._inter_neighbors(start)
                new_p_intra = 1.0
                new_p_inter = 0.0

            random.shuffle(neighbors)
            for neighbor in neighbors:
                if neighbor in visited:
                    continue

                solution = self.generate_solution_path(
                    graph,
                    neighbor,
                    end,
                    path,
                    visited,
                    min_edges=min_edges,
                    max_edges=max_edges,
                    p_intra=new_p_intra,
                    p_inter=new_p_inter,
                )
                if solution:
                    return solution

        visited.remove(start)
        return []

    def populate_graph(self, graph, path, non_solution_edges: int, p_intra_pop: float, p_inter_pop: float):
        all_connected_nodes = list(path)
        solution_length = len(path) - 1

        for _ in range(non_solution_edges):
            if not all_connected_nodes:
                break

            node = random.choice(all_connected_nodes)
            neighbors = self._intra_neighbors(node) + self._inter_neighbors(node)
            valid_neighbors = []
            for neighbor in neighbors:
                if neighbor in all_connected_nodes:
                    continue
                if graph.has_path(node, neighbor) or graph.has_path(neighbor, node):
                    continue
                if graph.has_path(self.start_pos, node) and graph.has_path(neighbor, self.exit_pos):
                    if graph.dijkstra_path_length(self.start_pos, node) + 1 + graph.dijkstra_path_length(neighbor, self.exit_pos) <= solution_length:
                        continue
                if graph.has_path(self.start_pos, neighbor) and graph.has_path(node, self.exit_pos):
                    if graph.dijkstra_path_length(self.start_pos, neighbor) + 1 + graph.dijkstra_path_length(node, self.exit_pos) <= solution_length:
                        continue
                valid_neighbors.append(neighbor)

            if not valid_neighbors:
                continue

            new_node = random.choice(valid_neighbors)
            if new_node[2] == node[2]:
                graph.add_edge(node, new_node, kind="floor")
                graph.add_edge(new_node, node, kind="floor")
            elif new_node[2] > node[2]:
                graph.add_edge(node, new_node, kind="stair")
            else:
                graph.add_edge(node, new_node, kind="slide")

            all_connected_nodes.append(new_node)

        return graph

    def draw_path(self, graph, path):
        for node_start, node_end in zip(path[:-1], path[1:]):
            if node_start[-1] < node_end[-1]:
                graph.add_edge(node_start, node_end, kind="stair")
            elif node_start[-1] > node_end[-1]:
                graph.add_edge(node_start, node_end, kind="slide")
            else:
                graph.add_edge(node_start, node_end, kind="floor")
                graph.add_edge(node_end, node_start, kind="floor")
        return graph

    def display_graph(self, graph) -> None:
        if plt is None:
            return

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        nodes = list(graph.nodes())
        xs = [node[0] for node in nodes]
        ys = [node[1] for node in nodes]
        zs = [node[2] for node in nodes]

        ax.scatter(xs, ys, zs, color='blue', s=40, depthshade=False)

        for edge in graph.edges.data():
            u, v, attrs = edge
            kind = attrs.get("kind", "floor")
            if kind == "slide":
                color = 'green'
            elif kind == "stair":
                color = 'red'
            else:
                color = 'gray'

            ax.plot([u[0], v[0]], [u[1], v[1]], [u[2], v[2]], color=color, alpha=0.8, linewidth=2)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_box_aspect([1, 1, 1])

        plt.show()
        return


if __name__ == "__main__":
    n, x, y = 2, 5, 5
    diff = 5
    building = Building(n, x, y, difficulty=diff)
