import sys

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.stats import norm

class Building(object):
    def __init__(self, 
                 floors: int, 
                 width: int, 
                 length: int,
                 difficulty: int = 3) -> None:
        assert isinstance(floors, int) and floors > 0, "floors must be positive"
        assert isinstance(width, int) and width > 0, "width must be positive"
        assert isinstance(length, int) and length > 0, "length must be positive"
        assert isinstance(difficulty, int) and difficulty in np.arange(1, 11), "difficulty should be between 1 and 10"
        
        self.floors = floors
        self.width = width
        self.length = length
        
        self.start_pos = (self.length-1, self.width-1, self.floors-1)
        self.exit_pos = (0, 0, 0)
        
        settings_kw = self.configure_difficulty(difficulty)
        
        graph = self.initialize_graph()
        path = self.generate_solution_path(graph=graph, 
                                           start=self.start_pos, 
                                           end=self.exit_pos, 
                                           path=np.empty((0, 3), dtype = int), 
                                           visited=set(), 
                                           p_intra=self.p_intra_sol,
                                           p_inter=self.p_inter_sol,
                                           min_edges=self.min_edges,
                                           max_edges=self.max_edges)
        self.path = list(map(tuple, path.tolist()))
        
        graph = self.draw_path(graph=graph, path=self.path)
        self.display_graph(graph=graph)
        
        self.graph = self.populate_graph(graph=graph, 
                                         path=self.path, 
                                         non_solution_edges=self.non_solution_edges,
                                         p_intra_pop=self.p_intra_pop,
                                         p_inter_pop=self.p_inter_pop)
        
        self.display_graph(graph=self.graph)

    def configure_difficulty(self, difficulty: int) -> None:
        volume = self.length * self.width * self.floors # TODO: Length, width and floors could be adjusted by difficulty
        self.min_edges = 5 * difficulty
        self.max_edges = self.min_edges + 10
        self.non_solution_edges = 50
        self.p_intra_sol = 0.9
        self.p_inter_sol = 0.1
        self.p_intra_pop = 0.99
        self.p_inter_pop = 0.01
        
        return
    
    def initialize_graph(self) -> nx.DiGraph:
        # Create a graph with no edges
        graph = nx.grid_graph(dim = (self.floors, self.width, self.length))
        graph = nx.DiGraph(nx.create_empty_copy(graph))
        
        return graph
    
    def generate_solution_path(self,
                               graph: nx.DiGraph, 
                               start: set, 
                               end: set, 
                               path: list,
                               visited: set,
                               min_edges: int = 2,
                               max_edges: int = 10,
                               p_intra: float = 0.9,
                               p_inter: float = 0.1) -> np.ndarray:
        no_solution = np.empty((0, 3), dtype = int)
        
        # Stop criterion
        if (start == end):
            if (min_edges <= len(path) <= max_edges):
                path = np.append(path, [end], axis = 0)
                return path
            else:
                return no_solution
        
        if len(path) >= max_edges:
            return no_solution
        
        # Add our current node to the path
        path = np.append(path, [start], axis = 0)
        visited.add(start)
        
        # Draw from a categorical distr, whether to ascend/descend/stay
        order = np.random.choice(["intra", "inter"], 2, p = [p_intra, p_inter], replace=False) if p_intra != 1 else ["intra"]
        for c in order:
            if c == "intra":
                # Only look at valid neighbors on the same floor
                non_diag_neigh = [[-1, 0], [0, -1], [0, 1], [1, 0]]
                avail_neigh = set((max(0, min(start[0]+i, self.length-1)), max(0, min(start[1]+j, self.width-1)), start[-1]) for i, j in non_diag_neigh)
                avail_neigh.discard(start)
                avail_neigh = list(avail_neigh)
                assert not np.any([avail_neigh[i] > (self.length-1, self.width-1, self.floors-1) for i in range(len(avail_neigh))]), "One or more of the node neighbors exceed the building's dimensions"

                # Make it more likely to ascend/descend after an intra-step
                new_p_intra = p_intra * 0.9 if p_intra != 1 else self.p_intra_sol
                new_p_inter = (1 - new_p_intra) if p_inter != 0 else self.p_inter_sol
            else:
                avail_neigh = np.indices((self.length, self.width, self.floors)).reshape(3, -1).T
                avail_neigh = avail_neigh[avail_neigh[...,-1] != start[-1]]
                avail_neigh = list(map(tuple, avail_neigh.tolist()))

                # An ascend or descend should always be followed by an intra-floor step
                new_p_intra = 1
                new_p_inter = 0
            
            # TODO: Make the avail neigh have a weight from a gaussian kernel and make 
            # np.random.choice where size = len(avail_neigh)
            # Shuffle neighbors to get more curvy paths
            rng = np.random.default_rng()
            rng.shuffle(avail_neigh)

            for n in avail_neigh:
                if n not in visited:
                    solution = self.generate_solution_path(graph,
                                                            n,
                                                            end,
                                                            path,
                                                            visited,
                                                            min_edges=min_edges,
                                                            max_edges=max_edges,
                                                            p_intra=new_p_intra,
                                                            p_inter=new_p_inter)
                    if solution.size != 0:
                        return solution

        
        return no_solution

                

    def populate_graph(self, 
                       graph: nx.Graph,
                       path: list[tuple],
                       non_solution_edges: int,
                       p_intra_pop: float,
                       p_inter_pop: float) -> nx.DiGraph:
        all_connected_nodes = path.copy()
        solution_length = nx.dijkstra_path_length(graph, self.start_pos, self.exit_pos)
        
        for _ in range(non_solution_edges):
            # Choose a random node from all connected nodes
            node_idx = np.random.randint(len(all_connected_nodes), size = 1)
            node = all_connected_nodes[node_idx.item()]
            
            # We cannot ascend/descend when on the solution path nor when there's already a stair/slide
            p_intra = 1 if (node in path) or (np.any([u[-1] != v[-1] for u, v in graph.edges(node)])) else p_intra_pop
            p_inter = 0 if (node in path) or (np.any([u[-1] != v[-1] for u, v in graph.edges(node)])) else p_inter_pop
            
            # Draw from a categorical distr, whether to ascend/descend/stay
            order = np.random.choice(["intra", "inter"], 1, p = [p_intra, p_inter], replace=False)
            if order == "intra":
                # Only look at valid neighbors on the same floor
                non_diag_neigh = [[-1, 0], [0, -1], [0, 1], [1, 0]]
                avail_neigh = set((max(0, min(node[0]+i, self.length-1)), max(0, min(node[1]+j, self.width-1)), node[-1]) for i, j in non_diag_neigh)
                avail_neigh.discard(node)
                avail_neigh = list(avail_neigh)
                assert not np.any([avail_neigh[i] > (self.length-1, self.width-1, self.floors-1) for i in range(len(avail_neigh))]), "One or more of the node neighbors exceed the building's dimensions"
            else:
                avail_neigh = np.indices((self.length, self.width, self.floors)).reshape(3, -1).T
                avail_neigh = avail_neigh[avail_neigh[...,-1] != node[-1]]
                avail_neigh = list(map(tuple, avail_neigh.tolist()))

            # Prune available neighbors
            for n in avail_neigh.copy():
                # We don't want open spaces/loops
                if nx.has_path(graph, node, n):
                    avail_neigh.remove(n)
                elif nx.has_path(graph, n, node):
                    avail_neigh.remove(n)
                
                # We do not want to accidentally create a shortcut or another solution path than the intended
                elif nx.has_path(graph, self.start_pos, node) and nx.has_path(graph, n, self.exit_pos):
                    if (nx.dijkstra_path_length(graph, self.start_pos, node) + 1 + \
                        nx.dijkstra_path_length(graph, n, self.exit_pos)
                        <= solution_length):
                            avail_neigh.remove(n)
                
                elif nx.has_path(graph, self.start_pos, n) and nx.has_path(graph, node, self.exit_pos):
                    if (nx.dijkstra_path_length(graph, self.start_pos, n) + 1 + \
                        nx.dijkstra_path_length(graph, node, self.exit_pos)
                        <= solution_length):
                            avail_neigh.remove(n)
            
            # TODO: Make the avail neigh have a weight from a gaussian kernel and make 
            # np.random.choice where size = len(avail_neigh)
            # Only if there are neighbors to connect
            if avail_neigh:
                new_node_idx = np.random.randint(len(avail_neigh))
                new_node = avail_neigh[new_node_idx]
                
                self.draw_path(graph, [node, new_node])    
                all_connected_nodes.append(new_node)
        
        return graph

    def draw_path(self,
                  graph: nx.DiGraph,
                  path: list[tuple]) -> nx.DiGraph:
        # Draw solution path
        for node_start, node_end in zip(path[:-1], path[1:]):
            # Ascend
            if node_start[-1] < node_end[-1]:
                graph.add_edge(node_start, node_end, kind="stair")
            # Descend
            elif node_start[-1] > node_end[-1]:
                graph.add_edge(node_start, node_end, kind="slide")
            # Stay
            else:
                graph.add_edge(node_start, node_end, kind="floor")
                graph.add_edge(node_end, node_start, kind="floor")
        
        return graph

    def display_graph(self, graph: nx.DiGraph) -> None:

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        nodes = list(graph.nodes())

        xs = [n[0] for n in nodes]
        ys = [n[1] for n in nodes]
        zs = [n[2] for n in nodes]

        ax.scatter(xs, ys, zs, color='blue', s=40, depthshade=False)

        for edge in graph.edges.data():
            u = edge[0]
            v = edge[1]
            kind = edge[2]["kind"]

            if kind == "slide":
                color = 'green'
            elif kind == "stair":
                color = 'red'
            else:
                color = 'gray'

            # Draw the complete edge
            ax.plot(
                [u[0], v[0]],
                [u[1], v[1]],
                [u[2], v[2]],
                color=color,
                alpha=0.8,
                linewidth=2
            )

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        ax.set_box_aspect([1, 1, 1])

        plt.show()

        return

if __name__ == "__main__":
    n, x, y = 2, 5, 5 # sys.argv[1:4]
    diff = 5
    building = Building(n, x, y, difficulty=diff)