import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from scipy.stats import gamma

class Building(object):
    def __init__(self,
                 difficulty: int = 3,
                 verbose: bool = False) -> None:
        assert isinstance(difficulty, int) and difficulty in np.arange(1, 11), "difficulty should be between 1 and 10"
        
        self.configure_difficulty(difficulty)
        self.start_pos = (self.length-1, self.width-1, self.floors-1)
        self.exit_pos = (0, 0, 0)
        
        invalid_graph = True
        while invalid_graph:
            graph = self.initialize_graph()
            path = self.generate_path(graph=graph, 
                                            start=self.start_pos, 
                                            targets=[self.exit_pos], 
                                            path=np.empty((0, 3), dtype = int), 
                                            visited=set(), 
                                            p_intra=self.p_intra_sol,
                                            p_inter=self.p_inter_sol,
                                            min_edges=self.min_edges,
                                            max_edges=self.max_edges)
            self.path = list(map(tuple, path.tolist()))
            
            self.graph = self.update_graph(graph=graph, path=self.path)
            
            if verbose:
                self.display_graph(graph=graph)
            
            self.graph, self.all_valid_notes = self.populate_graph(graph=self.graph, 
                                            path=self.path, 
                                            non_solution_edges=self.non_solution_edges,
                                            p_intra_pop=self.p_intra_pop,
                                            p_inter_pop=self.p_inter_pop)
            losing_nodes, unreachable_nodes = self.validate_graph()
            
            if not self.allow_dead_ends:
                for start in losing_nodes:
                    path = self.generate_path(graph=graph, 
                                            start=start, 
                                            targets=self.path, 
                                            path=np.empty((0, 3), dtype = int), 
                                            visited=set(), 
                                            p_intra=self.p_intra_sol,
                                            p_inter=self.p_inter_sol,
                                            min_edges=1,
                                            max_edges=self.max_edges) # TODO: Consider edge range according to difficulty
                    path = list(map(tuple, path.tolist()))
                    self.graph = self.update_graph(graph=graph, path=path)
                    
                    _, unreachable_nodes = self.validate_graph()
                    if unreachable_nodes:
                        print("Invalid graph: There are now nodes that are unreachable from the start")
                
                losing_nodes, _ = self.validate_graph()
                if losing_nodes:
                    print("Invalid graph: There are still dead ends that cannot reach the exit")
                
            if unreachable_nodes == [] and (self.allow_dead_ends or losing_nodes == []):
                invalid_graph = False
            
        if verbose:
            self.display_graph(graph=self.graph)
            print("Level created")

    def configure_difficulty(self, difficulty: int) -> None:
        self.floors = 4
        self.width = 10
        self.length = 10
        volume = self.length * self.width * self.floors # TODO: Length, width and floors could be adjusted by difficulty

        # TODO: According to Dimitra, the floors should by square
        self.min_edges = 5 * difficulty
        self.max_edges = self.min_edges + 10
        self.non_solution_edges = 100
        self.p_intra_sol = 0.9
        self.p_inter_sol = 0.1
        self.p_intra_pop = 0.99
        self.p_inter_pop = 0.01
        self.k = 1
        self.std = 1
        self.allow_dead_ends = False
        
        return
    
    def initialize_graph(self) -> nx.DiGraph:
        # Create a graph with no edges
        graph = nx.grid_graph(dim = (self.floors, self.width, self.length))
        graph = nx.DiGraph(nx.create_empty_copy(graph))
        nx.set_node_attributes(graph, "wall", "type")
        nx.set_node_attributes(graph, True, "dead_end")
        graph.nodes[self.start_pos]["type"] = "floor"
        
        return graph

    @staticmethod
    def gaussian_weight(u: tuple[int], v: tuple[int], std: float = 1.0) -> float:
        return np.exp(-1/2 * np.linalg.norm(u - v, axis = 1)**2 / std**2)
    
    def node_weight(self, target_nodes: list[tuple[int]], k: float = 1) -> float:
        """Returns the weight for a source node given by 
        the gamma function where the input is the shortest
        path length from the closest node in the solution path
        to the target node

        Args:
            target_nodes (list[tuple[int]]): list of target node(s)

        Returns:
            float: the weight(s)
        """
        # Run a shortest path algorithm
        all_distances = []
        for node in target_nodes:
            distance, path = nx.multi_source_dijkstra(self.graph, self.path, node, weight = "weight")
            all_distances.append(distance)
        all_distances = np.asarray(all_distances)
        
        # Compute weights by the gamma function # TODO: Reconsider weight function choice (not the real gamma one)
        return 1 + all_distances * np.exp(-k*np.asarray(all_distances))
        
    
    def generate_path(self,
                      graph: nx.DiGraph, 
                      start: tuple[int], 
                      targets: list[tuple[int]], 
                      path: list,
                      visited: set,
                      min_edges: int = 2,
                      max_edges: int = 10,
                      p_intra: float = 0.9,
                      p_inter: float = 0.1) -> np.ndarray:
        no_solution = np.empty((0, 3), dtype = int)
        
        # Stop criterion
        if (start in targets):
            if (min_edges <= len(path) <= max_edges):
                path = np.append(path, [start], axis = 0)
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
            avail_neigh = self.get_neighbors(start, step=c)
            new_p_intra, new_p_inter = self.update_transition_probabilities(p_intra, p_inter, step=c)
            
            # Reorder neighbors according to their distance from the source node
            w_neigh = self.gaussian_weight(np.repeat([start], len(avail_neigh), axis = 0), avail_neigh, std = self.std)
            w_neigh = w_neigh / sum(w_neigh)
            neigh_order_idx = np.random.choice(np.arange(len(avail_neigh)), size = len(avail_neigh), replace = False, p = w_neigh)

            for idx in neigh_order_idx:
                n = avail_neigh[idx]
                if n not in visited and (graph.nodes[n]["type"] == "wall" or (n in targets and graph.nodes[n]["type"] == "floor")):
                    solution = self.generate_path(graph,
                                                  n,
                                                  targets,
                                                  path,
                                                  visited,
                                                  min_edges=min_edges,
                                                  max_edges=max_edges,
                                                  p_intra=new_p_intra,
                                                  p_inter=new_p_inter)
                    if solution.size != 0:
                        return solution

        visited.remove(start)
        return no_solution

                

    def populate_graph(self, 
                       graph: nx.Graph,
                       path: list[tuple],
                       non_solution_edges: int,
                       p_intra_pop: float,
                       p_inter_pop: float) -> nx.DiGraph:
        # Define nodes to expand. Cannot be stair/slide head nodes nor the exit node
        all_valid_nodes = path[:-1].copy() 
        for i in range(len(path) - 1):
            valid = path[i][-1] == path[i + 1][-1]
            if not valid:
                all_valid_nodes.remove(path[i])
        all_weights = self.node_weight(all_valid_nodes, k=self.k)
        
        # TODO: Check that the weights make sense
        for _ in range(non_solution_edges):
            if not all_valid_nodes:
                break
            
            # (Re)-normalize the weights
            normalized_weights = all_weights / np.sum(all_weights)
            node_type = nx.get_node_attributes(graph, "type")
            node_dead_end = nx.get_node_attributes(graph, "dead_end")
            
            # Categorically sample a node from all connected nodes
            node_idx = np.random.choice(np.arange(len(all_valid_nodes)), size = 1, p = normalized_weights)
            node = all_valid_nodes[node_idx.item()]
            
            # We can only ascend/descend on dead ends which are not stair/slide heads/tails
            p_intra = 1 if (not node_dead_end[node] or node_type[node] in ("stair_head", "stair_tail", "slide_head", "slide_tail")) else p_intra_pop
            p_inter = 0 if (p_intra == 1) else p_inter_pop
            
            # Draw from a categorical distr, whether to ascend/descend/stay
            order = np.random.choice(["intra", "inter"], 2, p = [p_intra, p_inter], replace=False) if p_intra != 1 else ["intra"]
            for c in order:
                avail_neigh = self.get_neighbors(node, step=c)
                avail_neigh = self.prune_neighbors(graph, node, avail_neigh)
                    
                if avail_neigh:
                    break
                    
            # Only if there are neighbors to connect
            if avail_neigh:
                w_neigh = self.gaussian_weight(np.repeat([node], len(avail_neigh), axis = 0), avail_neigh, std = self.std)
                w_neigh = w_neigh / sum(w_neigh)
                new_node_idx = np.random.choice(np.arange(len(avail_neigh)), size = 1, p = w_neigh)
                new_node = avail_neigh[new_node_idx.item()]
                
                # Add the new path
                assert nx.has_path(graph, self.start_pos, node), (
                f"Selected node {node} is unreachable before expansion"
                )
                self.update_graph(graph, [node, new_node])  
                assert nx.has_path(graph, self.start_pos, new_node), (
                f"New node {new_node} became unreachable after expansion"
                )
                  
                all_valid_nodes.append(new_node)
                all_weights = np.append(all_weights, self.node_weight([new_node], k=self.k))
                
                # If we made a stair/slide, it's no longer valid
                if c == "inter":
                    all_weights = np.delete(all_weights, all_valid_nodes.index(node))
                    all_valid_nodes.remove(node)                  
            else:
                all_weights = np.delete(all_weights, all_valid_nodes.index(node))
                all_valid_nodes.remove(node)
        
        return graph, all_valid_nodes

    def get_neighbors(self, node: tuple[int], step: str) -> list[tuple[int]]:
        if step == "intra":
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
        
        return avail_neigh

    def prune_neighbors(self,
                        graph: nx.DiGraph,
                        node: tuple[int],
                        neigbors: list[tuple[int]]) -> list[tuple[int]]:
        # Prune available neighbors
        for n in neigbors.copy():                    
            # We don't want open spaces/loops # TODO: Consider whether this should be included
            if nx.has_path(graph, node, n):
                neigbors.remove(n)
                continue
            
            if nx.has_path(graph, n, node):
                neigbors.remove(n)
                continue
            
            # For a bidirectional connection, we need to check both paths
            edge_check = [(node, n)]
            if node[-1] == n[-1]:
                edge_check.append((n, node))
            
            for u, v in edge_check:
                # We do not want to accidentally create a shortcut or another solution path than the intended
                if nx.has_path(graph, self.start_pos, u) and nx.has_path(graph, v, self.exit_pos):
                    if (nx.dijkstra_path_length(graph, self.start_pos, u) + 1 + \
                        nx.dijkstra_path_length(graph, v, self.exit_pos)
                        <= len(self.path)):
                            neigbors.remove(n)
                            break
        
        return neigbors
    
    def update_transition_probabilities(self, 
                                        p_intra: float, 
                                        p_inter: float, 
                                        step: str) -> tuple[float]:
        if step == "intra":
            # Make it more likely to ascend/descend after an intra-step
            new_p_intra = p_intra * 0.9 if p_intra != 1 else self.p_intra_sol
            new_p_inter = (1 - new_p_intra) if p_inter != 0 else self.p_inter_sol
        else:
            # An ascend or descend should always be followed by an intra-floor step
            new_p_intra = 1
            new_p_inter = 0
        
        return new_p_intra, new_p_inter
        

    def validate_graph(self) -> tuple[list[tuple[int]], list[tuple[int]]]:
        reachable = nx.descendants(self.graph, self.start_pos)
        reachable.add(self.start_pos)
        unreachable = []
        dead_ends = []
        
        for u in self.graph.nodes:
            if self.graph.nodes[u]["type"] == "wall":
                continue
            
            # Check for dead ends where the goal isn't reachable
            if self.graph.nodes[u]["dead_end"] and not nx.has_path(self.graph, u, self.exit_pos):
                dead_ends.append(u)
            
            # Check for unreachable nodes from the start
            if u not in reachable:
                unreachable.append(u)
        
        return dead_ends, unreachable

    def update_graph(self,
                     graph: nx.DiGraph,
                     path: list[tuple]) -> nx.DiGraph:
        # Draw solution path
        for node_start, node_end in zip(path[:-1], path[1:]):
            graph.nodes[node_start]["dead_end"] = False
            
            # Ascend
            if node_start[-1] < node_end[-1]:
                graph.remove_edges_from(list(graph.out_edges(node_start)))
                graph.add_edge(node_start, node_end, type="stair")
                graph.nodes[node_start]["type"] = "stair_head"
                graph.nodes[node_end]["type"] = "stair_tail"
            # Descend
            elif node_start[-1] > node_end[-1]:
                graph.remove_edges_from(list(graph.out_edges(node_start)))
                graph.add_edge(node_start, node_end, type="slide")
                graph.nodes[node_start]["type"] = "slide_head"
                graph.nodes[node_end]["type"] = "slide_tail"
            # Stay
            else:        
                attr = nx.get_node_attributes(graph, "type")
                graph.add_edge(node_start, node_end, type="floor")
                
                graph.nodes[node_end]["type"] = "floor"
                
                # If the node we go to ascends/descends or is the exit, it shouldn't be bidirectional
                if not (attr[node_end] == "stair_head" or attr[node_end] == "slide_head" or node_end == self.exit_pos):
                    graph.add_edge(node_end, node_start, type="floor")
            
            if node_end == self.exit_pos:
                graph.nodes[node_end]["dead_end"] = False
            
        return graph

    def display_graph(self, graph: nx.DiGraph) -> None:

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Fixed arrowhead length in data coordinates
        arrow_size = 0.2

        for u, v, data in graph.edges(data=True):

            kind = data["type"]

            if kind == "slide":
                color = 'green'
            elif kind == "stair":
                color = 'red'
            else:
                color = 'gray'

            # Direction vector
            dx = v[0] - u[0]
            dy = v[1] - u[1]
            dz = v[2] - u[2]

            length = np.sqrt(dx**2 + dy**2 + dz**2)

            # Unit direction vector
            dx /= length
            dy /= length
            dz /= length

            # Draw complete edge
            ax.plot(
                [u[0], v[0]],
                [u[1], v[1]],
                [u[2], v[2]],
                color=color,
                alpha=0.8,
                linewidth=2
            )

            # Draw arrowhead with fixed length
            ax.quiver(
                v[0] - arrow_size * dx,
                v[1] - arrow_size * dy,
                v[2] - arrow_size * dz,
                arrow_size * dx,
                arrow_size * dy,
                arrow_size * dz,
                color=color,
                arrow_length_ratio=0.5,
                linewidth=2,
                normalize=False
            )

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        ax.set_box_aspect([1, 1, 1])

        plt.show()

if __name__ == "__main__":
    diff = 2
    building = Building(difficulty=diff, verbose = True)
    
    # TODO: Quantitatively evaluate e.g
    # 1) Number of off-roads and how long they are for different difficulty settings
    # 2) Parameter sweeps