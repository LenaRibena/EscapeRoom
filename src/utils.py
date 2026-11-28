from time import perf_counter

import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm

from graph import Building



def get_off_roads(building: Building) -> dict:
        
    def build_offroad(node, path=None):
        if path is None:
            path = []

        path = path + [node]

        # Reached a leaf in the DFS tree
        if node not in successors or not successors[node]:
            return [path]

        branches = []

        for neighbor in successors[node]:
            branches.extend(build_offroad(neighbor, path))

        return branches
            

    offroad_nodes = set(building.graph.nodes) - set(building.path)
    sub_graphs = building.graph.subgraph(offroad_nodes)

    all_offroads = []
    for node in building.path:
        for _, neigh in building.graph.out_edges(node):
            
            if neigh not in sub_graphs.nodes:
                continue

            successors = nx.dfs_successors(
                sub_graphs,
                source=neigh,
            )
            
            # E.g [1, 2, [3, 4], [5, 6]] -> [[1, 2, 3, 4], [1, 2, 5, 6]]
            off_roads = build_offroad(neigh, [])
            for road in off_roads:
                
                # Insert solution path node
                road.insert(1, node)
                
            all_offroads.extend(off_roads)

    return all_offroads

def dfs(building: Building) -> dict:
    visited = set()
    parent = {}

    start_time = perf_counter()

    successors = nx.dfs_successors(
        building.graph,
        source=building.start_pos,
        sort_neighbors=sorted
    )

    def search(node):
        visited.add(node)
        
        if node == building.exit_pos:
            return True

        # Dead end
        if node not in successors:
            return False

        for neighbor in successors[node]:
            if neighbor not in visited:
                parent[neighbor] = node

                if search(neighbor):
                    return True

        return False

    found = search(building.start_pos)

    elapsed_time = perf_counter() - start_time

    # Reconstruct the path
    solution = []

    if found:
        node = building.exit_pos

        while node != building.start_pos:
            solution.append(node)
            node = parent[node]

        solution.append(building.start_pos)
        solution.reverse()

    return {
        "found": found,
        "path": solution,
        "visited_count": len(visited),
        "computation_time": elapsed_time
    }