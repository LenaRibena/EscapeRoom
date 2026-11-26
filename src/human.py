import networkx as nx
import numpy as np

from graph import Building

class Human():
    def __init__(self, exploration: float = 0.1) -> None:
        self.pos = None
        self.exploration = exploration
    
    def spawn(self, building: Building) -> None:
        self.pos = building.start_pos
    
    def despawn(self) -> None:
        self.pos = None
    
    def get_avaiable_actions(self, 
                             building: Building, 
                             pos: tuple[int]) -> list[tuple[tuple[int], tuple[int], dict[str, str]]]:
        return list(building.graph.out_edges(pos, data = True))
    
    def get_path(self, 
                 building: Building) -> tuple[list[tuple[int]], list[str]]:
        if self.pos is None:
            self.spawn(building)
        
        path = [self.pos]
        path_types = []
        
        optimal_path = building.path
        while self.pos != building.exit_pos:
            actions = self.get_avaiable_actions(building, self.pos)
            
            # Exploration/exploitation
            eps = np.random.rand()
            if eps < self.exploration:
                cur_state, new_state, edge_dict = actions[np.random.randint(len(actions))]
            else:
                # Update the optimal path if we are astray
                if self.pos not in optimal_path:
                    optimal_path = nx.dijkstra_path(building.graph, self.pos, building.exit_pos)

                next_action_idx = optimal_path.index(self.pos) + 1
                new_state = optimal_path[next_action_idx]
                edge_dict = building.graph.edges[(self.pos, new_state)]
            
            path.append(new_state)
            path_types.append(edge_dict["type"])
            
            self.pos = new_state
        
        return path, path_types
                

if __name__ == "__main__":
    player = Human(0.5)
    game = Building(2, 5, 5, 5, verbose = False)
    path = player.get_path(game)   
    print("Done")