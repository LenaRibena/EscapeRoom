import argparse
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from graph import Building
from utils import dfs, get_off_roads

def parse_args(argv):
    parser = argparse.ArgumentParser(prog="Difficulty assessment",
                                     description="Quantitative analysis of the generated graph's difficulty")
    parser.add_argument("--reps", type=int, default=200, help="Amount of repetitions for each difficulty")
    parser.add_argument("--difficulties", nargs="+", choices=["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "all"], default="all", help="Which difficulties to assess")
    parser.add_argument("--save-dir", type=str, default="data", help="Which directory to save the CSV file")
    
    return parser.parse_args(argv)

def main(argv=None):
    args = parse_args(argv)
    reps = args.reps
    difficulties = np.arange(1, 11) if args.difficulties == "all" else list(map(int, args.difficulties))
    save_dir = args.save_dir
    
    metrics = ["volume", "solution_length", "num_off_roads", "off_road_lengths", "dfs_time", "dfs_found_optimal", "dfs_visited_count"]

    # ----------------------- Compute all metrics -----------------------
    data = np.empty((reps, len(difficulties), len(metrics))) 
    for i in tqdm(range(reps)):
        for d in range(len(difficulties)):
            print(f"Difficulty: {d+1}")
            building = Building(d+1)
            
            data[i, d, 0] = building.length * building.width * building.floors
            data[i, d, 1] = len(building.path)
            data[i, d, 2] = building.graph.number_of_edges() - len(building.path)
            
            # Expected length of off-roads
            off_roads = get_off_roads(building)
            if not off_roads:
                print("No off-roads...")
            lengths = [len(road)-1 for road in off_roads]
            data[i, d, 3] = np.mean(lengths)
            
            search_results = dfs(building)
            if search_results["found"]:
                data[i, d, 4] = search_results["computation_time"]
                data[i, d, 5] = int(search_results["path"] == building.path)
                data[i, d, 6] = search_results["visited_count"]
            else:
                raise ValueError("Path not found in DFS!")

    # --------------------- Create DataFrame from data -----------------------
    idx_diffs = np.resize(difficulties, reps*len(difficulties))
    idx_reps = np.repeat(np.arange(reps), len(difficulties))
    index = pd.MultiIndex.from_arrays([idx_reps, idx_diffs], names=("reps", "diffs"))

    df = pd.DataFrame(data.reshape(reps*len(difficulties), -1), index=index, columns=metrics)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "df.csv")
    df.to_csv(path)

if __name__ == "__main__":
    main()