"""Generate a building and export its accepted geometry as JSON."""

import argparse
from pathlib import Path

from src.json_export import export_building_json
from src.building_generation import generate_building
from src.wfc_building import WFCBuilding


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="new JSON file; must not already exist")
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--grid-resolution", type=int, default=3)
    parser.add_argument("--max-building-retries", type=int, default=0)
    args = parser.parse_args()

    options = dict(
        floors=4,
        width=10,
        length=10,
        difficulty=2,
        seed=args.seed,
        grid_resolution=args.grid_resolution,
        max_route_attempts=30,
        max_route_backtrack_depth=2,
        max_route_recovery_attempts=1,
        recovery_strategy="none",
        minimum_curvature_radius=0.02,
        endpoint_frontier=True,
    )
    if args.max_building_retries:
        building = generate_building(
            max_building_retries=args.max_building_retries, **options
        ).building
        print(
            f"Requested seed {building.requested_seed}; accepted seed "
            f"{building.accepted_seed} after {building.building_retry_count} retries."
        )
    else:
        building = WFCBuilding(**options)
    export_building_json(building, args.output)
    print(f"Exported {len(building.routes)} accepted routes to {args.output}")


if __name__ == "__main__":
    main()
