#!/usr/bin/env python3
"""Test script to verify A* pathfinding algorithm"""

import numpy as np
import sys
import os

# Add scripts directory to path
scripts_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, scripts_dir)

from asl_tb3_lib.girds import StochOccupancyGrid2D
from our_astar_navigator import AStar


def test_astar_simple_path():
    """Test A* finds a path in a simple empty grid"""
    print("\nTest 1: Simple Path in Empty Grid")
    print("=" * 50)
    
    resolution = 1.0
    size_xy = np.array([20, 20], dtype=int)
    origin_xy = np.array([0.0, 0.0])
    probs = np.zeros(size_xy[0] * size_xy[1], dtype=float)
    
    occupancy = StochOccupancyGrid2D(
        resolution=resolution,
        size_xy=size_xy,
        origin_xy=origin_xy,
        window_size=3,
        probs=probs.tolist(),
        thresh=0.5,
    )
    
    x_init = (1.0, 1.0)
    x_goal = (18.0, 18.0)
    
    astar = AStar(
        statespace_lo=(0.0, 0.0),
        statespace_hi=(20.0, 20.0),
        x_init=x_init,
        x_goal=x_goal,
        occupancy=occupancy,
        resolution=1.0
    )
    
    found = astar.solve()
    print(f"Initial state: {x_init}")
    print(f"Goal state: {x_goal}")
    print(f"Path found: {found}")
    
    if found:
        path = np.array(astar.path)
        print(f"Path length: {len(astar.path)} waypoints")
        print(f"Start: {astar.path[0]}")
        print(f"End: {astar.path[-1]}")
        # Verify path starts and ends correctly
        assert np.allclose(astar.path[0], x_init), "Path should start at initial state"
        assert np.allclose(astar.path[-1], x_goal), "Path should end at goal state"
        print("✓ PASS: A* found valid path in empty grid")
    else:
        print("✗ FAIL: A* should find path in empty grid")
        return False
    
    return True


def test_astar_obstacle_avoidance():
    """Test A* avoids obstacles"""
    print("\nTest 2: Obstacle Avoidance")
    print("=" * 50)
    
    resolution = 1.0
    size_xy = np.array([20, 20], dtype=int)
    origin_xy = np.array([0.0, 0.0])
    
    # Create grid with obstacle in the middle
    probs = np.zeros(size_xy[0] * size_xy[1], dtype=float)
    probs = probs.reshape((size_xy[1], size_xy[0]))
    # 100 = completely occupied
    probs[8:12, 8:12] = 100.0
    probs = probs.flatten().tolist()
    
    occupancy = StochOccupancyGrid2D(
        resolution=resolution,
        size_xy=size_xy,
        origin_xy=origin_xy,
        window_size=3,
        probs=probs,
        thresh=0.5,
    )
    
    x_init = (1.0, 1.0)
    x_goal = (18.0, 18.0)
    
    astar = AStar(
        statespace_lo=(0.0, 0.0),
        statespace_hi=(20.0, 20.0),
        x_init=x_init,
        x_goal=x_goal,
        occupancy=occupancy,
        resolution=1.0
    )
    
    found = astar.solve()
    print(f"Initial state: {x_init}")
    print(f"Goal state: {x_goal}")
    print(f"Obstacle: centered at (10, 10)")
    print(f"Path found: {found}")
    
    if found:
        path = np.array(astar.path)
        print(f"Path length: {len(astar.path)} waypoints")
        print(f"Start: {astar.path[0]}")
        print(f"End: {astar.path[-1]}")
        # Verify path avoids obstacle
        for point in astar.path:
            if 8 <= point[0] <= 12 and 8 <= point[1] <= 12:
                print("✗ FAIL: Path crosses obstacle!")
                return False
        print("✓ PASS: A* successfully avoided obstacle")
    else:
        print("✗ FAIL: A* should find path around obstacle")
        return False
    
    return True


def test_astar_no_path():
    """Test A* handles case where no path exists"""
    print("\nTest 3: No Path Exists")
    print("=" * 50)
    
    resolution = 1.0
    size_xy = np.array([20, 20], dtype=int)
    origin_xy = np.array([0.0, 0.0])
    
    # Create grid with walls blocking path
    probs = np.zeros(size_xy[0] * size_xy[1], dtype=float)
    probs = probs.reshape((size_xy[1], size_xy[0]))
    # Wall from top to bottom through the middle
    probs[0:20, 9:11] = 100.0
    probs = probs.flatten().tolist()
    
    occupancy = StochOccupancyGrid2D(
        resolution=resolution,
        size_xy=size_xy,
        origin_xy=origin_xy,
        window_size=3,
        probs=probs,
        thresh=0.5,
    )
    
    x_init = (1.0, 1.0)
    x_goal = (18.0, 18.0)
    
    astar = AStar(
        statespace_lo=(0.0, 0.0),
        statespace_hi=(20.0, 20.0),
        x_init=x_init,
        x_goal=x_goal,
        occupancy=occupancy,
        resolution=1.0
    )
    
    found = astar.solve()
    print(f"Initial state: {x_init}")
    print(f"Goal state: {x_goal}")
    print(f"Blocking wall: x in [9, 11] for all y")
    print(f"Path found: {found}")
    
    if not found:
        print("✓ PASS: A* correctly reported no path exists")
        return True
    else:
        print("✗ FAIL: A* should not find path through wall")
        return False


def test_astar_neighbors():
    """Test A* neighbor generation (8-directional)"""
    print("\nTest 4: Neighbor Generation (8-directional)")
    print("=" * 50)
    
    resolution = 1.0
    size_xy = np.array([20, 20], dtype=int)
    origin_xy = np.array([0.0, 0.0])
    probs = np.zeros(size_xy[0] * size_xy[1], dtype=float)
    
    occupancy = StochOccupancyGrid2D(
        resolution=resolution,
        size_xy=size_xy,
        origin_xy=origin_xy,
        window_size=3,
        probs=probs.tolist(),
        thresh=0.5,
    )
    
    x = (10.0, 10.0)
    
    astar = AStar(
        statespace_lo=(0.0, 0.0),
        statespace_hi=(20.0, 20.0),
        x_init=x,
        x_goal=(15.0, 15.0),
        occupancy=occupancy,
        resolution=1.0
    )
    
    neighbors = astar.get_neighbors(x)
    print(f"State: {x}")
    print(f"Number of neighbors: {len(neighbors)}")
    print(f"Neighbors: {sorted(neighbors)}")
    
    # In an empty 20x20 grid, interior point should have 8 neighbors
    if len(neighbors) == 8:
        print("✓ PASS: A* generates 8 neighbors in open space")
        return True
    else:
        print(f"✗ FAIL: Expected 8 neighbors, got {len(neighbors)}")
        return False


def main():
    print("\n" + "=" * 50)
    print("A* Algorithm Testing Suite")
    print("=" * 50)
    
    results = []
    
    results.append(("Simple Path", test_astar_simple_path()))
    results.append(("Obstacle Avoidance", test_astar_obstacle_avoidance()))
    results.append(("No Path Exists", test_astar_no_path()))
    results.append(("Neighbor Generation", test_astar_neighbors()))
    
    print("\n" + "=" * 50)
    print("Test Summary")
    print("=" * 50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ All A* tests passed!")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
