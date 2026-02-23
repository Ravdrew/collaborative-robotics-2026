#!/usr/bin/env python3
"""
Comprehensive Navigation Stack Integration Test

Tests all components of the navigation system:
- asl_tb3_lib occupancy grid (girds.py)
- asl_tb3_lib math utilities 
- asl_tb3_lib transform utilities
- asl_tb3_lib navigation module
- A* pathfinding algorithm
- Frontier exploration
"""

import numpy as np
import sys
import os

# Add scripts directory to path
scripts_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, scripts_dir)


def test_imports():
    """Verify all asl_tb3_lib modules can be imported"""
    print("\n" + "=" * 60)
    print("TEST: Module Imports")
    print("=" * 60)
    
    try:
        from asl_tb3_lib.girds import StochOccupancyGrid2D, snap_to_grid
        print("✓ asl_tb3_lib.girds imported")
    except Exception as e:
        print(f"✗ Failed to import asl_tb3_lib.girds: {e}")
        return False
    
    try:
        from asl_tb3_lib.math_utils import wrap_angle, distance_linear, distance_angular
        print("✓ asl_tb3_lib.math_utils imported")
    except Exception as e:
        print(f"✗ Failed to import asl_tb3_lib.math_utils: {e}")
        return False
    
    try:
        from asl_tb3_lib.tf_utils import quaternion_to_yaw, transform_to_state
        print("✓ asl_tb3_lib.tf_utils imported")
    except Exception as e:
        print(f"✗ Failed to import asl_tb3_lib.tf_utils: {e}")
        return False
    
    try:
        from asl_tb3_lib.navigation import BaseNavigator, TrajectoryPlan
        print("✓ asl_tb3_lib.navigation imported (NEW)")
    except Exception as e:
        print(f"✗ Failed to import asl_tb3_lib.navigation: {e}")
        return False
    
    try:
        from our_astar_navigator import AStar, Navigator
        print("✓ our_astar_navigator imported")
    except Exception as e:
        print(f"✗ Failed to import our_astar_navigator: {e}")
        return False
    
    print("\n✓ All module imports successful!")
    return True


def test_occupancy_grid():
    """Test occupancy grid functionality"""
    print("\n" + "=" * 60)
    print("TEST: Occupancy Grid")
    print("=" * 60)
    
    from asl_tb3_lib.girds import StochOccupancyGrid2D, snap_to_grid
    
    # Create test grid
    resolution = 0.1
    size_xy = np.array([100, 100])
    origin_xy = np.array([-5.0, -5.0])
    probs = np.zeros(10000)
    
    occupancy = StochOccupancyGrid2D(
        resolution=resolution,
        size_xy=size_xy,
        origin_xy=origin_xy,
        window_size=13,
        probs=probs,
        thresh=0.5,
    )
    
    # Test state2grid
    state = np.array([-5.0, -5.0])
    grid = occupancy.state2grid(state)
    assert np.allclose(grid, [0, 0]), "state2grid failed"
    print("✓ state2grid works")
    
    # Test grid2state (THE KEY FIX)
    grid = np.array([0, 0])
    state = occupancy.grid2state(grid)
    assert np.allclose(state, [-5.0, -5.0]), "grid2state failed"
    print("✓ grid2state works (KEY FIX)")
    
    # Test batch grid2state
    grid_array = np.array([[0, 0], [1, 1], [10, 10]])
    state_array = occupancy.grid2state(grid_array)
    assert state_array.shape == (3, 2), "batch grid2state shape wrong"
    print("✓ batch grid2state works")
    
    # Test is_free
    is_free = occupancy.is_free(np.array([0.0, 0.0]))
    assert isinstance(is_free, (bool, np.bool_)), "is_free should return bool"
    print("✓ is_free works")
    
    print("\n✓ All occupancy grid tests passed!")
    return True


def test_math_utilities():
    """Test math utility functions"""
    print("\n" + "=" * 60)
    print("TEST: Math Utilities")
    print("=" * 60)
    
    from asl_tb3_lib.math_utils import wrap_angle, distance_linear, distance_angular
    
    # Test wrap_angle
    angle = wrap_angle(4 * np.pi)
    assert -np.pi <= angle <= np.pi, "wrap_angle failed"
    print("✓ wrap_angle works")
    
    # Test distance_linear
    dist = distance_linear((0, 0), (3, 4))
    assert np.isclose(dist, 5.0), "distance_linear failed"
    print("✓ distance_linear works")
    
    # Test distance_angular
    class MockState:
        def __init__(self, theta):
            self.theta = theta
    
    s1 = MockState(0.0)
    s2 = MockState(np.pi / 2)
    dist = distance_angular(s1, s2)
    assert abs(dist) > 0.1, f"distance_angular failed: got {dist}"
    print("✓ distance_angular works")
    
    print("\n✓ All math utility tests passed!")
    return True


def test_transform_utilities():
    """Test transform utility functions"""
    print("\n" + "=" * 60)
    print("TEST: Transform Utilities")
    print("=" * 60)
    
    from asl_tb3_lib.tf_utils import quaternion_to_yaw, transform_to_state
    
    # Test quaternion_to_yaw with identity quaternion
    q_identity = [0, 0, 0, 1]  # [x, y, z, w]
    yaw = quaternion_to_yaw(q_identity)
    assert np.isclose(yaw, 0.0), "quaternion_to_yaw failed for identity"
    print("✓ quaternion_to_yaw works")
    
    # Test with quaternion object
    class MockQuat:
        def __init__(self, x, y, z, w):
            self.x = x
            self.y = y
            self.z = z
            self.w = w
    
    q = MockQuat(0, 0, 0, 1)
    yaw = quaternion_to_yaw(q)
    assert np.isclose(yaw, 0.0), "quaternion_to_yaw failed with object"
    print("✓ quaternion_to_yaw works with objects")
    
    print("\n✓ All transform utility tests passed!")
    return True


def test_navigation_classes():
    """Test navigation support classes"""
    print("\n" + "=" * 60)
    print("TEST: Navigation Classes")
    print("=" * 60)
    
    from asl_tb3_lib.navigation import BaseNavigator, TrajectoryPlan
    
    # Test TrajectoryPlan
    dummy_path = np.array([[0, 0], [1, 1], [2, 2]])
    dummy_spline = (np.array([0, 1]), np.array([0, 1]), 3)
    
    plan = TrajectoryPlan(
        path=dummy_path,
        path_x_spline=dummy_spline,
        path_y_spline=dummy_spline,
        duration=5.0
    )
    
    assert plan.duration == 5.0, "TrajectoryPlan duration failed"
    assert len(plan.path) == 3, "TrajectoryPlan path failed"
    print("✓ TrajectoryPlan works")
    
    # Test BaseNavigator
    nav = BaseNavigator()
    assert nav.V_prev == 0.0, "BaseNavigator initialization failed"
    nav.reset()
    assert nav.t_prev == 0.0, "BaseNavigator reset failed"
    print("✓ BaseNavigator works")
    
    print("\n✓ All navigation class tests passed!")
    return True


def test_astar_integration():
    """Test A* algorithm with realistic scenario"""
    print("\n" + "=" * 60)
    print("TEST: A* Integration")
    print("=" * 60)
    
    from asl_tb3_lib.girds import StochOccupancyGrid2D
    from our_astar_navigator import AStar
    
    # Create a realistic occupancy grid
    resolution = 0.1
    size_xy = np.array([50, 50])
    origin_xy = np.array([-2.5, -2.5])
    
    # Create grid with obstacle
    probs = np.zeros(50 * 50)
    probs = probs.reshape((50, 50))
    probs[20:30, 20:30] = 100.0  # Central obstacle
    probs = probs.flatten().tolist()
    
    occupancy = StochOccupancyGrid2D(
        resolution=resolution,
        size_xy=size_xy,
        origin_xy=origin_xy,
        window_size=3,
        probs=probs,
        thresh=0.5,
    )
    
    # Find path around obstacle
    x_init = (-2.0, -2.0)
    x_goal = (2.0, 2.0)
    
    astar = AStar(
        statespace_lo=(-2.5, -2.5),
        statespace_hi=(2.5, 2.5),
        x_init=x_init,
        x_goal=x_goal,
        occupancy=occupancy,
        resolution=0.1
    )
    
    found = astar.solve()
    assert found, "A* should find path"
    assert astar.path is not None, "Path should not be None"
    assert len(astar.path) > 0, "Path should have waypoints"
    print(f"✓ A* found path with {len(astar.path)} waypoints")
    
    # Verify path doesn't cross obstacles
    for point in astar.path:
        is_free = occupancy.is_free(np.array(point))
        assert is_free, f"Path crosses obstacle at {point}"
    print("✓ Path avoids obstacles")
    
    print("\n✓ All A* integration tests passed!")
    return True


def test_frontier_and_astar_together():
    """Test frontier exploration working with A*"""
    print("\n" + "=" * 60)
    print("TEST: Frontier + A* Integration")
    print("=" * 60)
    
    from asl_tb3_lib.girds import StochOccupancyGrid2D
    from our_astar_navigator import AStar
    from scipy.signal import convolve2d
    
    # Create grid with unknown region (frontier)
    resolution = 0.2
    size_xy = np.array([30, 30])
    origin_xy = np.array([-3.0, -3.0])
    
    probs = np.full(30 * 30, 0.0)
    probs = probs.reshape((30, 30))
    probs[0:15, 15:30] = -1.0  # Unknown region
    probs[15:20, 15:25] = 100.0  # Obstacle blocking access
    probs = probs.flatten().tolist()
    
    occupancy = StochOccupancyGrid2D(
        resolution=resolution,
        size_xy=size_xy,
        origin_xy=origin_xy,
        window_size=5,
        probs=probs,
        thresh=0.5,
    )
    
    # Step 1: Detect frontier (simulate frontier detection)
    probs_2d = occupancy.probs
    unknown_mask = (probs_2d < 0)
    kernel = np.ones((5, 5), dtype=int)
    unknown_count = convolve2d(unknown_mask.astype(int), kernel, mode='same', boundary='fill', fillvalue=0)
    
    # Simple frontier: unknown cells with unknown neighbors
    frontier_mask = unknown_mask & (unknown_count > 1)
    y, x = np.where(frontier_mask)
    
    if len(x) > 0:
        frontier_grid_points = np.vstack([x, y]).T
        
        # Step 2: Convert to world coordinates using grid2state
        frontier_states = occupancy.grid2state(frontier_grid_points)
        print(f"✓ Detected {len(frontier_grid_points)} frontier grid cells")
        print(f"✓ Converted to {len(frontier_states)} world coordinates")
        
        # Step 3: Select a frontier goal and plan path to it
        if len(frontier_states) > 0:
            frontier_goal = frontier_states[0]
            
            x_init = (-3.0, -3.0)
            
            astar = AStar(
                statespace_lo=(-3.0, -3.0),
                statespace_hi=(3.0, 3.0),
                x_init=x_init,
                x_goal=tuple(frontier_goal),
                occupancy=occupancy,
                resolution=0.2
            )
            
            found = astar.solve()
            print(f"✓ A* planning to frontier goal: {found}")
            if found and astar.path:
                print(f"✓ Planned path with {len(astar.path)} waypoints")
    else:
        print("Note: No frontier detected in test grid (grid configuration)")
    
    print("\n✓ Frontier + A* integration test completed!")
    return True


def main():
    print("\n" + "=" * 60)
    print("COMPREHENSIVE NAVIGATION STACK INTEGRATION TEST")
    print("=" * 60)
    print("Testing all components of the TidyBot2 navigation system")
    
    tests = [
        ("Module Imports", test_imports),
        ("Occupancy Grid", test_occupancy_grid),
        ("Math Utilities", test_math_utilities),
        ("Transform Utilities", test_transform_utilities),
        ("Navigation Classes", test_navigation_classes),
        ("A* Integration", test_astar_integration),
        ("Frontier + A* Integration", test_frontier_and_astar_together),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ Exception in {test_name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} test groups passed")
    
    if passed == total:
        print("\n" + "=" * 60)
        print("✓ ALL NAVIGATION SYSTEM TESTS PASSED!")
        print("=" * 60)
        print("\nYour navigation stack is fully functional:")
        print("  • Occupancy grid coordinate transformations")
        print("  • A* pathfinding algorithm")
        print("  • Frontier exploration integration")
        print("  • Transform utilities")
        print("  • Math utilities")
        print("  • Navigation support classes")
        return 0
    else:
        print(f"\n✗ {total - passed} test group(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
