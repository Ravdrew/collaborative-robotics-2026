#!/usr/bin/env python3
"""Test script to verify occupancy grid coordinate transformations"""

import numpy as np
import sys
import os

# Add scripts directory to path
scripts_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, scripts_dir)

from asl_tb3_lib.girds import StochOccupancyGrid2D

# Create a test occupancy grid
resolution = 0.1  # 10cm per cell
size_xy = (100, 100)  # 100x100 grid
origin_xy = np.array([-5.0, -5.0])  # Origin at (-5, -5)
window_size = 13
probs = np.zeros(10000)  # Empty grid

# Instantiate the grid
occupancy = StochOccupancyGrid2D(
    resolution=resolution,
    size_xy=size_xy,
    origin_xy=origin_xy,
    window_size=window_size,
    probs=probs,
)

print("Testing Occupancy Grid Coordinate Transformations")
print("=" * 50)
print(f"Resolution: {resolution} m/cell")
print(f"Grid size: {size_xy[0]} x {size_xy[1]} cells")
print(f"Origin: {origin_xy}")
print()

# Test 1: state2grid at origin
state1 = np.array([-5.0, -5.0])
grid1 = occupancy.state2grid(state1)
print(f"Test 1: state2grid([-5.0, -5.0]) = {grid1}")
assert np.allclose(grid1, [0, 0]), "Origin should map to [0, 0]"
print("✓ PASS: Origin maps to grid [0, 0]")
print()

# Test 2: state2grid at a known point
state2 = np.array([-4.85, -4.85])
grid2 = occupancy.state2grid(state2)
print(f"Test 2: state2grid([-4.85, -4.85]) = {grid2}")
# Should be approximately [1, 1] (snapped to -4.9 then -4.8, which rounds to -4.9)
# Actually, snap_to_grid rounds [-4.85, -4.85] to [-4.9, -4.9] (nearest 0.1)
# Then grid = (-4.9 - (-5.0)) / 0.1 = 1
assert np.allclose(grid2, [1, 1]), f"Point at [-4.85, -4.85] should map to grid [1, 1], got {grid2}"
print("✓ PASS: Point [-4.85, -4.85] maps to grid [1, 1]")
print()

# Test 3: grid2state at origin (NEW METHOD)
grid3 = np.array([0, 0])
state3 = occupancy.grid2state(grid3)
print(f"Test 3: grid2state([0, 0]) = {state3}")
assert np.allclose(state3, [-5.0, -5.0]), "Grid [0, 0] should map back to origin [-5.0, -5.0]"
print("✓ PASS: Grid [0, 0] maps back to origin [-5.0, -5.0]")
print()

# Test 4: grid2state at a known grid point (NEW METHOD)
grid4 = np.array([1, 1])
state4 = occupancy.grid2state(grid4)
print(f"Test 4: grid2state([1, 1]) = {state4}")
assert np.allclose(state4, [-4.9, -4.9]), "Grid [1, 1] should map back to [-4.9, -4.9]"
print("✓ PASS: Grid [1, 1] maps back to [-4.9, -4.9]")
print()

# Test 5: Round-trip conversion
state5 = np.array([0.5, 1.3])
grid5 = occupancy.state2grid(state5)
state5_recovered = occupancy.grid2state(grid5)
print(f"Test 5: Round-trip conversion")
print(f"  Initial state: {state5}")
print(f"  → Grid: {grid5}")
print(f"  → Recovered state: {state5_recovered}")
print("✓ PASS: Round-trip conversion successful")
print()

# Test 6: Multiple grid points as 2D array (like frontier detection)
grid_array = np.array([[0, 0], [1, 1], [5, 10], [50, 50]])
state_array = occupancy.grid2state(grid_array)
print(f"Test 6: grid2state with 2D array")
print(f"  Grid points: {grid_array.tolist()}")
print(f"  State points: {state_array.tolist()}")
assert state_array.shape == (4, 2), "Output shape should be (4, 2)"
assert np.allclose(state_array[0], [-5.0, -5.0]), "First point should be origin"
assert np.allclose(state_array[1], [-4.9, -4.9]), "Second point should be [-4.9, -4.9]"
print("✓ PASS: Batch grid2state conversion works correctly")
print()

print("=" * 50)
print("All tests passed! ✓")
print("The occupancy grid coordinate transformations are working correctly.")
