"""Alias module — re-exports everything from girds.py.

Both our_astar_navigator.py and our_frontier_exploration.py import from
asl_tb3_lib.grids, but the file was committed as girds.py (typo). This shim
keeps both names working without renaming the original.
"""
from asl_tb3_lib.girds import snap_to_grid, StochOccupancyGrid2D

__all__ = ["snap_to_grid", "StochOccupancyGrid2D"]
