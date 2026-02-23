"""Navigation base classes for trajectory planning and control.

This module provides the base classes and data structures used by the navigation
stack, including BaseNavigator for ROS2-based navigation nodes and TrajectoryPlan
for representing computed trajectories.
"""

from abc import ABC
from typing import NamedTuple, Optional
import numpy as np


class TrajectoryPlan(NamedTuple):
    """Represents a planned trajectory from A* and spline fitting.
    
    Attributes:
        path (np.ndarray): Array of shape (N, 2) containing waypoints as (x, y) coordinates
        path_x_spline: Spline representation for x-coordinates from scipy.interpolate.splrep
        path_y_spline: Spline representation for y-coordinates from scipy.interpolate.splrep
        duration (float): Total time duration for executing the trajectory in seconds
    """
    path: np.ndarray
    path_x_spline: tuple
    path_y_spline: tuple
    duration: float


class BaseNavigator(ABC):
    """Base class for navigation nodes in ROS2.
    
    Provides common functionality and interface for trajectory planning and tracking
    controllers. Subclasses implement specific navigation behaviors.
    """
    
    def __init__(self) -> None:
        """Initialize the navigator base class."""
        self.t_prev = 0.0
        self.V_prev = 0.0  # Linear velocity
        self.om_prev = 0.0  # Angular velocity
    
    def reset(self) -> None:
        """Reset internal variables for trajectory tracking controller."""
        self.t_prev = 0.0
        self.V_prev = 0.0
        self.om_prev = 0.0
