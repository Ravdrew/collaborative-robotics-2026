"""Grid compatibility shim that delegates to `mod_asl3_lib.grids` in this workspace.

This keeps the API used by existing code (`snap_to_grid`, `StochOccupancyGrid2D`).
"""
from typing import Any
import numpy as np

try:
    from mod_asl3_lib.grids import snap_to_grid as _snap, StochOccupancyGrid2D as _Stoch
except Exception:
    # best-effort fallback: define minimal versions
    def _snap(state: np.ndarray, resolution: float) -> np.ndarray:
        return resolution * np.round(state / resolution)

    class _Stoch:
        def __init__(self, resolution, size_xy, origin_xy, window_size, probs, thresh=0.5):
            self.resolution = resolution
            self.size_xy = size_xy
            self.origin_xy = origin_xy
            self.probs = np.reshape(np.asarray(probs), (size_xy[1], size_xy[0]))
            self.window_size = window_size
            self.thresh = thresh

        def state2grid(self, state_xy: np.ndarray) -> np.ndarray:
            state_snapped_xy = _snap(state_xy, self.resolution)
            grid_xy = ((state_snapped_xy - self.origin_xy) / self.resolution).astype(int)
            return grid_xy

        def is_free(self, state_xy: np.ndarray) -> bool:
            grid_xy = self.state2grid(state_xy)
            if grid_xy[0] < 0 or grid_xy[1] < 0 or grid_xy[0] >= self.size_xy[0] or grid_xy[1] >= self.size_xy[1]:
                return False
            return (self.probs[grid_xy[1], grid_xy[0]] / 100.0) < self.thresh


def snap_to_grid(state: Any, resolution: float) -> Any:
    return _snap(state, resolution)


class StochOccupancyGrid2D(_Stoch):
    pass
