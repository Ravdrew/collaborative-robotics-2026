import numpy as np


def wrap_angle(a: float) -> float:
    """Wrap angle to [-pi, pi]"""
    return float(np.arctan2(np.sin(a), np.cos(a)))


def distance_linear(s1, s2) -> float:
    """Linear Euclidean distance between two objects with x,y attributes or 2-tuples."""
    try:
        x1, y1 = s1.x, s1.y
        x2, y2 = s2.x, s2.y
    except Exception:
        x1, y1 = s1[0], s1[1]
        x2, y2 = s2[0], s2[1]
    return float(np.hypot(x1 - x2, y1 - y2))


def distance_angular(s1, s2) -> float:
    """Angular difference between two states with theta attribute or 3-tuple."""
    try:
        t1, t2 = s1.theta, s2.theta
    except Exception:
        t1, t2 = s1[2], s2[2]
    return float(wrap_angle(t1 - t2))
