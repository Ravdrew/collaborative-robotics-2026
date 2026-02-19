import math


def quaternion_to_yaw(q) -> float:
    """Convert quaternion-like object to yaw angle.

    Accepts objects with attributes (x,y,z,w) or indexable sequences [x,y,z,w].
    """
    try:
        x, y, z, w = q.x, q.y, q.z, q.w
    except Exception:
        x, y, z, w = q[0], q[1], q[2], q[3]

    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def transform_to_state(transform):
    """Convert a geometry_msgs Transform / Pose-like object to a TurtleBotState-like tuple.

    Returns an object with attributes x,y,theta when possible.
    """
    class _S:
        pass

    s = _S()
    try:
        s.x = transform.translation.x
        s.y = transform.translation.y
        s.theta = quaternion_to_yaw(transform.rotation)
    except Exception:
        # fallback for pose with position/orientation
        s.x = transform.position.x
        s.y = transform.position.y
        s.theta = quaternion_to_yaw(transform.orientation)
    return s
