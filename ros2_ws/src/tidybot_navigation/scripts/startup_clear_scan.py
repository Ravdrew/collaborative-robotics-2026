#!/usr/bin/env python3
"""Startup Clearing Scan Node.

Publishes a single synthetic 360-degree LaserScan on /scan at startup.
This seeds SLAM Toolbox's map and the Nav2 costmap with FREE space around
the robot's footprint, solving the blind-spot initialization problem caused
by the forward-facing D435 depth camera.

The node waits until SLAM Toolbox and the costmap are both subscribed to
/scan and TF has had time to settle, publishes exactly one scan, then exits.
"""

import math
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import LaserScan

# Clearing radius in metres — just outside the robot footprint (0.5m x 0.44m).
# All cells between the robot centre and this radius will be marked FREE.
# The thin ring at this radius will be marked as a temporary obstacle, which
# the real depth scan overwrites within the first scan cycle.
CLEAR_RADIUS = 0.6

# One ray per degree gives a smooth disc with no large gaps at the resolution
# used by the costmap (0.05m).
NUM_RAYS = 360

# Wait for at least this many subscribers before publishing.
# Covers: SLAM Toolbox (1) + global costmap obstacle layer (1).
# The local costmap obstacle layer is a bonus if it subscribes in time.
MIN_SUBSCRIBERS = 2

# Maximum seconds to wait for subscribers before publishing anyway.
SUBSCRIBER_TIMEOUT = 30.0

# Additional settle time after subscribers appear, to allow the TF tree
# (odom → base_link) to become valid before SLAM processes the scan.
TF_SETTLE_DELAY = 0.5


class StartupClearScan(Node):
    def __init__(self):
        super().__init__('startup_clear_scan')

        # Use sensor_data QoS so the message is accepted by SLAM Toolbox and
        # the Nav2 costmap obstacle layers, which both subscribe with BEST_EFFORT.
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self._pub = self.create_publisher(LaserScan, '/scan', sensor_qos)

    def wait_and_publish(self):
        """Block until subscribers are ready, then publish one clearing scan."""
        self.get_logger().info(
            f'Startup clear scan: waiting for {MIN_SUBSCRIBERS} /scan '
            f'subscribers (SLAM Toolbox + costmap)...'
        )

        deadline = time.time() + SUBSCRIBER_TIMEOUT
        while self.count_subscribers('/scan') < MIN_SUBSCRIBERS:
            if time.time() > deadline:
                self.get_logger().warn(
                    f'Timed out waiting for {MIN_SUBSCRIBERS} /scan subscribers '
                    f'(have {self.count_subscribers("/scan")}). Publishing anyway.'
                )
                break
            time.sleep(0.1)

        self.get_logger().info(
            f'Found {self.count_subscribers("/scan")} /scan subscribers. '
            f'Waiting {TF_SETTLE_DELAY}s for TF to settle...'
        )
        time.sleep(TF_SETTLE_DELAY)

        self._pub.publish(self._build_scan())
        self.get_logger().info(
            f'Published startup clearing scan: {NUM_RAYS} rays at '
            f'{CLEAR_RADIUS}m in frame base_link. Node exiting.'
        )

    def _build_scan(self):
        scan = LaserScan()
        scan.header.stamp = self.get_clock().now().to_msg()
        scan.header.frame_id = 'base_link'

        scan.angle_min = -math.pi
        scan.angle_max = math.pi - (2.0 * math.pi / NUM_RAYS)  # avoid duplicate at ±π
        scan.angle_increment = 2.0 * math.pi / NUM_RAYS
        scan.time_increment = 0.0
        scan.scan_time = 0.1
        scan.range_min = 0.0
        scan.range_max = 10.0

        # All rays report a hit at CLEAR_RADIUS.
        # The raytrace clears every cell from 0 → CLEAR_RADIUS (FREE),
        # covering the full robot footprint regardless of heading.
        scan.ranges = [float(CLEAR_RADIUS)] * NUM_RAYS
        scan.intensities = []

        return scan


def main():
    rclpy.init()
    node = StartupClearScan()
    node.wait_and_publish()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
