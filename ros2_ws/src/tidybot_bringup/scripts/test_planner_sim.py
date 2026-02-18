#!/usr/bin/env python3
"""
TidyBot2 Motion Planner Demo

Demonstrates the motion planning service with collision and singularity checking.
Plans arm motions to various target poses and executes them.

Usage:
    # Terminal 1: Start simulation
    ros2 launch tidybot_bringup sim.launch.py

    # Terminal 2: Run this demo
    ros2 run tidybot_bringup test_planner_sim.py
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
from tidybot_msgs.srv import PlanToTarget
import time
import numpy as np


class TestPlanner(Node):
    """Demo node for testing motion planning service."""

    # Common orientations in base_link frame (quaternion wxyz)
    #
    # The pinch_site x-axis is the gripper finger direction.
    # For a top-down grasp the fingers must point straight down (-Z).
    #
    # Fingers-down grasp (Ry(π/2) in base_link):
    #   site x -> base_link -Z (fingers point down)
    #   site z -> base_link +X (approach from front)
    ORIENT_FINGERS_DOWN = (0.5, 0.5, 0.5, -0.5)  # (qw, qx, qy, qz)

    # Fingers-down with wrist rotated 90° around the approach axis:
    ORIENT_FINGERS_DOWN_ROT90 = (0.707107, 0.0, 0.707107, 0.0)

    def __init__(self):
        super().__init__('test_planner')

        # Create client for planning service
        self.plan_client = self.create_client(PlanToTarget, '/plan_to_target')

        self.get_logger().info('=' * 50)
        self.get_logger().info('TidyBot2 Motion Planner Demo')
        self.get_logger().info('=' * 50)

        # Wait for service
        self.get_logger().info('Waiting for /plan_to_target service...')
        while not self.plan_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting...')

        self.get_logger().info('Service connected!')
        self.get_logger().info('')

    def create_pose(self, x: float, y: float, z: float,
                    qw: float = 1.0, qx: float = 0.0,
                    qy: float = 0.0, qz: float = 0.0) -> Pose:
        """Create a Pose message."""
        pose = Pose()
        pose.position.x = x
        pose.position.y = y
        pose.position.z = z
        pose.orientation.w = qw
        pose.orientation.x = qx
        pose.orientation.y = qy
        pose.orientation.z = qz
        return pose

    def call_service_sync(self, request, timeout_sec=15.0):
        """Call service synchronously using spin_until_future_complete."""
        future = self.plan_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout_sec)

        if not future.done():
            self.get_logger().error('Service call timed out!')
            return None

        if future.exception() is not None:
            self.get_logger().error(f'Service call exception: {future.exception()}')
            return None

        return future.result()

    def plan_and_execute(self, arm_name: str, pose: Pose,
                        use_orientation: bool = True,
                        duration: float = 2.0) -> bool:
        """Send planning request and wait for result."""
        request = PlanToTarget.Request()
        request.arm_name = arm_name
        request.target_pose = pose
        request.use_orientation = use_orientation
        request.execute = True
        request.duration = duration
        request.max_condition_number = 100.0

        pos_str = f'({pose.position.x:.3f}, {pose.position.y:.3f}, {pose.position.z:.3f})'
        if use_orientation:
            ori_str = (f' orient=({pose.orientation.w:.3f}, {pose.orientation.x:.3f}, '
                       f'{pose.orientation.y:.3f}, {pose.orientation.z:.3f})')
        else:
            ori_str = ' (position only)'
        self.get_logger().info(f'Planning {arm_name} arm to: {pos_str}{ori_str}')

        result = self.call_service_sync(request)
        if result is None:
            return False

        if result.success:
            self.get_logger().info(f'  SUCCESS: {result.message}')
            if use_orientation:
                self.get_logger().info(
                    f'  Errors: pos={result.position_error:.4f}m, '
                    f'ori={result.orientation_error:.4f}rad ({np.degrees(result.orientation_error):.1f}°)')
            if result.executed:
                self.get_logger().info(f'  Executing over {duration}s...')
            return True
        else:
            self.get_logger().warn(f'  FAILED: {result.message}')
            return False

    def run_demos(self):
        """Run the demo sequence.

        Coordinate frame (base_link):
          -Y is forward, +X is left, +Z is up.
          Arm shoulders: right=(-0.15, -0.12, 0.45), left=(0.15, -0.12, 0.45)
          Arms extend forward (-Y) from the shoulder mounts.
          Reachable workspace: ~0.3-0.5m from shoulder.

        Orientation conventions (quaternion wxyz in base_link frame):
          Fingers-down Ry(π/2):       (0.707, 0, 0.707, 0)  — gripper fingers point -Z
          Fingers-down rotated 90°:   (0.5, 0.5, 0.5, -0.5)
        """
        qw_fd, qx_fd, qy_fd, qz_fd = self.ORIENT_FINGERS_DOWN
        qw_fr, qx_fr, qy_fr, qz_fr = self.ORIENT_FINGERS_DOWN_ROT90

        # ── Part A: Position-only demos ──────────────────────────────

        self.get_logger().info('=' * 50)
        self.get_logger().info('Part A: Position-only IK demos')
        self.get_logger().info('=' * 50)

        # Demo 1: Right arm forward reach (position only)
        self.get_logger().info('-' * 40)
        self.get_logger().info('Demo 1: Right arm forward reach (position only)')
        pose1 = self.create_pose(-0.10, -0.35, 0.55)
        self.plan_and_execute('right', pose1, use_orientation=False, duration=2.0)
        input('\n  Press Enter for next demo...\n')

        # Demo 2: Right arm side reach (position only)
        self.get_logger().info('-' * 40)
        self.get_logger().info('Demo 2: Right arm side reach (position only)')
        pose2 = self.create_pose(-0.25, -0.35, 0.50)
        self.plan_and_execute('right', pose2, use_orientation=False, duration=2.0)
        input('\n  Press Enter for next demo...\n')

        # Demo 3: Left arm forward (position only)
        self.get_logger().info('-' * 40)
        self.get_logger().info('Demo 3: Left arm forward (position only)')
        pose3 = self.create_pose(0.10, -0.35, 0.55)
        self.plan_and_execute('left', pose3, use_orientation=False, duration=2.0)
        input('\n  Press Enter for next demo...\n')

        # ── Part B: Orientation-tracking demos ───────────────────────

        self.get_logger().info('')
        self.get_logger().info('=' * 50)
        self.get_logger().info('Part B: Position + Orientation IK demos')
        self.get_logger().info('=' * 50)

        # Demo 4: Right arm fingers-down grasp
        #   Gripper fingers point straight down — good for picking objects from a table
        self.get_logger().info('-' * 40)
        self.get_logger().info('Demo 4: Right arm fingers-down grasp')
        pose4 = self.create_pose(-0.10, -0.35, 0.55, qw_fd, qx_fd, qy_fd, qz_fd)
        self.plan_and_execute('right', pose4, use_orientation=True, duration=2.0)
        input('\n  Press Enter for next demo...\n')

        # Demo 5: Right arm fingers-down at different position
        self.get_logger().info('-' * 40)
        self.get_logger().info('Demo 5: Right arm fingers-down lower position')
        pose5 = self.create_pose(-0.05, -0.35, 0.50, qw_fd, qx_fd, qy_fd, qz_fd)
        self.plan_and_execute('right', pose5, use_orientation=True, duration=2.0)
        input('\n  Press Enter for next demo...\n')

        # Demo 6: Right arm fingers-down-rot90 (side reach)
        #   Wrist rotated 90° — different grasp angle for side objects
        self.get_logger().info('-' * 40)
        self.get_logger().info('Demo 6: Right arm fingers-down rotated (side reach)')
        pose6 = self.create_pose(-0.20, -0.30, 0.50, qw_fr, qx_fr, qy_fr, qz_fr)
        self.plan_and_execute('right', pose6, use_orientation=True, duration=2.0)
        input('\n  Press Enter for next demo...\n')

        # Demo 7: Left arm fingers-down
        self.get_logger().info('-' * 40)
        self.get_logger().info('Demo 7: Left arm fingers-down')
        pose7 = self.create_pose(0.10, -0.35, 0.55, qw_fd, qx_fd, qy_fd, qz_fd)
        self.plan_and_execute('left', pose7, use_orientation=True, duration=2.0)
        time.sleep(3.0)

        self.get_logger().info('')
        self.get_logger().info('=' * 50)
        self.get_logger().info('Demo complete!')
        self.get_logger().info('=' * 50)


def main(args=None):
    rclpy.init(args=args)
    node = TestPlanner()

    try:
        # Run tests directly instead of using timer callback
        # This avoids nested executor issues with spin_until_future_complete
        node.run_demos()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
