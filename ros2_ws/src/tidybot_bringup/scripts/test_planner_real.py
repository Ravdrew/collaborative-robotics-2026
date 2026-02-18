#!/usr/bin/env python3
"""
TidyBot2 Motion Planner Test for Real Hardware

Tests the IK motion planning service on real hardware.
Uses conservative poses suitable for real robot testing.

Usage:
    # Terminal 1: Start real hardware with motion planner
    ros2 launch tidybot_bringup real.launch.py use_planner:=true

    # Terminal 2: Run this test
    ros2 run tidybot_bringup test_planner_real.py
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
from tidybot_msgs.srv import PlanToTarget
from sensor_msgs.msg import JointState
from interbotix_xs_msgs.msg import JointGroupCommand
import numpy as np
import time


class TestPlannerReal(Node):
    """Test node for motion planning on real hardware."""

    # Sleep pose for arms (same as test_arms_real.py)
    SLEEP_POSE = [0.0, -1.80, 1.55, 0.0, 0.8, 0.0]

    # Common orientations in base_link frame (quaternion wxyz)
    #
    # The ee x-axis is the gripper finger direction.
    # For a top-down grasp the fingers must point straight down (-Z).
    #
    # Fingers-down grasp:
    #   ee x -> base_link -Z  (fingers point down)
    ORIENT_FINGERS_DOWN = (0.5, 0.5, 0.5, -0.5)  # (qw, qx, qy, qz)

    # Fingers-down with wrist rotated 90° around the approach axis:
    ORIENT_FINGERS_DOWN_ROT90 = (0.707107, 0.0, 0.707107, 0.0)

    def __init__(self):
        super().__init__('test_planner_real')

        # Create client for planning service
        self.plan_client = self.create_client(PlanToTarget, '/plan_to_target')

        # Publishers for direct arm commands (for returning to sleep pose)
        self.arm_cmd_pubs = {
            'right': self.create_publisher(JointGroupCommand, '/right_arm/commands/joint_group', 10),
            'left': self.create_publisher(JointGroupCommand, '/left_arm/commands/joint_group', 10),
        }

        # Subscribe to joint states to verify we're connected and track positions
        self.joint_states_received = False
        self.current_joint_positions = {}
        self.create_subscription(JointState, '/joint_states', self._js_callback, 10)

        self.get_logger().info('=' * 50)
        self.get_logger().info('TidyBot2 IK Planner Test (Real Hardware)')
        self.get_logger().info('=' * 50)

        # Wait for service
        self.get_logger().info('Waiting for /plan_to_target service...')
        if not self.plan_client.wait_for_service(timeout_sec=10.0):
            self.get_logger().error('Service not available! Make sure motion_planner_real_node is running.')
            self.get_logger().error('Launch with: ros2 launch tidybot_bringup real.launch.py use_planner:=true')
            raise RuntimeError('Planning service not available')

        self.get_logger().info('Service connected!')

        # Wait for joint states
        self.get_logger().info('Waiting for joint states...')
        for _ in range(50):
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.joint_states_received:
                break

        if not self.joint_states_received:
            self.get_logger().warn('No joint states received - proceeding anyway')

        self.get_logger().info('')

    def _js_callback(self, msg):
        self.joint_states_received = True
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                self.current_joint_positions[name] = msg.position[i]

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

    def plan_only(self, arm_name: str, pose: Pose,
                  use_orientation: bool = True) -> bool:
        """Plan without executing - for safety check."""
        request = PlanToTarget.Request()
        request.arm_name = arm_name
        request.target_pose = pose
        request.use_orientation = use_orientation
        request.execute = False  # Plan only
        request.duration = 2.0
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
            self.get_logger().info(f'  Solution: {[f"{j:.3f}" for j in result.joint_positions]}')
            if use_orientation:
                self.get_logger().info(
                    f'  Errors: pos={result.position_error:.4f}m, '
                    f'ori={result.orientation_error:.4f}rad ({np.degrees(result.orientation_error):.1f}°)')
            return True
        else:
            self.get_logger().warn(f'  FAILED: {result.message}')
            return False

    def plan_and_execute(self, arm_name: str, pose: Pose,
                         use_orientation: bool = True,
                         duration: float = 3.0) -> bool:
        """Send planning request and execute."""
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
        self.get_logger().info(f'Planning and executing {arm_name} arm to: {pos_str}{ori_str}')

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

    def get_arm_positions(self, arm_name: str) -> np.ndarray:
        """Get current joint positions for an arm."""
        joint_names = [f'{arm_name}_waist', f'{arm_name}_shoulder', f'{arm_name}_elbow',
                       f'{arm_name}_forearm_roll', f'{arm_name}_wrist_angle', f'{arm_name}_wrist_rotate']
        positions = np.array([self.current_joint_positions.get(jname, 0.0) for jname in joint_names])
        return positions

    def go_to_sleep_pose(self, arm_name: str, max_joint_speed: float = 0.5):
        """Send arm to sleep pose using smooth interpolated trajectory.

        Args:
            arm_name: 'right' or 'left'
            max_joint_speed: Maximum joint velocity in rad/s (default 0.5 rad/s ~ 30 deg/s)
        """
        # Spin to get latest joint states
        rclpy.spin_once(self, timeout_sec=0.1)

        # Get current and target positions
        current = self.get_arm_positions(arm_name)
        target = np.array(self.SLEEP_POSE)

        # Calculate required duration based on max joint difference
        max_diff = np.max(np.abs(target - current))
        duration = max(max_diff / max_joint_speed, 1.0)  # At least 1 second

        self.get_logger().info(f'Moving {arm_name} arm to sleep pose over {duration:.1f}s (max joint diff: {max_diff:.2f} rad)')

        # Interpolate trajectory with smooth cosine profile
        rate_hz = 50.0
        dt = 1.0 / rate_hz
        num_steps = max(int(duration * rate_hz), 1)

        for i in range(num_steps + 1):
            t = i / num_steps
            # Cosine interpolation for smooth acceleration/deceleration
            alpha = 0.5 * (1 - np.cos(np.pi * t))

            # Interpolate
            q = current + alpha * (target - current)

            cmd = JointGroupCommand()
            cmd.name = f'{arm_name}_arm'
            cmd.cmd = q.tolist()
            self.arm_cmd_pubs[arm_name].publish(cmd)

            if i < num_steps:
                time.sleep(dt)

    def run_tests(self):
        """Run the test sequence.

        Coordinate frame (base_link):
          -Y is forward, +X is left, +Z is up.
          Arm shoulders: right=(-0.15, -0.12, 0.45), left=(0.15, -0.12, 0.45)
          Arms extend forward (-Y) from the shoulder mounts.
          Reachable workspace: ~0.3-0.5m from shoulder.
        """
        qw_fd, qx_fd, qy_fd, qz_fd = self.ORIENT_FINGERS_DOWN
        qw_fr, qx_fr, qy_fr, qz_fr = self.ORIENT_FINGERS_DOWN_ROT90

        # ── Part A: Position-only planning (no execution) ────────────

        self.get_logger().info('=' * 50)
        self.get_logger().info('Part A: Position-only IK — PLAN ONLY (no execution)')
        self.get_logger().info('=' * 50)

        # Test 1: Plan only right arm
        self.get_logger().info('-' * 40)
        self.get_logger().info('Test 1: Right arm position-only — PLAN ONLY')
        pose1 = self.create_pose(0.05, -0.35, 0.55)
        success1 = self.plan_only('right', pose1, use_orientation=False)
        time.sleep(1.0)

        # Test 2: Plan only left arm
        self.get_logger().info('-' * 40)
        self.get_logger().info('Test 2: Left arm position-only — PLAN ONLY')
        pose2 = self.create_pose(-0.05, -0.35, 0.55)
        success2 = self.plan_only('left', pose2, use_orientation=False)
        time.sleep(1.0)

        # ── Part B: Orientation IK planning (no execution) ───────────

        self.get_logger().info('')
        self.get_logger().info('=' * 50)
        self.get_logger().info('Part B: Orientation IK — PLAN ONLY (no execution)')
        self.get_logger().info('=' * 50)

        # Test 3: Plan right arm fingers-down
        self.get_logger().info('-' * 40)
        self.get_logger().info('Test 3: Right arm fingers-down — PLAN ONLY')
        pose3 = self.create_pose(-0.10, -0.35, 0.55, qw_fd, qx_fd, qy_fd, qz_fd)
        success3 = self.plan_only('right', pose3, use_orientation=True)
        time.sleep(1.0)

        # # Test 4: Plan right arm fingers-down-rot90
        self.get_logger().info('-' * 40)
        self.get_logger().info('Test 4: Right arm fingers-down-rot90 — PLAN ONLY')
        pose4 = self.create_pose(-0.20, -0.30, 0.50, qw_fr, qx_fr, qy_fr, qz_fr)
        success4 = self.plan_only('right', pose4, use_orientation=True)
        time.sleep(1.0)

        # Test 5: Plan left arm fingers-down
        self.get_logger().info('-' * 40)
        self.get_logger().info('Test 5: Left arm fingers-down — PLAN ONLY')
        pose5 = self.create_pose(0.10, -0.35, 0.55, qw_fd, qx_fd, qy_fd, qz_fd)
        success5 = self.plan_only('left', pose5, use_orientation=True)
        time.sleep(1.0)

        plan_ok = success1 and success2 and success3 and success4 and success5
        # plan_ok = success1 and success2 and success3 and success5
        if not plan_ok:
            self.get_logger().error('Some planning tests failed — skipping execution tests')
        else:
            self.get_logger().info('')
            self.get_logger().info('All planning tests passed!')

            # ── Part C: Position-only execution ──────────────────────

            self.get_logger().info('')
            self.get_logger().info('=' * 50)
            self.get_logger().info('Part C: Position-only IK — EXECUTE')
            self.get_logger().info('The robot will move! Make sure the workspace is clear.')
            self.get_logger().info('=' * 50)
            input('\n  Press Enter to start execution (Ctrl+C to abort)...\n')

            # Test 6: Execute right arm position-only
            self.get_logger().info('-' * 40)
            self.get_logger().info('Test 6: Right arm position-only — EXECUTE')
            pose6 = self.create_pose(-0.10, -0.35, 0.55)
            self.plan_and_execute('right', pose6, use_orientation=False, duration=3.0)
            input('\n  Press Enter for next test...\n')

            # Test 7: Execute left arm position-only
            self.get_logger().info('-' * 40)
            self.get_logger().info('Test 7: Left arm position-only — EXECUTE')
            pose7 = self.create_pose(0.10, -0.35, 0.55)
            self.plan_and_execute('left', pose7, use_orientation=False, duration=3.0)
            input('\n  Press Enter for next test...\n')

            # ── Part D: Orientation execution ────────────────────────

            self.get_logger().info('')
            self.get_logger().info('=' * 50)
            self.get_logger().info('Part D: Orientation IK — EXECUTE')
            self.get_logger().info('Gripper fingers should point straight down.')
            self.get_logger().info('=' * 50)
            input('\n  Press Enter to start orientation execution (Ctrl+C to abort)...\n')

            # Test 8: Right arm fingers-down
            self.get_logger().info('-' * 40)
            self.get_logger().info('Test 8: Right arm fingers-down — EXECUTE')
            pose8 = self.create_pose(-0.10, -0.35, 0.55, qw_fd, qx_fd, qy_fd, qz_fd)
            self.plan_and_execute('right', pose8, use_orientation=True, duration=3.0)
            input('\n  Press Enter for next test...\n')

            # Test 9: Right arm fingers-down lower position
            self.get_logger().info('-' * 40)
            self.get_logger().info('Test 9: Right arm fingers-down lower — EXECUTE')
            pose9 = self.create_pose(-0.05, -0.35, 0.50, qw_fd, qx_fd, qy_fd, qz_fd)
            self.plan_and_execute('right', pose9, use_orientation=True, duration=3.0)
            input('\n  Press Enter for next test...\n')

            # Test 10: Right arm fingers-down-rot90 (side reach)
            self.get_logger().info('-' * 40)
            self.get_logger().info('Test 10: Right arm fingers-down-rot90 — EXECUTE')
            pose10 = self.create_pose(-0.20, -0.30, 0.50, qw_fr, qx_fr, qy_fr, qz_fr)
            self.plan_and_execute('right', pose10, use_orientation=True, duration=3.0)
            input('\n  Press Enter for next test...\n')

            # Test 11: Left arm fingers-down
            self.get_logger().info('-' * 40)
            self.get_logger().info('Test 11: Left arm fingers-down — EXECUTE')
            pose11 = self.create_pose(0.10, -0.35, 0.55, qw_fd, qx_fd, qy_fd, qz_fd)
            self.plan_and_execute('left', pose11, use_orientation=True, duration=3.0)
            time.sleep(3.0)

        # ── Return to sleep ──────────────────────────────────────────

        self.get_logger().info('')
        self.get_logger().info('=' * 50)
        self.get_logger().info('Returning arms to sleep pose...')
        self.get_logger().info('=' * 50)
        self.go_to_sleep_pose('right')
        self.go_to_sleep_pose('left')

        self.get_logger().info('')
        self.get_logger().info('=' * 50)
        self.get_logger().info('Test complete!')
        self.get_logger().info('=' * 50)


def main(args=None):
    rclpy.init(args=args)
    node = TestPlannerReal()

    try:
        # Run tests directly instead of using timer callback
        # This avoids nested executor issues with spin_until_future_complete
        node.run_tests()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
