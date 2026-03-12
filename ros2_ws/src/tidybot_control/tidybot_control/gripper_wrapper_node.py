#!/usr/bin/env python3
"""
Gripper Wrapper Node for TidyBot2.

Provides simulation-compatible gripper interface for real hardware.
Translates from:
    /right_gripper/cmd (Float64MultiArray, 0-1 normalized)
    /left_gripper/cmd (Float64MultiArray, 0-1 normalized)
To Interbotix SDK:
    /right_arm/commands/joint_single (JointSingleCommand)
    /left_arm/commands/joint_single (JointSingleCommand)

Uses PWM mode for both open and close to avoid operating mode switches
that can fail when the motor stalls against an object:
    - Open  (0.0): positive PWM pushes fingers apart
    - Close (1.0): negative PWM pushes fingers together

This allows the same user code to work for both simulation and real hardware.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from interbotix_xs_msgs.msg import JointSingleCommand
from interbotix_xs_msgs.srv import OperatingModes


class GripperWrapperNode(Node):
    """Wrapper node to translate sim gripper commands to Interbotix SDK."""

    GRIPPER_PRESSURE_LOWER = 150   # Minimum PWM for movement
    GRIPPER_PRESSURE_UPPER = 350   # Maximum PWM (avoid motor overload)

    def __init__(self):
        super().__init__('gripper_wrapper')

        self.declare_parameter('pressure', 1.0)
        self.pressure = self.get_parameter('pressure').value

        self.pwm_value = self.GRIPPER_PRESSURE_LOWER + self.pressure * (
            self.GRIPPER_PRESSURE_UPPER - self.GRIPPER_PRESSURE_LOWER
        )

        # Publishers to Interbotix SDK
        self.right_gripper_pub = self.create_publisher(
            JointSingleCommand, '/right_arm/commands/joint_single', 10
        )
        self.left_gripper_pub = self.create_publisher(
            JointSingleCommand, '/left_arm/commands/joint_single', 10
        )

        # Service clients for one-time PWM mode setup at startup
        self.right_mode_client = self.create_client(
            OperatingModes, '/right_arm/set_operating_modes'
        )
        self.left_mode_client = self.create_client(
            OperatingModes, '/left_arm/set_operating_modes'
        )

        self.right_pwm_ready = False
        self.left_pwm_ready = False
        self.right_mode_future = None
        self.left_mode_future = None

        # Subscribers - same topics as MuJoCo simulation
        self.right_gripper_sub = self.create_subscription(
            Float64MultiArray, '/right_gripper/cmd',
            lambda msg: self._gripper_callback(msg, 'right'), 10
        )
        self.left_gripper_sub = self.create_subscription(
            Float64MultiArray, '/left_gripper/cmd',
            lambda msg: self._gripper_callback(msg, 'left'), 10
        )

        self._setup_timer = self.create_timer(1.0, self._startup_set_pwm_mode)
        self.create_timer(0.05, self._check_mode_futures)

        self.get_logger().info('Gripper wrapper node started (PWM-only mode)')
        self.get_logger().info(f'  Pressure: {self.pressure * 100:.0f}% (PWM: {self.pwm_value:.0f})')
        self.get_logger().info('  Listening on /right_gripper/cmd and /left_gripper/cmd')
        self.get_logger().info('  Command: 0.0=open (+PWM), 1.0=close (-PWM)')

    # ------------------------------------------------------------------
    # One-time PWM mode setup
    # ------------------------------------------------------------------

    def _startup_set_pwm_mode(self):
        """Retry setting both grippers to PWM mode until successful."""
        if not self.right_pwm_ready and self.right_mode_future is None:
            self._request_pwm_mode('right')
        if not self.left_pwm_ready and self.left_mode_future is None:
            self._request_pwm_mode('left')
        if self.right_pwm_ready and self.left_pwm_ready:
            self.get_logger().info('Both grippers in PWM mode — ready.')
            self._setup_timer.cancel()

    def _request_pwm_mode(self, side: str):
        """Send a one-time request to set gripper to PWM mode."""
        client = self.right_mode_client if side == 'right' else self.left_mode_client

        if not client.wait_for_service(timeout_sec=0.1):
            self.get_logger().debug(
                f'{side} gripper: set_operating_modes service not yet available')
            return

        req = OperatingModes.Request()
        req.cmd_type = 'single'
        req.name = f'{side}_gripper'
        req.mode = 'pwm'
        req.profile_type = 'velocity'
        req.profile_velocity = 131
        req.profile_acceleration = 25

        future = client.call_async(req)
        if side == 'right':
            self.right_mode_future = future
        else:
            self.left_mode_future = future
        self.get_logger().info(f'{side} gripper: requesting PWM mode ...')

    def _check_mode_futures(self):
        """Process pending mode-switch futures."""
        for side in ('right', 'left'):
            future = self.right_mode_future if side == 'right' else self.left_mode_future
            if future is None or not future.done():
                continue

            if future.cancelled() or future.exception() is not None:
                self.get_logger().warn(
                    f'{side} gripper: PWM mode switch failed — will retry')
            else:
                if side == 'right':
                    self.right_pwm_ready = True
                else:
                    self.left_pwm_ready = True
                self.get_logger().info(f'{side} gripper: PWM mode active')

            if side == 'right':
                self.right_mode_future = None
            else:
                self.left_mode_future = None

    # ------------------------------------------------------------------
    # Gripper command handling
    # ------------------------------------------------------------------

    def _gripper_callback(self, msg: Float64MultiArray, side: str):
        """
        Handle gripper command from simulation-compatible topic.

        Maps normalized input [0, 1] to PWM:
            0.0 (open)  -> +pwm_value
            1.0 (close) -> -pwm_value
        """
        if len(msg.data) < 1:
            return

        normalized = max(0.0, min(1.0, msg.data[0]))
        pwm = self.pwm_value - normalized * (2 * self.pwm_value)

        pwm_ready = self.right_pwm_ready if side == 'right' else self.left_pwm_ready
        if not pwm_ready:
            self.get_logger().warn(
                f'{side} gripper: PWM mode not ready yet, dropping command')
            return

        self._publish_cmd(side, pwm)

    def _publish_cmd(self, side: str, value: float):
        """Publish a JointSingleCommand to the requested arm gripper."""
        cmd = JointSingleCommand()
        cmd.cmd = float(value)
        if side == 'right':
            cmd.name = 'right_gripper'
            self.right_gripper_pub.publish(cmd)
        else:
            cmd.name = 'left_gripper'
            self.left_gripper_pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = GripperWrapperNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
