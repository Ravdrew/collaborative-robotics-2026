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
And switches modes via:
    /right_arm/set_operating_modes (OperatingModes)
    /left_arm/set_operating_modes (OperatingModes)

Hybrid behavior:
    - Open command: position mode, move to a configured open position
    - Close command: pwm mode, apply configured grasp pressure

This allows the same user code to work for both simulation and real hardware.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from interbotix_xs_msgs.msg import JointSingleCommand
from interbotix_xs_msgs.srv import OperatingModes


class GripperWrapperNode(Node):
    """Wrapper node to translate sim gripper commands to Interbotix SDK."""

    # PWM pressure limits (from Interbotix SDK)
    GRIPPER_PRESSURE_LOWER = 150   # Minimum PWM for movement
    GRIPPER_PRESSURE_UPPER = 350   # Maximum PWM (avoid motor overload)
    LEFT_OPEN_POSITION_RAD = 1.478758
    RIGHT_OPEN_POSITION_RAD = 0.733243
    OPEN_CLOSE_THRESHOLD = 0.5

    def __init__(self):
        super().__init__('gripper_wrapper')

        # Declare pressure parameter (0.0 to 1.0)
        self.declare_parameter('pressure', 1.0)
        self.pressure = self.get_parameter('pressure').value

        # Calculate PWM from pressure
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

        # Clients to switch operating mode between position (open) and PWM (close)
        self.right_mode_client = self.create_client(
            OperatingModes, '/right_arm/set_operating_modes'
        )
        self.left_mode_client = self.create_client(
            OperatingModes, '/left_arm/set_operating_modes'
        )

        # Track current mode to avoid unnecessary service calls
        self.right_mode = 'pwm'
        self.left_mode = 'pwm'
        self.right_mode_pending = None
        self.left_mode_pending = None
        self.right_mode_future = None
        self.left_mode_future = None
        self.right_deferred_cmd = None
        self.left_deferred_cmd = None

        # Subscribers - same topics as MuJoCo simulation
        self.right_gripper_sub = self.create_subscription(
            Float64MultiArray, '/right_gripper/cmd',
            lambda msg: self._gripper_callback(msg, 'right'), 10
        )
        self.left_gripper_sub = self.create_subscription(
            Float64MultiArray, '/left_gripper/cmd',
            lambda msg: self._gripper_callback(msg, 'left'), 10
        )
        self.create_timer(0.05, self._deferred_command_timer)

        self.get_logger().info('Gripper wrapper node started')
        self.get_logger().info(f'  Pressure: {self.pressure * 100:.0f}% (PWM: {self.pwm_value:.0f})')
        self.get_logger().info(
            f'  Open positions: left={self.LEFT_OPEN_POSITION_RAD:.3f} rad, '
            f'right={self.RIGHT_OPEN_POSITION_RAD:.3f} rad'
        )
        self.get_logger().info('  Listening on /right_gripper/cmd and /left_gripper/cmd')
        self.get_logger().info('  Publishing to Interbotix SDK joint_single topics')
        self.get_logger().info('  Command: 0.0=open (position mode), 1.0=close (PWM mode)')

    def _gripper_callback(self, msg: Float64MultiArray, side: str):
        """
        Handle gripper command from simulation-compatible topic.

        Args:
            msg: Float64MultiArray with data[0] = normalized position (0=open, 1=closed)
            side: 'right' or 'left'
        """
        if len(msg.data) < 1:
            return

        # Normalize input to 0-1 range
        normalized = max(0.0, min(1.0, msg.data[0]))
        # New command supersedes any previously deferred command.
        self._clear_deferred_cmd(side)

        # Open command: move to explicit position.
        if normalized <= self.OPEN_CLOSE_THRESHOLD:
            open_pos = self.RIGHT_OPEN_POSITION_RAD if side == 'right' else self.LEFT_OPEN_POSITION_RAD
            if self._ensure_mode(side, 'position'):
                self._publish_cmd(side, open_pos)
            else:
                # One-shot compatibility: queue command and execute after mode switch.
                self._set_deferred_cmd(side, 'position', open_pos)

        # Close command: PWM mapping behavior to deal with variability in object sizes
        else:
            pwm = self.pwm_value - normalized * (2 * self.pwm_value)
            if self._ensure_mode(side, 'pwm'):
                self._publish_cmd(side, pwm)
            else:
                # Queue close command for one-shot compatibility.
                self._set_deferred_cmd(side, 'pwm', pwm)

    def _deferred_command_timer(self):
        """Process mode-switch completions and execute deferred one-shot commands."""
        for side in ('right', 'left'):
            self._process_mode_future(side)
            self._flush_deferred_cmd(side)

    def _set_deferred_cmd(self, side: str, mode: str, cmd: float):
        if side == 'right':
            self.right_deferred_cmd = (mode, float(cmd))
        else:
            self.left_deferred_cmd = (mode, float(cmd))

    def _clear_deferred_cmd(self, side: str):
        if side == 'right':
            self.right_deferred_cmd = None
        else:
            self.left_deferred_cmd = None

    def _flush_deferred_cmd(self, side: str):
        deferred = self.right_deferred_cmd if side == 'right' else self.left_deferred_cmd
        if deferred is None:
            return
        desired_mode, cmd = deferred
        current_mode = self.right_mode if side == 'right' else self.left_mode
        if current_mode != desired_mode:
            return
        self._publish_cmd(side, cmd)
        if side == 'right':
            self.right_deferred_cmd = None
        else:
            self.left_deferred_cmd = None

    def _process_mode_future(self, side: str):
        pending_mode = self.right_mode_pending if side == 'right' else self.left_mode_pending
        pending_future = self.right_mode_future if side == 'right' else self.left_mode_future
        if pending_future is None or not pending_future.done():
            return

        if pending_future.cancelled() or pending_future.exception() is not None:
            self.get_logger().warn(f'{side} gripper: mode switch to {pending_mode} failed')
        else:
            if side == 'right':
                self.right_mode = pending_mode
            else:
                self.left_mode = pending_mode

        if side == 'right':
            self.right_mode_pending = None
            self.right_mode_future = None
        else:
            self.left_mode_pending = None
            self.left_mode_future = None

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

    def _ensure_mode(self, side: str, desired_mode: str) -> bool:
        """
        Ensure gripper operating mode is set for this side.

        Returns True if mode is ready, False if service is unavailable/failed.
        """
        client = self.right_mode_client if side == 'right' else self.left_mode_client
        gripper_name = 'right_gripper' if side == 'right' else 'left_gripper'
        current_mode = self.right_mode if side == 'right' else self.left_mode
        pending_future = self.right_mode_future if side == 'right' else self.left_mode_future

        if current_mode == desired_mode:
            return True

        # Consume any completed mode switch before making a new request.
        self._process_mode_future(side)
        current_mode = self.right_mode if side == 'right' else self.left_mode
        pending_future = self.right_mode_future if side == 'right' else self.left_mode_future
        if current_mode == desired_mode:
            return True

        # If a mode switch is already in-flight, don't send duplicates.
        if pending_future is not None:
            return False

        if not client.wait_for_service(timeout_sec=0.1):
            self.get_logger().warn(
                f'{side} gripper: set_operating_modes service unavailable; keeping {current_mode}'
            )
            return False

        req = OperatingModes.Request()
        req.cmd_type = 'single'
        req.name = gripper_name
        req.mode = desired_mode
        req.profile_type = 'velocity'
        req.profile_velocity = 131
        req.profile_acceleration = 25

        future = client.call_async(req)
        if side == 'right':
            self.right_mode_pending = desired_mode
            self.right_mode_future = future
        else:
            self.left_mode_pending = desired_mode
            self.left_mode_future = future
        return False


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
