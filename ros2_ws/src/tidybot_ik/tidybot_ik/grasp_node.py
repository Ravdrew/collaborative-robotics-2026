#!/usr/bin/env python3
"""
TidyBot2 Grasp Execution Node

Orchestrates the full grasp sequence as a state machine:
  1. Relay pick target through grasp_generation_node to get EEF pose
  2. Call /plan_to_target to move arm to EEF pose
  3. Wait until the EEF (ee_arm_link TF) is within eef_arrival_threshold of the target
  4. Close gripper
  5. Call /plan_to_target to retract arm to neutral pose
  6. Check finger position to verify grasp success
  7. Publish result on /grasp_completed

Requires grasp_generation_node to be running (ros2 run tidybot_ik grasp_generation_node).

Topics subscribed:
- /pick_target_local  (geometry_msgs/Pose)      Object pose in camera_color_optical_frame
- /EEF_pose_command   (geometry_msgs/Pose)       EEF pose from grasp_generation_node
- /joint_states       (sensor_msgs/JointState)   For gripper finger position check

Topics published:
- /object_pose_in_camera  (geometry_msgs/Pose)        Relay to grasp_generation_node
- /right_gripper/cmd      (std_msgs/Float64MultiArray) Right gripper open/close
- /left_gripper/cmd       (std_msgs/Float64MultiArray) Left gripper open/close
- /grasp_completed        (std_msgs/Bool)              True if object was grasped

Service clients:
- /plan_to_target (tidybot_msgs/srv/PlanToTarget)

Parameters:
- arm_name               (str,   default: 'right') Which arm to use
- gripper_toggle_time     (float, default: 1.5)     Seconds to keep gripper closing command
- grasp_finger_threshold (float, default: 0.033)   Finger pos (m) below which object detected
                                                    (open=0.037 m, closed=0.015 m)
- eef_arrival_threshold  (float, default: 0.03)    Distance (m) from target to declare arrival
- neutral_x/y/z          (float) Retract pose position in base_link
- neutral_qw/qx/qy/qz   (float) Retract pose orientation in base_link

Usage:
    # Terminal 1
    ros2 launch tidybot_bringup sim.launch.py scene:=scene_pickup.xml

    # Terminal 2 - trigger a grasp
    ros2 topic pub --once /pick_target_local geometry_msgs/msg/Pose \\
        "{position: {x: 0.0, y: 0.3, z: 0.5}, orientation: {w: 1.0}}"

    # Terminal 3 - watch result
    ros2 topic echo /grasp_completed
"""

import math
import time
from enum import Enum, auto

import rclpy
import rclpy.time
from rclpy.node import Node
from geometry_msgs.msg import Pose
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, Bool
import tf2_ros

from tidybot_msgs.srv import PlanToTarget


# ---------------------------------------------------------------------------
# Top-down grasp orientation (fingers-down) in base_link frame — wxyz
# Matches test_planner_sim.py
# ---------------------------------------------------------------------------
_FINGERS_DOWN_HORIZONTAL = (0.5, 0.5, 0.5, -0.5)  # (qw, qx, qy, qz)
_FINGERS_DOWN_VERTICAL = (0.707107, 0.0, 0.707107, 0.0)  # (qw, qx, qy, qz)

# Gripper positions (0.0 = open, 1.0 = closed — matches test_arms_sim.py)
_GRIPPER_OPEN   = 0.0
_GRIPPER_CLOSED = 1.0

# Finger joint open position (metres) — from MuJoCo model range
_FINGER_OPEN_POS = 0.037


class State(Enum):
    IDLE              = auto()  # waiting for /pick_target_local
    WAIT_EEF_POSE     = auto()  # waiting for /EEF_pose_command after relaying pick target
    PLAN_GRASP        = auto()  # issuing /plan_to_target call (single entry tick)
    WAIT_PLAN_GRASP   = auto()  # waiting for plan failure OR EEF arrival at target
    TOGGLE_GRIPPER     = auto()  # gripper closing, timing out
    PLAN_NEUTRAL      = auto()  # issuing /plan_to_target for retract (single entry tick)
    WAIT_PLAN_NEUTRAL = auto()  # waiting for retract future to resolve
    CHECK_GRASP       = auto()  # inspect finger position
    DONE              = auto()  # grasp succeeded
    FAILED            = auto()  # grasp failed


class GraspNode(Node):
    """State-machine node that executes a full grasp sequence."""

    def __init__(self):
        super().__init__('grasp_node')

        # ------------------------------------------------------------------
        # Parameters
        # ------------------------------------------------------------------
        self.declare_parameter('gripper_toggle_time', 5.0)
        self.declare_parameter('grasp_finger_threshold', 0.033)
        self.declare_parameter('eef_arrival_threshold', 0.1)
        # Neutral/retract pose in base_link (safe overhead position)
        self.declare_parameter('neutral_x',  -0.15)
        self.declare_parameter('neutral_y', -0.40)
        self.declare_parameter('neutral_z',  0.40)
        self.declare_parameter('neutral_qw', _FINGERS_DOWN_HORIZONTAL[0])
        self.declare_parameter('neutral_qx', _FINGERS_DOWN_HORIZONTAL[1])
        self.declare_parameter('neutral_qy', _FINGERS_DOWN_HORIZONTAL[2])
        self.declare_parameter('neutral_qz', _FINGERS_DOWN_HORIZONTAL[3])

        self.arm_name            = None
        self.gripper_state       = 'open'
        self.gripper_toggle_time  = self.get_parameter('gripper_toggle_time').get_parameter_value().double_value
        self.grasp_finger_threshold = self.get_parameter('grasp_finger_threshold').get_parameter_value().double_value
        self.eef_arrival_threshold  = self.get_parameter('eef_arrival_threshold').get_parameter_value().double_value

        neutral = Pose()
        neutral.position.x    = self.get_parameter('neutral_x').get_parameter_value().double_value
        neutral.position.y    = self.get_parameter('neutral_y').get_parameter_value().double_value
        neutral.position.z    = self.get_parameter('neutral_z').get_parameter_value().double_value
        neutral.orientation.w = self.get_parameter('neutral_qw').get_parameter_value().double_value
        neutral.orientation.x = self.get_parameter('neutral_qx').get_parameter_value().double_value
        neutral.orientation.y = self.get_parameter('neutral_qy').get_parameter_value().double_value
        neutral.orientation.z = self.get_parameter('neutral_qz').get_parameter_value().double_value
        self._neutral_pose = neutral

        # ------------------------------------------------------------------
        # TF2 — for EEF arrival check
        # ------------------------------------------------------------------
        self._tf_buffer   = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        # ------------------------------------------------------------------
        # Publishers
        # ------------------------------------------------------------------
        self._object_pose_pub   = self.create_publisher(Pose, '/object_pose_in_camera', 10)
        self._right_gripper_pub = self.create_publisher(Float64MultiArray, '/right_gripper/cmd', 10)
        self._left_gripper_pub  = self.create_publisher(Float64MultiArray, '/left_gripper/cmd', 10)
        self._result_pub        = self.create_publisher(Bool, '/grasp_completed', 10)

        # ------------------------------------------------------------------
        # Subscribers
        # ------------------------------------------------------------------
        self.create_subscription(Pose,       '/pick_target_local',   self._on_pick_target,   10)
        self.create_subscription(Pose,       '/EEF_pose_command',    self._on_eef_pose,       10)
        self.create_subscription(JointState, '/joint_states',        self._on_joint_states,   10)

        # ------------------------------------------------------------------
        # Service client
        # ------------------------------------------------------------------
        self._plan_client = self.create_client(PlanToTarget, '/plan_to_target')

        # ------------------------------------------------------------------
        # State machine
        # ------------------------------------------------------------------
        self._state            = State.IDLE
        self._state_start_time = None
        self._eef_pose         = None   # target EEF pose in base_link
        self._waiting_for_eef  = False  # guard: only capture EEF after relaying a pick target
        self._plan_future      = None   # pending service call future
        self._plan_accepted    = False  # True once the planner has accepted the grasp request
        self._finger_pos       = None   # latest finger position (metres)

        # 20 Hz control loop
        self.create_timer(0.05, self._control_loop)

        self.get_logger().info('=' * 50)
        self.get_logger().info('Grasp Node')
        self.get_logger().info('=' * 50)
        self.get_logger().info(f'  Gripper close time     : {self.gripper_toggle_time:.1f} s')
        self.get_logger().info(f'  EEF arrival threshold  : {self.eef_arrival_threshold:.3f} m')
        self.get_logger().info(f'  Finger grasp threshold : {self.grasp_finger_threshold:.4f} m  '
                               f'(open={_FINGER_OPEN_POS:.3f} m)')
        self.get_logger().info(f'  Neutral pose (base_link): '
                               f'({neutral.position.x:.2f}, '
                               f'{neutral.position.y:.2f}, '
                               f'{neutral.position.z:.2f})')
        self.get_logger().info('Waiting for /pick_target_local ...')

    # ------------------------------------------------------------------
    # Subscriber callbacks
    # ------------------------------------------------------------------

    def _on_pick_target(self, msg: Pose) -> None:
        """Received a new pick target. Only act when idle."""
        if self._state not in (State.IDLE,):
            self.get_logger().warn('Grasp already in progress — ignoring new pick target.')
            return

        self.get_logger().info('Pick target received — relaying to grasp_generation_node.')
        self._waiting_for_eef = True
        self._eef_pose        = None
        self._object_pose_pub.publish(msg)
        self._transition(State.WAIT_EEF_POSE)

    def _on_eef_pose(self, msg: Pose) -> None:
        """Capture the EEF pose only when we are waiting for one."""
        if self._waiting_for_eef:
            self._eef_pose        = msg
            self._waiting_for_eef = False
            self.get_logger().info(
                f'EEF pose received: ({msg.position.x:.3f}, '
                f'{msg.position.y:.3f}, {msg.position.z:.3f})')

    def _on_joint_states(self, msg: JointState) -> None:
        """Cache the latest gripper finger position."""
        finger_joint = f'{self.arm_name}_left_finger'
        if finger_joint in msg.name:
            idx = msg.name.index(finger_joint)
            self._finger_pos = msg.position[idx]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _transition(self, new_state: State) -> None:
        self.get_logger().info(f'  [{self._state.name}] -> [{new_state.name}]')
        self._state            = new_state
        self._state_start_time = time.time()

    def _elapsed(self) -> float:
        if self._state_start_time is None:
            return 0.0
        return time.time() - self._state_start_time

    def _send_gripper(self, position: float) -> None:
        msg = Float64MultiArray()
        msg.data = [position]
        if self.arm_name == 'left':
            self._left_gripper_pub.publish(msg)
        else:
            self._right_gripper_pub.publish(msg)

    def _call_plan_to_target(self, pose: Pose,
                              use_orientation: bool = True,
                              duration: float = 2.0) -> None:
        """Issue an async /plan_to_target call and store the future."""
        if not self._plan_client.service_is_ready():
            self.get_logger().warn('/plan_to_target service not ready.')
            self._plan_future = None
            return
        req = PlanToTarget.Request()
        req.arm_name             = self.arm_name
        req.target_pose          = pose
        req.use_orientation      = use_orientation
        req.execute              = True
        req.duration             = duration
        req.max_condition_number = 100.0
        self._plan_future = self._plan_client.call_async(req)

    def _eef_distance_to_target(self) -> float | None:
        """
        Look up the current ee_arm_link position via TF and return its
        Euclidean distance to self._eef_pose in base_link.
        Returns None if the TF lookup fails.
        """
        ee_frame = f'{self.arm_name}_ee_arm_link'
        try:
            tf = self._tf_buffer.lookup_transform(
                'base_link', ee_frame, rclpy.time.Time())
        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException):
            return None

        t = tf.transform.translation
        p = self._eef_pose.position
        return math.sqrt((t.x - p.x)**2 + (t.y - p.y)**2 + (t.z - p.z)**2)

    # ------------------------------------------------------------------
    # State machine
    # ------------------------------------------------------------------

    def _control_loop(self) -> None:
        """20 Hz state machine tick."""

        # ── IDLE ─────────────────────────────────────────────────────────
        if self._state == State.IDLE:
            return

        # ── WAIT_EEF_POSE ────────────────────────────────────────────────
        elif self._state == State.WAIT_EEF_POSE:
            if self._eef_pose is not None:
                self._transition(State.PLAN_GRASP)
            elif self._elapsed() > 5.0:
                self.get_logger().error('Timed out waiting for /EEF_pose_command.')
                self._transition(State.FAILED)

        # ── PLAN_GRASP (single entry tick) ───────────────────────────────
        elif self._state == State.PLAN_GRASP:
            self.get_logger().info(
                f'Calling /plan_to_target: '
                f'({self._eef_pose.position.x:.3f}, '
                f'{self._eef_pose.position.y:.3f}, '
                f'{self._eef_pose.position.z:.3f})')
            if not self.arm_name:
                self.arm_name = 'left' if self._eef_pose.position.x > 0 else 'right'
            self._plan_accepted = False
            self._call_plan_to_target(self._eef_pose, use_orientation=True, duration=2.0)
            self._transition(State.WAIT_PLAN_GRASP)

        # ── WAIT_PLAN_GRASP ──────────────────────────────────────────────
        # Waits for two conditions:
        #   1. The planner service responds (to catch planning failures early)
        #   2. The EEF (ee_arm_link) is within eef_arrival_threshold of the target
        elif self._state == State.WAIT_PLAN_GRASP:
            if self._plan_future is None:
                self.get_logger().error('No plan future — service may not be available.')
                self._transition(State.FAILED)
                return

            # Check for planning failure as soon as the service responds
            if not self._plan_accepted and self._plan_future.done():
                result = self._plan_future.result()
                if result is None or not result.success:
                    msg = result.message if result else 'no result'
                    self.get_logger().error(f'Planning failed: {msg}')
                    self._transition(State.FAILED)
                    return
                self.get_logger().info(
                    f'Plan accepted (pos_err={result.position_error:.4f} m) — '
                    f'waiting for EEF to arrive ...')
                self._plan_accepted = True

            # Poll EEF distance via TF
            dist = self._eef_distance_to_target()
            if dist is not None:
                self.get_logger().debug(
                    f'EEF distance to target: {dist:.4f} m '
                    f'(threshold: {self.eef_arrival_threshold:.3f} m)')
                if dist < self.eef_arrival_threshold:
                    self.get_logger().info(
                        f'EEF arrived at target (dist={dist:.4f} m). Closing gripper.')
                    self._transition(State.TOGGLE_GRIPPER)
                    return

            if self._elapsed() > 20.0:
                self.get_logger().error('Timed out waiting for EEF to arrive.')
                self._transition(State.FAILED)

        # ── TOGGLE_GRIPPER ────────────────────────────────────────────────
        elif self._state == State.TOGGLE_GRIPPER:
            elapsed = self._elapsed()
            if self.gripper_state == 'closed':
                if elapsed < 0.1:
                    self.get_logger().info('Opening gripper ...')
                self._send_gripper(_GRIPPER_OPEN)
            else:
                if elapsed < 0.1:
                    self.get_logger().info('Closing gripper ...')
                self._send_gripper(_GRIPPER_CLOSED)
            if elapsed > self.gripper_toggle_time:
                if self.gripper_state == 'closed': 
                    self.gripper_state = 'open'
                    self.get_logger().info('Gripper opened')
                else: 
                    self.gripper_state = 'closed'
                    self.get_logger().info('Gripper closed')
                self._transition(State.PLAN_NEUTRAL)   
                return

        # ── PLAN_NEUTRAL (single entry tick) ─────────────────────────────
        elif self._state == State.PLAN_NEUTRAL:
            if self.arm_name == 'left':
                self._neutral_pose.position.x = -self._neutral_pose.position.x
            self.get_logger().info(
                f'Retracting to neutral pose '
                f'({self._neutral_pose.position.x:.2f}, '
                f'{self._neutral_pose.position.y:.2f}, '
                f'{self._neutral_pose.position.z:.2f}) ...')
            self._call_plan_to_target(self._neutral_pose, use_orientation=True, duration=2.0)
            self._transition(State.WAIT_PLAN_NEUTRAL)

        # ── WAIT_PLAN_NEUTRAL ────────────────────────────────────────────
        elif self._state == State.WAIT_PLAN_NEUTRAL:
            if self._plan_future is None:
                self.get_logger().warn('No retract future — skipping to grasp check.')
                self._transition(State.CHECK_GRASP)
                return
            if not self._plan_future.done():
                if self._elapsed() > 20.0:
                    self.get_logger().warn('Retract plan timed out — proceeding to grasp check.')
                    self._transition(State.CHECK_GRASP)
                return
            result = self._plan_future.result()
            if result is None or not result.success:
                self.get_logger().warn(
                    f'Retract planning failed ({result.message if result else "no result"}) '
                    '— still checking grasp.')
            else:
                self.get_logger().info('Arm retracted successfully.')
            self._transition(State.CHECK_GRASP)

        # ── CHECK_GRASP ──────────────────────────────────────────────────
        elif self._state == State.CHECK_GRASP:
            if self._finger_pos is None:
                self.get_logger().warn('No joint state received yet — retrying ...')
                if self._elapsed() > 3.0:
                    self.get_logger().error('Timed out waiting for joint states.')
                    self._transition(State.FAILED)
                return

            self.get_logger().info(
                f'Finger position: {self._finger_pos:.4f} m  '
                f'(threshold < {self.grasp_finger_threshold:.4f} m for success)')

            if self._finger_pos < self.grasp_finger_threshold:
                self.get_logger().info('Grasp SUCCESSFUL — object detected in gripper.')
                self._transition(State.DONE)
            else:
                self.get_logger().warn('Grasp FAILED — fingers fully closed, no object detected.')
                self._transition(State.FAILED)

        # ── DONE ─────────────────────────────────────────────────────────
        elif self._state == State.DONE:
            if self._elapsed() < 0.1:
                result_msg = Bool()
                result_msg.data = True
                self._result_pub.publish(result_msg)
                self.get_logger().info('=' * 50)
                self.get_logger().info('Grasp complete! Published grasp_completed=True')
                self.get_logger().info('=' * 50)
                self._transition(State.IDLE)

        # ── FAILED ───────────────────────────────────────────────────────
        elif self._state == State.FAILED:
            if self._elapsed() < 0.1:
                self._send_gripper(_GRIPPER_OPEN)
                result_msg = Bool()
                result_msg.data = False
                self._result_pub.publish(result_msg)
                self.get_logger().error('Grasp FAILED. Published grasp_completed=False')
                self._transition(State.IDLE)


def main(args=None):
    rclpy.init(args=args)
    node = GraspNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
