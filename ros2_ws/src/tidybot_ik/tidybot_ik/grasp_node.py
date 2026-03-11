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
    ros2 launch tidybot_bringup sim.launch.py OR ros2 launch tidybot_bringup real.launch.py

    # Terminal 2 - publish a pick target, then trigger a grasp
    ros2 topic pub --once /pick_target_local geometry_msgs/msg/Pose "{position: {x: 0.05, y: 0.25, z: 0.25}, orientation: {w: 1.0}}"
    ros2 topic pub --once /fsm_pick_request std_msgs/msg/Bool "{data: true}"
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
from tidybot_control.gripper_controller import GripperController


# ---------------------------------------------------------------------------
# Top-down grasp orientation (fingers-down) in base_link frame — wxyz
# Matches test_planner_sim.py
# ---------------------------------------------------------------------------
_FINGERS_DOWN_HORIZONTAL = (0.5, 0.5, 0.5, -0.5)  # (qw, qx, qy, qz)
_FINGERS_DOWN_VERTICAL = (0, 0.707107, 0, 0.707107)   # (qw, qx, qy, qz)

_GRIPPER_CLOSE_SETTLE_TIME = 1.0

_CAMERA_PAN = 0.6
_CAMERA_NEUTRAL = 0.3

# Finger joint open position (metres) — from MuJoCo model range
_FINGER_OPEN_POS = 0.037


class State(Enum):
    IDLE              = auto()  # waiting for /fsm_pick_request or /fsm_place_request
    PAN_CAMERA_DOWN   = auto()  # tilting camera down to see workspace before relaying target
    WAIT_EEF_POSE     = auto()  # waiting for /EEF_pose_command after relaying pick target
    PLAN_GRASP        = auto()  # issuing /plan_to_target call (single entry tick)
    WAIT_PLAN_GRASP   = auto()  # waiting for plan failure OR EEF arrival at target
    CLOSE_GRIPPER     = auto()  # sending close command for gripper_toggle_time seconds (pick)
    OPEN_GRIPPER      = auto()  # sending open command for gripper_toggle_time seconds (place)
    PLAN_NEUTRAL      = auto()  # issuing /plan_to_target for retract (single entry tick)
    WAIT_PLAN_NEUTRAL = auto()  # waiting for retract future to resolve
    CHECK_GRASP       = auto()  # inspect finger position
    PAN_CAMERA_UP     = auto()  # tilting camera back to neutral before signalling done
    DONE              = auto()  # action succeeded
    FAILED            = auto()  # action failed


class GraspNode(Node):
    """State-machine node that executes a full grasp sequence."""

    def __init__(self):
        super().__init__('grasp_node')

        # ------------------------------------------------------------------
        # Parameters
        # ------------------------------------------------------------------
        self.declare_parameter('gripper_toggle_time', 5.0)
        self.declare_parameter('grasp_finger_threshold', 0.033)
        self.declare_parameter('eef_arrival_threshold', 0.05)
        self.declare_parameter('camera_pan_time', 1.5)
        self.declare_parameter('eef_stall_timeout', 5.0)
        self.declare_parameter('eef_stall_threshold', 0.005)
        self.declare_parameter('gripper_mode', 'sim')
        self.declare_parameter('gripper_pressure', 1.0)
        # Neutral/retract pose in base_link (safe overhead position)
        self.declare_parameter('neutral_x',  0.40)
        self.declare_parameter('neutral_y', 0.15)
        self.declare_parameter('neutral_z',  0.40)
        self.declare_parameter('neutral_qw', _FINGERS_DOWN_HORIZONTAL[0])
        self.declare_parameter('neutral_qx', _FINGERS_DOWN_HORIZONTAL[1])
        self.declare_parameter('neutral_qy', _FINGERS_DOWN_HORIZONTAL[2])
        self.declare_parameter('neutral_qz', _FINGERS_DOWN_HORIZONTAL[3])

        self.arm_name               = None
        self.gripper_toggle_time    = self.get_parameter('gripper_toggle_time').get_parameter_value().double_value
        self.grasp_finger_threshold = self.get_parameter('grasp_finger_threshold').get_parameter_value().double_value
        self.eef_arrival_threshold  = self.get_parameter('eef_arrival_threshold').get_parameter_value().double_value
        self.camera_pan_time        = self.get_parameter('camera_pan_time').get_parameter_value().double_value
        self.eef_stall_timeout      = self.get_parameter('eef_stall_timeout').get_parameter_value().double_value
        self.eef_stall_threshold    = self.get_parameter('eef_stall_threshold').get_parameter_value().double_value
        self._last_eef_dist = None
        self._last_eef_move_time = None

        gripper_mode = self.get_parameter('gripper_mode').get_parameter_value().string_value
        gripper_pressure = self.get_parameter('gripper_pressure').get_parameter_value().double_value
        self._gripper = GripperController(self, mode=gripper_mode, pressure=gripper_pressure)
        self._gripper_mode = gripper_mode

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
        self._pick_result_pub   = self.create_publisher(Bool, '/successful_pick', 10)
        self._place_result_pub  = self.create_publisher(Bool, '/placing_done', 10)
        self._pan_tilt_pub      = self.create_publisher(Float64MultiArray, '/camera/pan_tilt_cmd', 10)

        # ------------------------------------------------------------------
        # Subscribers
        # ------------------------------------------------------------------
        self.create_subscription(Pose,       '/pick_target_local',   self._on_pick_target, 10)
        self.create_subscription(Pose,       '/place_target_local',  self._on_place_target, 10)
        self.create_subscription(Pose,       '/EEF_pose_command',    self._on_eef_pose, 10)
        self.create_subscription(JointState, '/joint_states',        self._on_joint_states, 10)
        self.create_subscription(Bool,       '/fsm_pick_request',    self._on_pick_start, 10)
        self.create_subscription(Bool,       '/fsm_place_request',   self._on_place_start, 10)
        self.create_subscription(JointState, '/camera/pan_tilt_state', self._on_pan_tilt_state, 10)

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
        self._pick_target_pose = None   # most recent target pose in camera_color_optical_frame
        self._place_target_pose = None   # most recent target pose in camera_color_optical_frame
        self.action            = None   # 'pick' or 'place'
        self.current_pan       = 0.0
        self.current_tilt      = 0.0

        # Pan camera to neutral on startup (fires once after 0.5 s)
        self._startup_timer = self.create_timer(0.5, self._startup_pan)

        # 20 Hz control loop
        self.create_timer(0.05, self._control_loop)

        self.get_logger().info('=' * 50)
        self.get_logger().info('Grasp Node')
        self.get_logger().info('=' * 50)
        self.get_logger().info(f'  Gripper mode           : {self._gripper_mode}')
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
        self._pick_target_pose = msg
    def _on_place_target(self, msg: Pose) -> None:
        self._place_target_pose = msg
    
    def _on_pick_start(self, msg: Bool) -> None:
        """Received a pick command — pan camera down, then move to target and CLOSE gripper."""
        if self._state != State.IDLE:
            self.get_logger().warn('Action already in progress — ignoring pick request.')
            return
        self.get_logger().info('Pick request received — panning camera down.')
        self.action    = 'pick'
        self._eef_pose = None
        self._pick_target_pose = None
        self._transition(State.PAN_CAMERA_DOWN)

    def _on_place_start(self, msg: Bool) -> None:
        """Received a place command — pan camera down, then move to target and OPEN gripper."""
        if self._state != State.IDLE:
            self.get_logger().warn('Action already in progress — ignoring place request.')
            return
        self.get_logger().info('Place request received — panning camera down.')
        self.action    = 'place'
        self._eef_pose = None
        self._place_target_pose = None
        self._transition(State.PAN_CAMERA_DOWN)

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

    def _on_pan_tilt_state(self, msg: JointState) -> None:
        for i, name in enumerate(msg.name):
            if name == 'camera_pan' and i < len(msg.position):
                self.current_pan = msg.position[i]
            elif name == 'camera_tilt' and i < len(msg.position):
                self.current_tilt = msg.position[i]

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
        ee_frame = f'{self.arm_name}_pinch_site'
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

    def _send_pan_tilt(self, pan: float, tilt: float) -> None:
        """Publish a single pan-tilt command (non-blocking)."""
        msg = Float64MultiArray()
        msg.data = [pan, tilt]
        self._pan_tilt_pub.publish(msg)

    def _startup_pan(self) -> None:
        """Pan camera to neutral once on startup, then cancel this timer."""
        self.get_logger().info('Startup: panning camera to neutral (pan=0.0, tilt=0.0).')
        self._send_pan_tilt(0.0, _CAMERA_NEUTRAL)
        self._startup_timer.cancel()

    # ------------------------------------------------------------------
    # State machine
    # ------------------------------------------------------------------

    def _control_loop(self) -> None:
        """20 Hz state machine tick."""

        # ── IDLE ─────────────────────────────────────────────────────────
        if self._state == State.IDLE:
            return

        # ── PAN_CAMERA_DOWN ──────────────────────────────────────────────
        elif self._state == State.PAN_CAMERA_DOWN:
            elapsed = self._elapsed()
            if elapsed < 0.1:
                self.get_logger().info(f'Panning camera down (tilt={_CAMERA_PAN:.2f}) ...')
            self._send_pan_tilt(0.0, _CAMERA_PAN)
            if elapsed > self.camera_pan_time:
                # Camera is in position — wait for a target pose then relay it
                target_pose = (self._pick_target_pose if self.action == 'pick'
                               else self._place_target_pose)
                if target_pose is None:
                    self.get_logger().debug(
                        f'Camera down — waiting for /{self.action}_target_local ...')
                    return
                self.get_logger().info('Camera down — relaying target to grasp_generation_node.')
                self._waiting_for_eef = True
                self._object_pose_pub.publish(target_pose)
                self._transition(State.WAIT_EEF_POSE)

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
                        f'EEF arrived at target (dist={dist:.4f} m). '
                        f'Action: {self.action}.')
                    self._last_eef_dist = None
                    self._last_eef_move_time = None
                    next_state = State.CLOSE_GRIPPER if self.action == 'pick' else State.OPEN_GRIPPER
                    self._transition(next_state)
                    return

            if self._elapsed() > 20.0:
                self.get_logger().warn('Timed out waiting for EEF — closing gripper and raising.')
                self._last_eef_dist = None
                self._last_eef_move_time = None
                next_state = State.CLOSE_GRIPPER if self.action == 'pick' else State.OPEN_GRIPPER
                self._transition(next_state)

        # ── CLOSE_GRIPPER (pick) ─────────────────────────────────────────
        elif self._state == State.CLOSE_GRIPPER:
            if self._elapsed() < _GRIPPER_CLOSE_SETTLE_TIME:
                return
            self.get_logger().info('Closing gripper ...')
            self._gripper.close(self.arm_name, duration=self.gripper_toggle_time,
                                stop_after=False)
            self.get_logger().info('Gripper closed.')
            self._transition(State.PLAN_NEUTRAL)

        # ── OPEN_GRIPPER (place) ─────────────────────────────────────────
        elif self._state == State.OPEN_GRIPPER:
            self.get_logger().info('Opening gripper ...')
            self._gripper.open(self.arm_name, duration=self.gripper_toggle_time)
            self.get_logger().info('Gripper opened.')
            self._transition(State.PLAN_NEUTRAL)

        # ── PLAN_NEUTRAL (single entry tick) ─────────────────────────────
        elif self._state == State.PLAN_NEUTRAL:
            if self.arm_name == 'left':
                self._neutral_pose.position.x = abs(self._neutral_pose.position.x)
            else:
                self._neutral_pose.position.x = -abs(self._neutral_pose.position.x)
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

            if self.action == 'place':
                self.get_logger().info('Place action — skipping grasp check.')
                self._transition(State.PAN_CAMERA_UP)
                return

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
                self._transition(State.PAN_CAMERA_UP)
            else:
                self.get_logger().warn('Grasp FAILED — fingers fully closed, no object detected.')
                self._transition(State.FAILED)

        # ── PAN_CAMERA_UP ────────────────────────────────────────────────
        elif self._state == State.PAN_CAMERA_UP:
            elapsed = self._elapsed()
            if elapsed < 0.1:
                self.get_logger().info('Panning camera back to neutral (tilt=0.0) ...')
            self._send_pan_tilt(0.0, _CAMERA_NEUTRAL)
            if elapsed > self.camera_pan_time:
                self.get_logger().info('Camera back to neutral.')
                self._transition(State.DONE)

        # ── DONE ─────────────────────────────────────────────────────────
        elif self._state == State.DONE:
            if self._elapsed() < 0.1:
                result_msg = Bool()
                result_msg.data = True
                if self.action == 'pick':
                    self._pick_result_pub.publish(result_msg)
                else:
                    self._place_result_pub.publish(result_msg)
                self.get_logger().info('=' * 50)
                self.get_logger().info(f'{self.action.capitalize()} complete! Published {self.action}_result=True')
                self.get_logger().info('=' * 50)
                if self.action == 'place':
                    self.arm_name = None
                self._transition(State.IDLE)

        # ── FAILED ───────────────────────────────────────────────────────
        elif self._state == State.FAILED:
            if self._elapsed() < 0.1:
                if self.arm_name:
                    self._gripper.open(self.arm_name, duration=1.0)
                result_msg = Bool()
                result_msg.data = False
                if self.action == 'pick':
                    self._pick_result_pub.publish(result_msg)
                else:
                    self._place_result_pub.publish(result_msg)
                self.get_logger().error(f'{self.action.capitalize()} FAILED. Published {self.action}_result=False')
                self._transition(State.IDLE)


def main(args=None):
    rclpy.init(args=args)
    node = GraspNode()

    try:
        node._send_pan_tilt(0.0, 0.3)
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
