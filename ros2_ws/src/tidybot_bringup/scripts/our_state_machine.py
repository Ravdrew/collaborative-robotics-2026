#!/usr/bin/env python3
"""
Task-oriented ROS 2 state machine for exploration + navigation + pick/place.

State flow:
1) audio_processing:
   wait for non-empty /pick_target and /place_target.
2) pick_exploration:
   start frontier exploration, wait for fruit detection via /pick_target_local.
3) pick_navigation:
   send NavigateToPose goal to Nav2, wait for result.
4) picking:
   publish /fsm_pick_request and wait for /successful_pick true.
5) place_exploration:
   start frontier exploration, wait for bowl detection via /place_target_local.
6) place_navigation:
   send NavigateToPose goal to Nav2, wait for result.
7) placing:
   publish /fsm_place_request and wait for /placing_done.
8) finished:
   terminal state.
"""

import math
from enum import Enum
from typing import Optional

import rclpy
import tf2_ros
from geometry_msgs.msg import Pose, PoseStamped, Quaternion
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool, String


class SMState(str, Enum):
    AUDIO_PROCESSING = "audio_processing"
    PICK_EXPLORATION = "pick_exploration"
    PICK_NAVIGATION = "pick_navigation"
    PICKING = "picking"
    PLACE_EXPLORATION = "place_exploration"
    PLACE_NAVIGATION = "place_navigation"
    PLACING = "placing"
    FINISHED = "finished"


class StateMachineNode(Node):
    def __init__(self):
        super().__init__("state_machine_node")

        # ---- Parameters ----
        self.declare_parameter("state_topic", "/state_machine")
        self.declare_parameter("pick_target_topic", "/pick_target")
        self.declare_parameter("place_target_topic", "/place_target")
        self.declare_parameter("pick_target_local_topic", "/pick_target_local")
        self.declare_parameter("place_target_local_topic", "/place_target_local")
        self.declare_parameter("successful_pick_topic", "/successful_pick")
        self.declare_parameter("placing_done_topic", "/placing_done")
        self.declare_parameter("fsm_pick_request_topic", "/fsm_pick_request")
        self.declare_parameter("fsm_place_request_topic", "/fsm_place_request")
        self.declare_parameter("explore_resume_topic", "explore/resume")
        self.declare_parameter("nav_offset_m", 0.20)
        self.declare_parameter("camera_frame", "camera_color_optical_frame")
        self.declare_parameter("map_frame", "map")

        # ---- State ----
        self.state: SMState = SMState.AUDIO_PROCESSING
        self.pick_target_ok = False
        self.place_target_ok = False
        self.place_target_value = ""
        self.pick_map_pose: Optional[Pose] = None
        self.place_map_pose: Optional[Pose] = None
        self.nav_goal_handle = None

        self.nav_offset_m = float(self.get_parameter("nav_offset_m").value)
        self.camera_frame = str(self.get_parameter("camera_frame").value)
        self.map_frame = str(self.get_parameter("map_frame").value)
        self.pick_attempt = 0
        self.max_pick_attempts = 3

        self.transition_count = 0
        self.last_transition_reason = "startup"
        self.state_enter_ns = self.get_clock().now().nanoseconds
        self.event_counts = {
            "pick_target": 0,
            "place_target": 0,
            "pick_target_local": 0,
            "place_target_local": 0,
            "successful_pick": 0,
            "placing_done": 0,
            "explore_resume_sent": 0,
            "nav_goal_sent": 0,
            "fsm_pick_request_sent": 0,
            "fsm_place_request_sent": 0,
        }

        # ---- TF2 ----
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ---- Publishers ----
        self.state_pub = self.create_publisher(
            String, str(self.get_parameter("state_topic").value), 10
        )
        self.explore_resume_pub = self.create_publisher(
            Bool, str(self.get_parameter("explore_resume_topic").value), 10
        )
        self.fsm_pick_request_pub = self.create_publisher(
            Bool, str(self.get_parameter("fsm_pick_request_topic").value), 10
        )
        self.fsm_place_request_pub = self.create_publisher(
            Bool, str(self.get_parameter("fsm_place_request_topic").value), 10
        )
        self.detection_enabled_pub = self.create_publisher(
            Bool, "/detection_enabled", 10
        )
        self.cmd_vel_pub = self.create_publisher(Twist, "/cmd_vel", 10)

        # Nudge behavior: slow turn if no detection after waiting
        self.nudge_timeout_timer = None
        self.nudge_step_timer = None
        self.nudge_step = 0  # 0=left, 1=dwell_left, 2=right, 3=dwell_right, 4=center, 5=done
        self.nudge_speed = 0.05  # rad/s — very slow
        self.nudge_angle_deg = 5.0
        self.nudge_turn_duration = math.radians(self.nudge_angle_deg) / self.nudge_speed  # ~1.7s
        self.nudge_dwell = 3.0  # seconds to dwell at each offset

        # ---- Subscriptions ----
        self.create_subscription(
            String,
            str(self.get_parameter("pick_target_topic").value),
            self._on_pick_target,
            10,
        )
        self.create_subscription(
            String,
            str(self.get_parameter("place_target_topic").value),
            self._on_place_target,
            10,
        )
        self.create_subscription(
            Pose,
            str(self.get_parameter("pick_target_local_topic").value),
            self._on_pick_target_local,
            10,
        )
        self.create_subscription(
            Pose,
            str(self.get_parameter("place_target_local_topic").value),
            self._on_place_target_local,
            10,
        )
        self.create_subscription(
            Bool,
            str(self.get_parameter("successful_pick_topic").value),
            self._on_successful_pick,
            10,
        )
        self.create_subscription(
            Bool,
            str(self.get_parameter("placing_done_topic").value),
            self._on_placing_done,
            10,
        )

        # ---- Nav2 Action Client ----
        self.nav_action_client = ActionClient(
            self, NavigateToPose, "navigate_to_pose"
        )

        # ---- Timers ----
        self.create_timer(5.0, self._periodic_state_log)

        # Keep stopping exploration until we actually need it (explore_lite
        # starts navigating immediately on launch, so we suppress it)
        self._explore_suppress_timer = self.create_timer(
            1.0, self._suppress_exploration
        )

        # Initial publish/logs
        # self._publish_state()
        # self.get_logger().info(f"Started in state: {self.state.value}")
        # self.get_logger().info(
        #     "Waiting for /pick_target and /place_target before entering pick_exploration"
        # )

    def _suppress_exploration(self):
        """Repeatedly publish stop until we enter an exploration state."""
        if self.state in (SMState.PICK_EXPLORATION, SMState.PLACE_EXPLORATION):
            # We want exploration now — stop suppressing
            self._explore_suppress_timer.cancel()
            return
        self.explore_resume_pub.publish(Bool(data=False))

    # ===================== Callbacks =====================

    def _on_pick_target(self, msg: String):
        self.event_counts["pick_target"] += 1
        self.get_logger().info(f"Event /pick_target: '{msg.data}'")
        if msg.data.strip():
            self.pick_target_ok = True
            self._maybe_finish_audio()
        else:
            self.get_logger().warn("Received empty /pick_target; ignoring")

    def _on_place_target(self, msg: String):
        self.event_counts["place_target"] += 1
        self.get_logger().info(f"Event /place_target: '{msg.data}'")
        if msg.data.strip():
            self.place_target_value = msg.data.strip().lower()
            self.place_target_ok = True
            self._maybe_finish_audio()
        else:
            self.get_logger().warn("Received empty /place_target; ignoring")

    def _on_pick_target_local(self, msg: Pose):
        self.event_counts["pick_target_local"] += 1

        map_pose = self._transform_to_map(msg)
        if map_pose is None:
            self.get_logger().error("Failed to transform pick target to map frame")
            return

        self.pick_map_pose = map_pose
        self.get_logger().info(
            f"Pick target in map: ({map_pose.position.x:.2f}, {map_pose.position.y:.2f}) "
            f"[state={self.state.value}]"
        )

        if self.state == SMState.PICK_EXPLORATION:
            self._publish_explore_resume(False)
            self._transition(SMState.PICK_NAVIGATION, "pick target detected and transformed")

    def _on_place_target_local(self, msg: Pose):
        self.event_counts["place_target_local"] += 1

        map_pose = self._transform_to_map(msg)
        if map_pose is None:
            self.get_logger().error("Failed to transform place target to map frame")
            return

        self.place_map_pose = map_pose
        self.get_logger().info(
            f"Place target in map: ({map_pose.position.x:.2f}, {map_pose.position.y:.2f}) "
            f"[state={self.state.value}]"
        )

        if self.state == SMState.PLACE_EXPLORATION:
            self._publish_explore_resume(False)
            self._transition(SMState.PLACE_NAVIGATION, "place target detected and transformed")

    def _on_successful_pick(self, msg: Bool):
        self.event_counts["successful_pick"] += 1
        if self.state != SMState.PICKING:
            self.get_logger().warn(
                f"Ignoring /successful_pick={msg.data} in state={self.state.value}"
            )
            return

        if bool(msg.data):
            self.pick_attempt = 0
            if self.place_target_value == "none":
                self._transition(SMState.FINISHED, "successful_pick=true, no place target")
            else:
                self._transition(SMState.PLACE_EXPLORATION, "successful_pick=true")
        else:
            self.pick_attempt += 1
            if self.pick_attempt < self.max_pick_attempts:
                self.get_logger().warn(
                    f"Pick failed (attempt {self.pick_attempt}/{self.max_pick_attempts}), retrying..."
                )
                self._publish_pick_request_once()
            else:
                self.get_logger().error(
                    f"Pick failed after {self.max_pick_attempts} attempts, giving up"
                )
                self.pick_attempt = 0
                self._transition(SMState.FINISHED, "pick failed after max retries")

    def _on_placing_done(self, msg: Bool):
        self.event_counts["placing_done"] += 1
        if self.state != SMState.PLACING:
            self.get_logger().warn(
                f"Ignoring /placing_done={msg.data} in state={self.state.value}"
            )
            return

        if bool(msg.data):
            self._transition(SMState.FINISHED, "placing_done=true")
        else:
            self.get_logger().warn("placing_done=false; staying in placing and waiting")

    # ===================== Nav2 Action =====================

    def _send_nav_goal(self, target_pose: Pose, phase: str):
        """Compute an offset goal and send NavigateToPose to Nav2."""
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = PoseStamped()
        goal_msg.pose.header.frame_id = self.map_frame
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()

        # Compute approach pose: offset toward robot's current position
        robot_pose = self._get_robot_pose_in_map()
        tx, ty = target_pose.position.x, target_pose.position.y

        if robot_pose is not None:
            rx, ry = robot_pose.position.x, robot_pose.position.y
            dx, dy = rx - tx, ry - ty
            dist = math.hypot(dx, dy)
            if dist > 1e-3:
                offset_x = tx + self.nav_offset_m * dx / dist
                offset_y = ty + self.nav_offset_m * dy / dist
                yaw = math.atan2(-dy, -dx)  # face the target
            else:
                offset_x, offset_y, yaw = tx, ty, 0.0
        else:
            self.get_logger().warn("Could not get robot pose; navigating directly to target")
            offset_x, offset_y, yaw = tx, ty, 0.0

        goal_msg.pose.pose.position.x = offset_x
        goal_msg.pose.pose.position.y = offset_y
        goal_msg.pose.pose.position.z = 0.0
        goal_msg.pose.pose.orientation = self._yaw_to_quaternion(yaw)

        self.get_logger().info(
            f"Sending Nav2 goal for {phase}: ({offset_x:.2f}, {offset_y:.2f}, yaw={yaw:.2f})"
        )

        if not self.nav_action_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("Nav2 action server not available!")
            return

        send_future = self.nav_action_client.send_goal_async(goal_msg)
        send_future.add_done_callback(
            lambda future: self._nav_goal_response_cb(future, phase)
        )
        self.event_counts["nav_goal_sent"] += 1

    def _nav_goal_response_cb(self, future, phase: str):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error(f"Nav2 goal rejected for {phase}")
            return

        self.nav_goal_handle = goal_handle
        self.get_logger().info(f"Nav2 goal accepted for {phase}")
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(
            lambda future: self._nav_result_cb(future, phase)
        )

    def _nav_result_cb(self, future, phase: str):
        result = future.result()
        status = result.status
        # action_msgs/GoalStatus: STATUS_SUCCEEDED = 4
        if status == 4:
            self.get_logger().info(f"Nav2 succeeded for {phase}")
            if phase == "pick" and self.state == SMState.PICK_NAVIGATION:
                self._transition(SMState.PICKING, "nav2 succeeded for pick")
            elif phase == "place" and self.state == SMState.PLACE_NAVIGATION:
                self._transition(SMState.PLACING, "nav2 succeeded for place")
        else:
            self.get_logger().warn(
                f"Nav2 failed for {phase} with status={status}; retrying..."
            )
            if phase == "pick" and self.state == SMState.PICK_NAVIGATION and self.pick_map_pose is not None:
                self._send_nav_goal(self.pick_map_pose, "pick")
            elif phase == "place" and self.state == SMState.PLACE_NAVIGATION and self.place_map_pose is not None:
                self._send_nav_goal(self.place_map_pose, "place")

    # ===================== Helpers =====================

    def _maybe_finish_audio(self):
        if (
            self.state == SMState.AUDIO_PROCESSING
            and self.pick_target_ok
            and self.place_target_ok
        ):
            self._transition(
                SMState.PICK_EXPLORATION,
                "both /pick_target and /place_target received",
            )

    def _publish_detection_enabled(self, enabled: bool):
        msg = Bool()
        msg.data = enabled
        self.detection_enabled_pub.publish(msg)
        self.get_logger().info(f"Published /detection_enabled={enabled}")

    def _start_nudge_timeout(self):
        """Start a 10s timer; if it fires, begin the nudge sequence."""
        self._cancel_nudge()
        self.nudge_timeout_timer = self.create_timer(15.0, self._begin_nudge, callback_group=None)

    def _cancel_nudge(self):
        """Cancel any active nudge timers."""
        if self.nudge_timeout_timer is not None:
            self.nudge_timeout_timer.cancel()
            self.nudge_timeout_timer = None
        if self.nudge_step_timer is not None:
            self.nudge_step_timer.cancel()
            self.nudge_step_timer = None
        self.nudge_step = 0
        # Stop any residual rotation
        self.cmd_vel_pub.publish(Twist())

    def _begin_nudge(self):
        """Called after 10s with no detection. Start nudge sequence."""
        if self.nudge_timeout_timer is not None:
            self.nudge_timeout_timer.cancel()
            self.nudge_timeout_timer = None
        self.get_logger().info("No detection after 15s — starting nudge (turn left/right)")
        self.nudge_step = 0
        self._execute_nudge_step()

    def _execute_nudge_step(self):
        """Step through the nudge sequence: left, dwell, right, dwell, center."""
        if self.nudge_step_timer is not None:
            self.nudge_step_timer.cancel()
            self.nudge_step_timer = None

        twist = Twist()

        if self.nudge_step == 0:
            # Turn left (positive angular.z)
            self.get_logger().info("Nudge: turning left 5°")
            twist.angular.z = self.nudge_speed
            self.cmd_vel_pub.publish(twist)
            self.nudge_step_timer = self.create_timer(
                self.nudge_turn_duration, self._nudge_next)

        elif self.nudge_step == 1:
            # Stop and dwell
            self.cmd_vel_pub.publish(twist)
            self.get_logger().info("Nudge: dwelling left")
            self.nudge_step_timer = self.create_timer(self.nudge_dwell, self._nudge_next)

        elif self.nudge_step == 2:
            # Turn right (10° to go from +5° to -5°)
            self.get_logger().info("Nudge: turning right 10°")
            twist.angular.z = -self.nudge_speed
            self.cmd_vel_pub.publish(twist)
            self.nudge_step_timer = self.create_timer(
                self.nudge_turn_duration * 2.0, self._nudge_next)

        elif self.nudge_step == 3:
            # Stop and dwell
            self.cmd_vel_pub.publish(twist)
            self.get_logger().info("Nudge: dwelling right")
            self.nudge_step_timer = self.create_timer(self.nudge_dwell, self._nudge_next)

        elif self.nudge_step == 4:
            # Return to center (turn left 5°)
            self.get_logger().info("Nudge: returning to center")
            twist.angular.z = self.nudge_speed
            self.cmd_vel_pub.publish(twist)
            self.nudge_step_timer = self.create_timer(
                self.nudge_turn_duration, self._nudge_next)

        else:
            # Done
            self.cmd_vel_pub.publish(twist)
            self.get_logger().info("Nudge sequence complete")

    def _nudge_next(self):
        """Advance to the next nudge step."""
        if self.nudge_step_timer is not None:
            self.nudge_step_timer.cancel()
            self.nudge_step_timer = None
        self.nudge_step += 1
        self._execute_nudge_step()

    def _publish_explore_resume(self, resume: bool):
        msg = Bool()
        msg.data = resume
        self.explore_resume_pub.publish(msg)
        self.event_counts["explore_resume_sent"] += 1
        self.get_logger().info(f"Published explore/resume={resume}")

    def _publish_pick_request_once(self):
        self.fsm_pick_request_pub.publish(Bool(data=True))
        self.event_counts["fsm_pick_request_sent"] += 1
        self.get_logger().info("Published /fsm_pick_request=true (enter PICKING)")

    def _publish_place_request_once(self):
        self.fsm_place_request_pub.publish(Bool(data=True))
        self.event_counts["fsm_place_request_sent"] += 1
        self.get_logger().info("Published /fsm_place_request=true (enter PLACING)")

    def _transform_to_map(self, local_pose: Pose) -> Optional[Pose]:
        """Transform a Pose from camera frame to map frame using TF2."""
        try:
            transform = self.tf_buffer.lookup_transform(
                self.map_frame,
                self.camera_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=1.0),
            )
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as e:
            self.get_logger().error(f"TF2 lookup failed: {e}")
            return None

        # Manual transform application
        t = transform.transform.translation
        r = transform.transform.rotation

        # Convert quaternion rotation to apply to the point
        px, py, pz = local_pose.position.x, local_pose.position.y, local_pose.position.z
        # Rotate point by quaternion: p' = q * p * q_inv
        rx, ry, rz, rw = r.x, r.y, r.z, r.w
        # Using quaternion rotation formula
        tx_out = (1 - 2 * (ry * ry + rz * rz)) * px + 2 * (rx * ry - rz * rw) * py + 2 * (rx * rz + ry * rw) * pz + t.x
        ty_out = 2 * (rx * ry + rz * rw) * px + (1 - 2 * (rx * rx + rz * rz)) * py + 2 * (ry * rz - rx * rw) * pz + t.y
        tz_out = 2 * (rx * rz - ry * rw) * px + 2 * (ry * rz + rx * rw) * py + (1 - 2 * (rx * rx + ry * ry)) * pz + t.z

        result = Pose()
        result.position.x = tx_out
        result.position.y = ty_out
        result.position.z = tz_out
        return result

    def _get_robot_pose_in_map(self) -> Optional[Pose]:
        """Get robot's current position in map frame via TF2."""
        try:
            transform = self.tf_buffer.lookup_transform(
                self.map_frame,
                "base_link",
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=1.0),
            )
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as e:
            self.get_logger().warn(f"Could not get robot pose: {e}")
            return None

        pose = Pose()
        pose.position.x = transform.transform.translation.x
        pose.position.y = transform.transform.translation.y
        pose.position.z = transform.transform.translation.z
        pose.orientation = transform.transform.rotation
        return pose

    @staticmethod
    def _yaw_to_quaternion(yaw: float) -> Quaternion:
        q = Quaternion()
        q.x = 0.0
        q.y = 0.0
        q.z = math.sin(yaw / 2.0)
        q.w = math.cos(yaw / 2.0)
        return q

    def _on_state_entry(self):
        """Run one-shot actions on state entry."""
        if self.state == SMState.PICK_EXPLORATION:
            if self.pick_map_pose is not None:
                self.get_logger().info("Pick target already known, skipping exploration")
                self._transition(SMState.PICK_NAVIGATION, "pick target already saved")
                return
            self._publish_explore_resume(True)

        elif self.state == SMState.PICK_NAVIGATION:
            self._publish_detection_enabled(False)
            if self.pick_map_pose is not None:
                self._send_nav_goal(self.pick_map_pose, "pick")
            else:
                self.get_logger().error("No pick_map_pose set for PICK_NAVIGATION!")

        elif self.state == SMState.PICKING:
            self._publish_detection_enabled(True)
            self._publish_pick_request_once()
            self._start_nudge_timeout()

        elif self.state == SMState.PLACE_EXPLORATION:
            if self.place_map_pose is not None:
                self.get_logger().info("Place target already known, skipping exploration")
                self._transition(SMState.PLACE_NAVIGATION, "place target already saved")
                return
            self._publish_explore_resume(True)

        elif self.state == SMState.PLACE_NAVIGATION:
            self._publish_detection_enabled(False)
            if self.place_map_pose is not None:
                self._send_nav_goal(self.place_map_pose, "place")
            else:
                self.get_logger().error("No place_map_pose set for PLACE_NAVIGATION!")

        elif self.state == SMState.PLACING:
            self._publish_detection_enabled(True)
            self._publish_place_request_once()
            self._start_nudge_timeout()

        elif self.state == SMState.FINISHED:
            self.get_logger().info(":) Mission complete! All done.")

    def _transition(self, new_state: SMState, reason: str):
        if new_state == self.state:
            return

        prev_state = self.state
        now_ns = self.get_clock().now().nanoseconds
        dwell_s = (now_ns - self.state_enter_ns) * 1e-9

        self._cancel_nudge()
        self.state = new_state
        self.transition_count += 1
        self.last_transition_reason = reason
        self.state_enter_ns = now_ns
        self._on_state_entry()
        self._publish_state()

        self.get_logger().info(
            f"Transition #{self.transition_count}: {prev_state.value} -> {self.state.value} "
            f"after {dwell_s:.2f}s (reason: {reason})"
        )

    def _publish_state(self):
        out = String()
        out.data = self.state.value
        self.state_pub.publish(out)
        self.get_logger().info(f"Published state: '{self.state.value}'")

    def _periodic_state_log(self):
        now_ns = self.get_clock().now().nanoseconds
        dwell_s = (now_ns - self.state_enter_ns) * 1e-9
        self.get_logger().info(f"[state] {self.state.value} (dwell={dwell_s:.1f}s)")


def main():
    rclpy.init()
    node = StateMachineNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
