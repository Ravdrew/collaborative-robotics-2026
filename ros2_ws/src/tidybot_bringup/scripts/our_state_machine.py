#!/usr/bin/env python3
"""
Minimal ROS 2 state machine:

States (in order):
1) audio_processing
   - waits until /pick_target (String) AND /place_target (String) are received (non-empty)

2) pick_navigation
   - transitions when /pick_target_local (PointStamped) is received

3) picking
   - transitions when /successful_pick (Bool) is received
     - if False: do nothing (keep waiting in picking)

4) place_navigation
   - transitions when /place_target_local (PoseStamped) is received

5) placing
   - waits for /placing_done (Bool), then transitions to finished

6) finished
   - stays here

Publishes:
- /state_machine (String): current state name (latched via periodic publish + immediate on transitions)
"""

from enum import Enum

import rclpy
from rclpy.node import Node

from std_msgs.msg import String, Bool
from geometry_msgs.msg import PointStamped, PoseStamped


class SMState(str, Enum):
    AUDIO_PROCESSING = "audio_processing"
    PICK_NAVIGATION = "pick_navigation"
    PICKING = "picking"
    PLACE_NAVIGATION = "place_navigation"
    PLACING = "placing"
    FINISHED = "finished"


class StateMachineNode(Node):
    def __init__(self):
        super().__init__("state_machine_node")

        # ---- Params (easy to remap) ----
        self.declare_parameter("state_topic", "/state_machine")
        self.declare_parameter("state_pub_hz", 5.0)

        self.declare_parameter("pick_target_topic", "/pick_target")
        self.declare_parameter("place_target_topic", "/place_target")

        self.declare_parameter("pick_target_local_topic", "/pick_target_local")
        self.declare_parameter("successful_pick_topic", "/successful_pick")

        self.declare_parameter("place_target_local_topic", "/place_target_local")
        self.declare_parameter("placing_done_topic", "/placing_done")
        self.declare_parameter("heartbeat_log_s", 2.0)

        # ---- State ----
        self.state: SMState = SMState.AUDIO_PROCESSING
        self.pick_target_ok = False
        self.place_target_ok = False
        self.transition_count = 0
        self.last_transition_reason = "startup"
        self.state_enter_ns = self.get_clock().now().nanoseconds
        self.event_counts = {
            "pick_target": 0,
            "place_target": 0,
            "pick_target_local": 0,
            "successful_pick": 0,
            "place_target_local": 0,
            "placing_done": 0,
        }

        # ---- Publisher ----
        self.state_pub = self.create_publisher(
            String, self.get_parameter("state_topic").get_parameter_value().string_value, 10
        )

        # ---- Subscriptions ----
        self.create_subscription(
            String,
            self.get_parameter("pick_target_topic").get_parameter_value().string_value,
            self._on_pick_target,
            10,
        )
        self.create_subscription(
            String,
            self.get_parameter("place_target_topic").get_parameter_value().string_value,
            self._on_place_target,
            10,
        )

        self.create_subscription(
            PointStamped,
            self.get_parameter("pick_target_local_topic").get_parameter_value().string_value,
            self._on_pick_target_local,
            10,
        )

        self.create_subscription(
            Bool,
            self.get_parameter("successful_pick_topic").get_parameter_value().string_value,
            self._on_successful_pick,
            10,
        )

        self.create_subscription(
            PoseStamped,
            self.get_parameter("place_target_local_topic").get_parameter_value().string_value,
            self._on_place_target_local,
            10,
        )

        self.create_subscription(
            Bool,
            self.get_parameter("placing_done_topic").get_parameter_value().string_value,
            self._on_placing_done,
            10,
        )

        # ---- Timer to continuously publish state ----
        hz = float(self.get_parameter("state_pub_hz").get_parameter_value().double_value)
        period = 1.0 / max(hz, 0.1)
        self.create_timer(period, self._publish_state)
        heartbeat_period = float(self.get_parameter("heartbeat_log_s").value)
        self.create_timer(max(0.5, heartbeat_period), self._heartbeat_log)

        # Publish initial state immediately
        self._publish_state()
        self.get_logger().info(f"Started in state: {self.state.value}")
        self.get_logger().info(
            "Waiting for /pick_target and /place_target before transitioning out of audio_processing"
        )

    # ---------------- Callbacks ----------------

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
            self.place_target_ok = True
            self._maybe_finish_audio()
        else:
            self.get_logger().warn("Received empty /place_target; ignoring")

    def _on_pick_target_local(self, _msg: PointStamped):
        self.event_counts["pick_target_local"] += 1
        if self.state == SMState.PICK_NAVIGATION:
            self._transition(SMState.PICKING, "received /pick_target_local")
        else:
            self.get_logger().warn(
                f"Ignoring /pick_target_local in state={self.state.value}"
            )

    def _on_successful_pick(self, msg: Bool):
        self.event_counts["successful_pick"] += 1
        if self.state != SMState.PICKING:
            self.get_logger().warn(
                f"Ignoring /successful_pick={msg.data} in state={self.state.value}"
            )
            return
        if bool(msg.data):
            self._transition(SMState.PLACE_NAVIGATION, "successful_pick=true")
        else:
            self.get_logger().warn("successful_pick=false; staying in picking and waiting")
        # if False: stay in picking (just wait)

    def _on_place_target_local(self, _msg: PoseStamped):
        self.event_counts["place_target_local"] += 1
        if self.state == SMState.PLACE_NAVIGATION:
            self._transition(SMState.PLACING, "received /place_target_local")
        else:
            self.get_logger().warn(
                f"Ignoring /place_target_local in state={self.state.value}"
            )

    def _on_placing_done(self, msg: Bool):
        self.event_counts["placing_done"] += 1
        if self.state == SMState.PLACING and bool(msg.data):
            self._transition(SMState.FINISHED, "placing_done=true")
        elif self.state != SMState.PLACING:
            self.get_logger().warn(
                f"Ignoring /placing_done={msg.data} in state={self.state.value}"
            )
        else:
            self.get_logger().warn("placing_done=false; staying in placing and waiting")

    # ---------------- Helpers ----------------

    def _maybe_finish_audio(self):
        if self.state == SMState.AUDIO_PROCESSING and self.pick_target_ok and self.place_target_ok:
            self._transition(
                SMState.PICK_NAVIGATION,
                "both /pick_target and /place_target received",
            )

    def _transition(self, new_state: SMState, reason: str):
        if new_state == self.state:
            return
        prev_state = self.state
        now_ns = self.get_clock().now().nanoseconds
        dwell_s = (now_ns - self.state_enter_ns) * 1e-9
        self.state = new_state
        self.transition_count += 1
        self.last_transition_reason = reason
        self.state_enter_ns = now_ns
        self._publish_state()  # immediate publish on transition
        self.get_logger().info(
            f"Transition #{self.transition_count}: {prev_state.value} -> {self.state.value} "
            f"after {dwell_s:.2f}s (reason: {reason})"
        )

    def _publish_state(self):
        out = String()
        out.data = self.state.value
        self.state_pub.publish(out)

    def _heartbeat_log(self):
        now_ns = self.get_clock().now().nanoseconds
        dwell_s = (now_ns - self.state_enter_ns) * 1e-9
        self.get_logger().info(
            f"[heartbeat] state={self.state.value} dwell={dwell_s:.1f}s "
            f"pick_ok={self.pick_target_ok} place_ok={self.place_target_ok} "
            f"events={self.event_counts} last_reason='{self.last_transition_reason}'"
        )


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
