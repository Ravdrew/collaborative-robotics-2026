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
from typing import Optional

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

        # ---- State ----
        self.state: SMState = SMState.AUDIO_PROCESSING
        self.pick_target_ok = False
        self.place_target_ok = False

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

        # Publish initial state immediately
        self._publish_state()
        self.get_logger().info(f"Started in state: {self.state.value}")

    # ---------------- Callbacks ----------------

    def _on_pick_target(self, msg: String):
        if msg.data.strip():
            self.pick_target_ok = True
            self._maybe_finish_audio()

    def _on_place_target(self, msg: String):
        if msg.data.strip():
            self.place_target_ok = True
            self._maybe_finish_audio()

    def _on_pick_target_local(self, _msg: PointStamped):
        if self.state == SMState.PICK_NAVIGATION:
            self._transition(SMState.PICKING)

    def _on_successful_pick(self, msg: Bool):
        if self.state != SMState.PICKING:
            return
        if bool(msg.data):
            self._transition(SMState.PLACE_NAVIGATION)
        # if False: stay in picking (just wait)

    def _on_place_target_local(self, _msg: PoseStamped):
        if self.state == SMState.PLACE_NAVIGATION:
            self._transition(SMState.PLACING)

    def _on_placing_done(self, msg: Bool):
        if self.state == SMState.PLACING and bool(msg.data):
            self._transition(SMState.FINISHED)

    # ---------------- Helpers ----------------

    def _maybe_finish_audio(self):
        if self.state == SMState.AUDIO_PROCESSING and self.pick_target_ok and self.place_target_ok:
            self._transition(SMState.PICK_NAVIGATION)

    def _transition(self, new_state: SMState):
        if new_state == self.state:
            return
        self.state = new_state
        self._publish_state()  # immediate publish on transition
        self.get_logger().info(f"Transitioned to: {self.state.value}")

    def _publish_state(self):
        out = String()
        out.data = self.state.value
        self.state_pub.publish(out)


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
