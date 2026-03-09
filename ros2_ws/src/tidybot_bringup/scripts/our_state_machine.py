#!/usr/bin/env python3
"""
Task-oriented ROS 2 state machine for navigation + pick/place orchestration.

State flow:
1) audio_processing:
   wait for non-empty /pick_target and /place_target.
2) pick_navigation:
   wait for a pick nav goal, publish /cmd_nav, then wait for /nav_success true.
3) picking:
   publish one-shot /fsm_pick_request and wait for /successful_pick true.
4) place_navigation:
   wait for a place nav goal, publish /cmd_nav, then wait for /nav_success true.
5) placing:
   publish one-shot /fsm_place_request and wait for /placing_done (or sim fallback).
6) finished:
   terminal state.
"""

from enum import Enum
from typing import Optional

import rclpy
from geometry_msgs.msg import PointStamped, Pose2D, PoseStamped
from rclpy.node import Node
from std_msgs.msg import Bool, String


class SMState(str, Enum):
    AUDIO_PROCESSING = "audio_processing"
    PICK_NAVIGATION = "pick_navigation"
    PICKING = "picking"
    PLACE_NAVIGATION = "place_navigation"
    PLACING = "placing"
    FINISHED = "finished"


class StateMachineNode(Node):
    # Initialize FSM state, topics, timers, and wiring.
    def __init__(self):
        super().__init__("state_machine_node")

        # Core state publication and logging parameters.
        self.declare_parameter("state_topic", "/state_machine")
        self.declare_parameter("state_pub_hz", 5.0)
        self.declare_parameter("heartbeat_log_s", 2.0)

        # Audio/intent gate input parameters.
        self.declare_parameter("pick_target_topic", "/pick_target")
        self.declare_parameter("place_target_topic", "/place_target")

        # Navigation and task completion input/output parameters.
        self.declare_parameter("cmd_nav_topic", "/cmd_nav")
        self.declare_parameter("nav_success_topic", "/nav_success")
        self.declare_parameter("successful_pick_topic", "/successful_pick")
        self.declare_parameter("placing_done_topic", "/placing_done")
        self.declare_parameter("placing_done_sim_topic", "/placing_done_sim")

        # Canonical and legacy navigation goal input parameters.
        self.declare_parameter("fsm_pick_nav_goal_topic", "/fsm_pick_nav_goal")
        self.declare_parameter("fsm_place_nav_goal_topic", "/fsm_place_nav_goal")
        self.declare_parameter("pick_target_global_topic", "/pick_target_global")
        self.declare_parameter("place_target_local_topic", "/place_target_local")

        # Pick/place handoff signaling parameters.
        self.declare_parameter("fsm_pick_request_topic", "/fsm_pick_request")
        self.declare_parameter("fsm_place_request_topic", "/fsm_place_request")

        # Sim-only placing timeout fallback parameters.
        self.declare_parameter("enable_placing_done_sim_fallback", False)
        self.declare_parameter("placing_done_sim_delay_s", 5.0)

        # ---- State ----
        self.state: SMState = SMState.AUDIO_PROCESSING
        self.pick_target_ok = False
        self.place_target_ok = False
        self.pending_nav_phase: Optional[str] = None
        self.last_cmd_nav: Optional[Pose2D] = None
        self.nav_goal_sent_ns: Optional[int] = None
        self.placing_done_sim_deadline_ns: Optional[int] = None

        self.enable_placing_done_sim_fallback = bool(
            self.get_parameter("enable_placing_done_sim_fallback").value
        )
        self.placing_done_sim_delay_s = float(
            self.get_parameter("placing_done_sim_delay_s").value
        )

        self.transition_count = 0
        self.last_transition_reason = "startup"
        self.state_enter_ns = self.get_clock().now().nanoseconds
        self.event_counts = {
            "pick_target": 0,
            "place_target": 0,
            "fsm_pick_nav_goal": 0,
            "fsm_place_nav_goal": 0,
            "pick_target_global": 0,
            "place_target_local": 0,
            "cmd_nav_sent": 0,
            "nav_success": 0,
            "successful_pick": 0,
            "fsm_pick_request_sent": 0,
            "fsm_place_request_sent": 0,
            "placing_done": 0,
            "placing_done_sim": 0,
        }

        # ---- Publishers ----
        self.state_pub = self.create_publisher(
            String, str(self.get_parameter("state_topic").value), 10
        )
        self.cmd_nav_pub = self.create_publisher(
            Pose2D, str(self.get_parameter("cmd_nav_topic").value), 10
        )
        self.fsm_pick_request_pub = self.create_publisher(
            Bool, str(self.get_parameter("fsm_pick_request_topic").value), 10
        )
        self.fsm_place_request_pub = self.create_publisher(
            Bool, str(self.get_parameter("fsm_place_request_topic").value), 10
        )

        # ---- Subscriptions ----
        self.create_subscription(
            String, str(self.get_parameter("pick_target_topic").value), self._on_pick_target, 10
        )
        self.create_subscription(
            String, str(self.get_parameter("place_target_topic").value), self._on_place_target, 10
        )

        self.create_subscription(
            Pose2D,
            str(self.get_parameter("fsm_pick_nav_goal_topic").value),
            self._on_fsm_pick_nav_goal,
            10,
        )
        self.create_subscription(
            Pose2D,
            str(self.get_parameter("fsm_place_nav_goal_topic").value),
            self._on_fsm_place_nav_goal,
            10,
        )

        self.create_subscription(
            PointStamped,
            str(self.get_parameter("pick_target_global_topic").value),
            self._on_pick_target_global,
            10,
        )
        self.create_subscription(
            PoseStamped,
            str(self.get_parameter("place_target_local_topic").value),
            self._on_place_target_local,
            10,
        )

        self.create_subscription(
            Bool, str(self.get_parameter("nav_success_topic").value), self._on_nav_success, 10
        )
        self.create_subscription(
            Bool,
            str(self.get_parameter("successful_pick_topic").value),
            self._on_successful_pick,
            10,
        )
        self.create_subscription(
            Bool, str(self.get_parameter("placing_done_topic").value), self._on_placing_done, 10
        )
        self.create_subscription(
            Bool,
            str(self.get_parameter("placing_done_sim_topic").value),
            self._on_placing_done_sim,
            10,
        )

        # ---- Timers ----
        state_hz = float(self.get_parameter("state_pub_hz").value)
        self.create_timer(1.0 / max(state_hz, 0.1), self._publish_state)

        heartbeat_period = float(self.get_parameter("heartbeat_log_s").value)
        self.create_timer(max(0.5, heartbeat_period), self._heartbeat_log)
        self.create_timer(0.2, self._placing_done_sim_fallback_tick)

        # Initial publish/logs
        self._publish_state()
        self.get_logger().info(f"Started in state: {self.state.value}")
        self.get_logger().info(
            "Waiting for /pick_target and /place_target before entering pick_navigation"
        )

    # ---------------- Callbacks ----------------

    # Process pick target string from audio pipeline.
    def _on_pick_target(self, msg: String):
        self.event_counts["pick_target"] += 1
        self.get_logger().info(f"Event /pick_target: '{msg.data}'")
        if msg.data.strip():
            self.pick_target_ok = True
            self._maybe_finish_audio()
        else:
            self.get_logger().warn("Received empty /pick_target; ignoring")

    # Process place target string from audio pipeline.
    def _on_place_target(self, msg: String):
        self.event_counts["place_target"] += 1
        self.get_logger().info(f"Event /place_target: '{msg.data}'")
        if msg.data.strip():
            self.place_target_ok = True
            self._maybe_finish_audio()
        else:
            self.get_logger().warn("Received empty /place_target; ignoring")

    # Accept canonical pick nav goal in pick_navigation.
    def _on_fsm_pick_nav_goal(self, msg: Pose2D):
        self.event_counts["fsm_pick_nav_goal"] += 1
        if self.state != SMState.PICK_NAVIGATION:
            self.get_logger().warn(
                f"Ignoring /fsm_pick_nav_goal in state={self.state.value}"
            )
            return
        self._publish_cmd_nav(msg, phase="pick", reason="received /fsm_pick_nav_goal")

    # Accept canonical place nav goal in place_navigation.
    def _on_fsm_place_nav_goal(self, msg: Pose2D):
        self.event_counts["fsm_place_nav_goal"] += 1
        if self.state != SMState.PLACE_NAVIGATION:
            self.get_logger().warn(
                f"Ignoring /fsm_place_nav_goal in state={self.state.value}"
            )
            return
        self._publish_cmd_nav(msg, phase="place", reason="received /fsm_place_nav_goal")

    # Convert legacy global pick target into Pose2D goal.
    def _on_pick_target_global(self, msg: PointStamped):
        self.event_counts["pick_target_global"] += 1
        if self.state != SMState.PICK_NAVIGATION:
            self.get_logger().warn(
                f"Ignoring /pick_target_global in state={self.state.value}"
            )
            return
        goal = Pose2D(x=float(msg.point.x), y=float(msg.point.y), theta=0.0)
        self._publish_cmd_nav(goal, phase="pick", reason="converted /pick_target_global")

    # Convert legacy local place target into Pose2D goal.
    def _on_place_target_local(self, msg: PoseStamped):
        self.event_counts["place_target_local"] += 1
        if self.state != SMState.PLACE_NAVIGATION:
            self.get_logger().warn(
                f"Ignoring /place_target_local in state={self.state.value}"
            )
            return
        goal = Pose2D(
            x=float(msg.pose.position.x),
            y=float(msg.pose.position.y),
            theta=0.0,
        )
        self._publish_cmd_nav(goal, phase="place", reason="converted /place_target_local")

    # Transition nav states when navigator reports success.
    def _on_nav_success(self, msg: Bool):
        self.event_counts["nav_success"] += 1
        if not bool(msg.data):
            self.get_logger().warn("Received /nav_success=false; staying in current state")
            return

        if self.pending_nav_phase is None:
            self.get_logger().warn(
                "Received /nav_success=true but no pending_nav_phase is active"
            )
            return

        if self.pending_nav_phase == "pick" and self.state == SMState.PICK_NAVIGATION:
            self.pending_nav_phase = None
            self._transition(SMState.PICKING, "nav_success=true for pick phase")
            return

        if self.pending_nav_phase == "place" and self.state == SMState.PLACE_NAVIGATION:
            self.pending_nav_phase = None
            self._transition(SMState.PLACING, "nav_success=true for place phase")
            return

        self.get_logger().warn(
            "Received /nav_success=true but phase/state mismatch: "
            f"phase={self.pending_nav_phase} state={self.state.value}"
        )

    # Transition from picking after manipulator success.
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

    # Complete placing when real placing_done arrives.
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

    # Complete placing when sim helper topic arrives.
    def _on_placing_done_sim(self, msg: Bool):
        self.event_counts["placing_done_sim"] += 1
        if self.state != SMState.PLACING:
            self.get_logger().warn(
                f"Ignoring /placing_done_sim={msg.data} in state={self.state.value}"
            )
            return

        if bool(msg.data):
            self._transition(SMState.FINISHED, "placing_done_sim=true")
        else:
            self.get_logger().warn("placing_done_sim=false; staying in placing and waiting")

    # ---------------- Helpers ----------------

    # Leave audio state once both text targets exist.
    def _maybe_finish_audio(self):
        if self.state == SMState.AUDIO_PROCESSING and self.pick_target_ok and self.place_target_ok:
            self._transition(
                SMState.PICK_NAVIGATION,
                "both /pick_target and /place_target received",
            )

    # Publish cmd_nav and mark active nav phase.
    def _publish_cmd_nav(self, goal: Pose2D, phase: str, reason: str):
        out = Pose2D()
        out.x = float(goal.x)
        out.y = float(goal.y)
        out.theta = float(goal.theta)
        self.cmd_nav_pub.publish(out)

        self.pending_nav_phase = phase
        self.last_cmd_nav = out
        self.nav_goal_sent_ns = self.get_clock().now().nanoseconds
        self.event_counts["cmd_nav_sent"] += 1

        self.get_logger().info(
            f"Published /cmd_nav phase={phase} goal=({out.x:.2f}, {out.y:.2f}, {out.theta:.2f}) "
            f"(reason: {reason})"
        )

    # Send one-shot pick handoff request.
    def _publish_pick_request_once(self):
        self.fsm_pick_request_pub.publish(Bool(data=True))
        self.event_counts["fsm_pick_request_sent"] += 1
        self.get_logger().info("Published /fsm_pick_request=true (enter PICKING)")

    # Send one-shot place handoff request.
    def _publish_place_request_once(self):
        self.fsm_place_request_pub.publish(Bool(data=True))
        self.event_counts["fsm_place_request_sent"] += 1
        self.get_logger().info("Published /fsm_place_request=true (enter PLACING)")

    # Run one-shot actions on state entry.
    def _on_state_entry(self):
        if self.state in (SMState.PICK_NAVIGATION, SMState.PLACE_NAVIGATION):
            self.pending_nav_phase = None
            self.nav_goal_sent_ns = None

        if self.state == SMState.PICKING:
            self._publish_pick_request_once()

        if self.state == SMState.PLACING:
            self._publish_place_request_once()
            if self.enable_placing_done_sim_fallback:
                self.placing_done_sim_deadline_ns = (
                    self.get_clock().now().nanoseconds
                    + int(self.placing_done_sim_delay_s * 1e9)
                )
                self.get_logger().warn(
                    "Sim-only placing fallback is enabled; auto-finish timer armed "
                    f"for {self.placing_done_sim_delay_s:.1f}s"
                )
            else:
                self.placing_done_sim_deadline_ns = None
        else:
            self.placing_done_sim_deadline_ns = None

    # Trigger sim fallback timeout while in placing.
    def _placing_done_sim_fallback_tick(self):
        if not self.enable_placing_done_sim_fallback:
            return
        if self.state != SMState.PLACING:
            return
        if self.placing_done_sim_deadline_ns is None:
            return

        now_ns = self.get_clock().now().nanoseconds
        if now_ns >= self.placing_done_sim_deadline_ns:
            self._transition(
                SMState.FINISHED,
                f"placing_done_sim_fallback timeout ({self.placing_done_sim_delay_s:.1f}s)",
            )

    # Apply and log state transitions.
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
        self._on_state_entry()
        self._publish_state()

        self.get_logger().info(
            f"Transition #{self.transition_count}: {prev_state.value} -> {self.state.value} "
            f"after {dwell_s:.2f}s (reason: {reason})"
        )

    # Publish current FSM state text.
    def _publish_state(self):
        out = String()
        out.data = self.state.value
        self.state_pub.publish(out)

    # Emit periodic heartbeat for runtime debugging.
    def _heartbeat_log(self):
        now_ns = self.get_clock().now().nanoseconds
        dwell_s = (now_ns - self.state_enter_ns) * 1e-9

        nav_goal_age_s = (
            -1.0
            if self.nav_goal_sent_ns is None
            else (now_ns - self.nav_goal_sent_ns) * 1e-9
        )
        nav_goal_desc = (
            "none"
            if self.last_cmd_nav is None
            else f"({self.last_cmd_nav.x:.2f}, {self.last_cmd_nav.y:.2f}, {self.last_cmd_nav.theta:.2f})"
        )

        fallback_remaining_s = None
        if self.placing_done_sim_deadline_ns is not None:
            fallback_remaining_s = max(
                0.0, (self.placing_done_sim_deadline_ns - now_ns) * 1e-9
            )

        self.get_logger().info(
            f"[heartbeat] state={self.state.value} dwell={dwell_s:.1f}s "
            f"pick_ok={self.pick_target_ok} place_ok={self.place_target_ok} "
            f"pending_nav_phase={self.pending_nav_phase} nav_goal={nav_goal_desc} "
            f"nav_goal_age_s={nav_goal_age_s:.1f} fallback_enabled={self.enable_placing_done_sim_fallback} "
            f"fallback_remaining_s={fallback_remaining_s} events={self.event_counts} "
            f"last_reason='{self.last_transition_reason}'"
        )


# Start and spin state machine node.
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
