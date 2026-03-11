#!/usr/bin/env python3
"""
Simple Rotate-and-Navigate Explorer — replaces frontier exploration (explore_lite).

Picks random free points from the SLAM map, navigates there, does a full 360°
rotation in slices (with dwell for YOLO detection), then picks another point.
Repeats until stopped by the state machine.

Subscribes:
    explore/resume (Bool) — True to start exploring, False to stop
    map (OccupancyGrid) — SLAM map to sample free cells from

Publishes:
    cmd_vel (Twist) — base rotation commands

Uses:
    navigate_to_pose (Nav2 action) — to move to viewpoints
    TF (map → base_footprint) — to get robot pose
"""

import math
import random

import numpy as np
import rclpy
import tf2_ros
from enum import Enum
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import OccupancyGrid
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from rclpy.node import Node
from std_msgs.msg import Bool


class State(Enum):
    IDLE = "idle"
    ROTATING = "rotating"
    DWELL = "dwell"
    NAVIGATING = "navigating"


class SimpleExplorer(Node):
    def __init__(self):
        super().__init__("camera_coverage_explorer")

        # Parameters
        self.declare_parameter("rotation_steps", 12)
        self.declare_parameter("rotation_speed", 0.15)
        self.declare_parameter("dwell_time", 1.5)
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("base_frame", "base_footprint")
        self.declare_parameter("min_dist_from_robot", 0.5)
        self.declare_parameter("max_sample_attempts", 50)
        self.declare_parameter("obstacle_margin_cells", 3)

        self.rotation_steps = self.get_parameter("rotation_steps").value
        self.rotation_speed = self.get_parameter("rotation_speed").value
        self.dwell_time = self.get_parameter("dwell_time").value
        self.map_frame = self.get_parameter("map_frame").value
        self.base_frame = self.get_parameter("base_frame").value
        self.min_dist_from_robot = self.get_parameter("min_dist_from_robot").value
        self.max_sample_attempts = self.get_parameter("max_sample_attempts").value
        self.obstacle_margin_cells = self.get_parameter("obstacle_margin_cells").value

        # SLAM map
        self.slam_map_data = None
        self.slam_map_info = None

        # State
        self.state = State.IDLE
        self.current_rotation_step = 0
        self.target_yaw = 0.0
        self.dwell_start_time = None

        # TF
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, "cmd_vel", 10)

        # Subscribers
        self.create_subscription(Bool, "explore/resume", self._on_explore_resume, 10)
        self.create_subscription(OccupancyGrid, "map", self._on_slam_map, 10)

        # Nav2 action client
        self.nav_client = ActionClient(self, NavigateToPose, "navigate_to_pose")
        self.nav_goal_handle = None

        # Main loop timer (10 Hz)
        self.create_timer(0.1, self._tick)

        self.get_logger().info(
            f"Simple explorer ready: {self.rotation_steps} rotation steps, "
            f"{self.dwell_time}s dwell"
        )

    def _on_slam_map(self, msg: OccupancyGrid):
        self.slam_map_info = msg.info
        self.slam_map_data = np.array(msg.data, dtype=np.int8).reshape(
            (msg.info.height, msg.info.width)
        )

    def _on_explore_resume(self, msg: Bool):
        if msg.data:
            if self.state == State.IDLE:
                self.get_logger().info("Starting exploration: rotate first, then navigate")
                self._start_rotation()
        else:
            if self.state != State.IDLE:
                self.get_logger().info("Stopping exploration")
                self._cancel_nav_if_active()
                self._stop_base()
                self.state = State.IDLE

    def _get_robot_pose(self):
        try:
            t = self.tf_buffer.lookup_transform(
                self.map_frame, self.base_frame, rclpy.time.Time()
            )
            x = t.transform.translation.x
            y = t.transform.translation.y
            q = t.transform.rotation
            yaw = math.atan2(
                2.0 * (q.w * q.z + q.x * q.y),
                1.0 - 2.0 * (q.y * q.y + q.z * q.z),
            )
            return x, y, yaw
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ):
            return None

    def _sample_free_point(self, robot_x, robot_y):
        """Sample a random free point from the SLAM map, away from obstacles."""
        if self.slam_map_data is None or self.slam_map_info is None:
            self.get_logger().warn("No SLAM map available yet")
            return None

        info = self.slam_map_info
        margin = self.obstacle_margin_cells

        # Find all free cells (value == 0) that are away from obstacles
        free_mask = self.slam_map_data == 0

        # Erode free space by margin to avoid cells near obstacles
        if margin > 0:
            from scipy.ndimage import binary_erosion
            struct = np.ones((2 * margin + 1, 2 * margin + 1))
            free_mask = binary_erosion(free_mask, structure=struct)

        free_cells = np.argwhere(free_mask)
        if len(free_cells) == 0:
            self.get_logger().warn("No free cells found in SLAM map")
            return None

        for _ in range(self.max_sample_attempts):
            idx = random.randint(0, len(free_cells) - 1)
            my, mx = free_cells[idx]
            wx = info.origin.position.x + (mx + 0.5) * info.resolution
            wy = info.origin.position.y + (my + 0.5) * info.resolution
            dist = math.hypot(wx - robot_x, wy - robot_y)
            if dist >= self.min_dist_from_robot:
                return wx, wy

        # Fallback: just pick any free cell
        idx = random.randint(0, len(free_cells) - 1)
        my, mx = free_cells[idx]
        wx = info.origin.position.x + (mx + 0.5) * info.resolution
        wy = info.origin.position.y + (my + 0.5) * info.resolution
        return wx, wy

    @staticmethod
    def _angle_diff(a, b):
        d = a - b
        while d > math.pi:
            d -= 2.0 * math.pi
        while d < -math.pi:
            d += 2.0 * math.pi
        return d

    def _normalize_angle(self, a):
        while a > math.pi:
            a -= 2.0 * math.pi
        while a < -math.pi:
            a += 2.0 * math.pi
        return a

    def _start_rotation(self):
        pose = self._get_robot_pose()
        if pose is None:
            self.get_logger().warn("No TF available, retrying next tick")
            return
        _, _, yaw = pose
        self.current_rotation_step = 0
        step_angle = 2.0 * math.pi / self.rotation_steps
        self.target_yaw = self._normalize_angle(yaw + step_angle)
        self.state = State.ROTATING
        self.get_logger().info(f"Rotating: step 1/{self.rotation_steps}")

    def _stop_base(self):
        self.cmd_vel_pub.publish(Twist())

    def _tick(self):
        if self.state == State.IDLE:
            return

        pose = self._get_robot_pose()
        if pose is None:
            return

        robot_x, robot_y, robot_yaw = pose

        if self.state == State.ROTATING:
            yaw_err = self._angle_diff(self.target_yaw, robot_yaw)
            if abs(yaw_err) < 0.08:
                self._stop_base()
                self.state = State.DWELL
                self.dwell_start_time = self.get_clock().now()
                self.get_logger().info(
                    f"Dwelling at step {self.current_rotation_step + 1}/{self.rotation_steps}"
                )
            else:
                cmd = Twist()
                cmd.angular.z = self.rotation_speed if yaw_err > 0 else -self.rotation_speed
                self.cmd_vel_pub.publish(cmd)

        elif self.state == State.DWELL:
            elapsed = (self.get_clock().now() - self.dwell_start_time).nanoseconds / 1e9
            if elapsed >= self.dwell_time:
                self.current_rotation_step += 1
                if self.current_rotation_step >= self.rotation_steps:
                    self.get_logger().info("Full rotation complete, picking next point")
                    self._pick_and_navigate(robot_x, robot_y)
                else:
                    step_angle = 2.0 * math.pi / self.rotation_steps
                    self.target_yaw = self._normalize_angle(
                        self.target_yaw + step_angle
                    )
                    self.state = State.ROTATING
                    self.get_logger().info(
                        f"Rotating: step {self.current_rotation_step + 1}/{self.rotation_steps}"
                    )

        elif self.state == State.NAVIGATING:
            pass

    def _pick_and_navigate(self, robot_x, robot_y):
        point = self._sample_free_point(robot_x, robot_y)
        if point is None:
            self.get_logger().warn("Could not sample a point, retrying rotation")
            self._start_rotation()
            return

        x, y = point
        self.get_logger().info(f"Navigating to random free point ({x:.2f}, {y:.2f})")

        goal = NavigateToPose.Goal()
        goal.pose = PoseStamped()
        goal.pose.header.frame_id = self.map_frame
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x = x
        goal.pose.pose.position.y = y
        yaw = math.atan2(y - robot_y, x - robot_x)
        goal.pose.pose.orientation.z = math.sin(yaw / 2.0)
        goal.pose.pose.orientation.w = math.cos(yaw / 2.0)

        self.state = State.NAVIGATING

        if not self.nav_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().warn("Nav2 action server not available, picking another point")
            self._pick_and_navigate(robot_x, robot_y)
            return

        future = self.nav_client.send_goal_async(goal)
        future.add_done_callback(self._nav_goal_response)

    def _nav_goal_response(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn("Nav goal rejected, picking another point")
            self._retry_navigation()
            return
        self.nav_goal_handle = goal_handle
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._nav_result)

    def _nav_result(self, future):
        result = future.result()
        status = result.status
        if status == 4:  # STATUS_SUCCEEDED
            self.get_logger().info("Navigation succeeded, starting rotation")
            if self.state == State.NAVIGATING:
                self._start_rotation()
        else:
            self.get_logger().warn(f"Navigation failed (status={status}), picking another point")
            if self.state == State.NAVIGATING:
                self._retry_navigation()

    def _retry_navigation(self):
        pose = self._get_robot_pose()
        if pose is None:
            self.get_logger().warn("No TF for retry, going back to rotation")
            self._start_rotation()
            return
        self._pick_and_navigate(pose[0], pose[1])

    def _cancel_nav_if_active(self):
        if self.nav_goal_handle is not None:
            self.get_logger().info("Cancelling active nav goal")
            self.nav_goal_handle.cancel_goal_async()
            self.nav_goal_handle = None


def main(args=None):
    rclpy.init(args=args)
    node = SimpleExplorer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
