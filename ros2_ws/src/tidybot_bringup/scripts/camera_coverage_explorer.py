#!/usr/bin/env python3
"""
Camera Coverage Explorer — replaces frontier exploration (explore_lite).

Instead of using lidar frontiers (which pick unreachable goals outside cage walls),
this node tracks which map cells the camera has actually observed and systematically
rotates + navigates to cover the arena.

States: IDLE → ROTATING → DWELL → SELECTING_VIEWPOINT → NAVIGATING → ROTATING ...

Subscribes:
    explore/resume (Bool) — True to start exploring, False to stop

Publishes:
    camera_coverage_grid (OccupancyGrid) — visualization of seen/unseen cells
    cmd_vel (Twist) — base rotation commands

Uses:
    navigate_to_pose (Nav2 action) — to move to new viewpoints
    TF (map → base_link) — to get robot pose
"""

import math

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
    SELECTING_VIEWPOINT = "selecting_viewpoint"
    NAVIGATING = "navigating"


class CameraCoverageExplorer(Node):
    def __init__(self):
        super().__init__("camera_coverage_explorer")

        # Parameters
        self.declare_parameter("grid_resolution", 0.2)
        self.declare_parameter("grid_size", 4.0)
        self.declare_parameter("camera_hfov_deg", 69.0)
        self.declare_parameter("camera_max_range", 2.5)
        self.declare_parameter("rotation_steps", 6)
        self.declare_parameter("rotation_speed", 0.3)
        self.declare_parameter("dwell_time", 1.5)
        self.declare_parameter("viewpoint_spacing", 0.5)
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("base_frame", "base_link")

        self.grid_resolution = self.get_parameter("grid_resolution").value
        self.grid_size = self.get_parameter("grid_size").value
        self.camera_hfov = math.radians(self.get_parameter("camera_hfov_deg").value)
        self.camera_max_range = self.get_parameter("camera_max_range").value
        self.rotation_steps = self.get_parameter("rotation_steps").value
        self.rotation_speed = self.get_parameter("rotation_speed").value
        self.dwell_time = self.get_parameter("dwell_time").value
        self.viewpoint_spacing = self.get_parameter("viewpoint_spacing").value
        self.map_frame = self.get_parameter("map_frame").value
        self.base_frame = self.get_parameter("base_frame").value

        # Coverage grid: 0 = unseen, 1 = seen
        self.grid_cells = int(self.grid_size / self.grid_resolution)
        self.coverage = np.zeros((self.grid_cells, self.grid_cells), dtype=np.uint8)
        # Grid origin: centered on map origin
        self.grid_origin_x = -self.grid_size / 2.0
        self.grid_origin_y = -self.grid_size / 2.0

        # State
        self.state = State.IDLE
        self.current_rotation_step = 0
        self.target_yaw = 0.0
        self.dwell_start_time = None
        self.nav_candidates = []  # sorted list of candidate viewpoints

        # TF
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, "cmd_vel", 10)
        self.grid_pub = self.create_publisher(OccupancyGrid, "camera_coverage_grid", 10)

        # Subscribers
        self.create_subscription(Bool, "explore/resume", self._on_explore_resume, 10)

        # Nav2 action client
        self.nav_client = ActionClient(self, NavigateToPose, "navigate_to_pose")
        self.nav_goal_handle = None

        # Main loop timer (10 Hz)
        self.create_timer(0.1, self._tick)

        # Publish coverage grid for RViz (1 Hz)
        self.create_timer(1.0, self._publish_coverage_grid)

        self.get_logger().info(
            f"Camera coverage explorer ready: {self.grid_cells}x{self.grid_cells} grid, "
            f"{self.rotation_steps} rotation steps"
        )

    def _on_explore_resume(self, msg: Bool):
        if msg.data:
            if self.state == State.IDLE:
                self.get_logger().info("Starting camera coverage exploration")
                self._start_rotation()
        else:
            if self.state != State.IDLE:
                self.get_logger().info("Stopping camera coverage exploration")
                self._cancel_nav_if_active()
                self._stop_base()
                self.state = State.IDLE

    def _get_robot_pose(self):
        """Returns (x, y, yaw) in map frame, or None if TF unavailable."""
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

    def _world_to_grid(self, wx, wy):
        """Convert world coords to grid indices."""
        gx = int((wx - self.grid_origin_x) / self.grid_resolution)
        gy = int((wy - self.grid_origin_y) / self.grid_resolution)
        return gx, gy

    def _grid_to_world(self, gx, gy):
        """Convert grid indices to world coords (cell center)."""
        wx = self.grid_origin_x + (gx + 0.5) * self.grid_resolution
        wy = self.grid_origin_y + (gy + 0.5) * self.grid_resolution
        return wx, wy

    def _mark_fov(self, robot_x, robot_y, robot_yaw):
        """Mark cells within camera FOV cone as seen."""
        half_fov = self.camera_hfov / 2.0
        max_range_sq = self.camera_max_range ** 2

        for gx in range(self.grid_cells):
            for gy in range(self.grid_cells):
                if self.coverage[gy, gx]:
                    continue
                cx, cy = self._grid_to_world(gx, gy)
                dx = cx - robot_x
                dy = cy - robot_y
                dist_sq = dx * dx + dy * dy
                if dist_sq > max_range_sq:
                    continue
                angle_to_cell = math.atan2(dy, dx)
                angle_diff = self._angle_diff(angle_to_cell, robot_yaw)
                if abs(angle_diff) <= half_fov:
                    self.coverage[gy, gx] = 1

    def _count_visible_unseen(self, vx, vy, heading):
        """Count unseen cells visible from a position at a given heading."""
        half_fov = self.camera_hfov / 2.0
        max_range_sq = self.camera_max_range ** 2
        count = 0
        for gx in range(self.grid_cells):
            for gy in range(self.grid_cells):
                if self.coverage[gy, gx]:
                    continue
                cx, cy = self._grid_to_world(gx, gy)
                dx = cx - vx
                dy = cy - vy
                dist_sq = dx * dx + dy * dy
                if dist_sq > max_range_sq:
                    continue
                angle_to_cell = math.atan2(dy, dx)
                angle_diff = self._angle_diff(angle_to_cell, heading)
                if abs(angle_diff) <= half_fov:
                    count += 1
        return count

    @staticmethod
    def _angle_diff(a, b):
        """Signed angle difference, normalized to [-pi, pi]."""
        d = a - b
        while d > math.pi:
            d -= 2.0 * math.pi
        while d < -math.pi:
            d += 2.0 * math.pi
        return d

    def _start_rotation(self):
        """Begin a full rotation sequence at the current position."""
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

    def _normalize_angle(self, a):
        while a > math.pi:
            a -= 2.0 * math.pi
        while a < -math.pi:
            a += 2.0 * math.pi
        return a

    def _stop_base(self):
        self.cmd_vel_pub.publish(Twist())

    def _tick(self):
        """Main state machine tick at 10 Hz."""
        if self.state == State.IDLE:
            return

        pose = self._get_robot_pose()
        if pose is None:
            return

        robot_x, robot_y, robot_yaw = pose

        if self.state == State.ROTATING:
            # Rotate toward target_yaw
            yaw_err = self._angle_diff(self.target_yaw, robot_yaw)
            if abs(yaw_err) < 0.08:  # ~4.5 degrees tolerance
                self._stop_base()
                # Mark coverage at this heading
                self._mark_fov(robot_x, robot_y, robot_yaw)
                # Enter dwell
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
                    # Full rotation done, select next viewpoint
                    self.get_logger().info("Full rotation complete, selecting next viewpoint")
                    self.state = State.SELECTING_VIEWPOINT
                else:
                    # Next rotation step
                    step_angle = 2.0 * math.pi / self.rotation_steps
                    self.target_yaw = self._normalize_angle(
                        self.target_yaw + step_angle
                    )
                    self.state = State.ROTATING
                    self.get_logger().info(
                        f"Rotating: step {self.current_rotation_step + 1}/{self.rotation_steps}"
                    )

        elif self.state == State.SELECTING_VIEWPOINT:
            self._select_and_navigate(robot_x, robot_y)

        elif self.state == State.NAVIGATING:
            # Waiting for Nav2 result callback
            pass

    def _select_and_navigate(self, robot_x, robot_y):
        """Score candidate viewpoints and navigate to the best one."""
        total_unseen = int(np.sum(self.coverage == 0))
        total_cells = self.grid_cells * self.grid_cells
        coverage_pct = 100.0 * (1.0 - total_unseen / total_cells)
        self.get_logger().info(
            f"Coverage: {coverage_pct:.1f}% ({total_cells - total_unseen}/{total_cells} cells seen)"
        )

        if total_unseen == 0:
            self.get_logger().info("Full coverage achieved! Returning to IDLE.")
            self.state = State.IDLE
            return

        # Generate candidate viewpoints on a grid within arena bounds
        candidates = []
        step = self.viewpoint_spacing
        # Use slightly inset bounds to avoid edge positions
        margin = self.grid_resolution * 2
        x_min = self.grid_origin_x + margin
        x_max = self.grid_origin_x + self.grid_size - margin
        y_min = self.grid_origin_y + margin
        y_max = self.grid_origin_y + self.grid_size - margin

        x = x_min
        while x <= x_max:
            y = y_min
            while y <= y_max:
                # Skip if too close to current position (we already rotated here)
                dist = math.hypot(x - robot_x, y - robot_y)
                if dist < self.viewpoint_spacing * 0.5:
                    y += step
                    continue
                # Score: total unseen cells visible across all rotation headings
                score = 0
                for i in range(self.rotation_steps):
                    heading = i * 2.0 * math.pi / self.rotation_steps
                    score += self._count_visible_unseen(x, y, heading)
                if score > 0:
                    candidates.append((score, dist, x, y))
                y += step
            x += step

        if not candidates:
            self.get_logger().warn("No candidate viewpoints found. Returning to IDLE.")
            self.state = State.IDLE
            return

        # Sort by score descending, then distance ascending as tiebreaker
        candidates.sort(key=lambda c: (-c[0], c[1]))
        self.nav_candidates = candidates

        self._try_next_candidate()

    def _try_next_candidate(self):
        """Send nav goal to the next best candidate viewpoint."""
        if not self.nav_candidates:
            self.get_logger().warn("All candidates exhausted. Returning to IDLE.")
            self.state = State.IDLE
            return

        score, dist, x, y = self.nav_candidates.pop(0)
        self.get_logger().info(
            f"Navigating to viewpoint ({x:.2f}, {y:.2f}), "
            f"score={score}, dist={dist:.2f}m, "
            f"{len(self.nav_candidates)} candidates remaining"
        )

        goal = NavigateToPose.Goal()
        goal.pose = PoseStamped()
        goal.pose.header.frame_id = self.map_frame
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x = x
        goal.pose.pose.position.y = y
        # Face toward map center (rough heuristic for initial heading)
        yaw = math.atan2(-y, -x)
        goal.pose.pose.orientation.z = math.sin(yaw / 2.0)
        goal.pose.pose.orientation.w = math.cos(yaw / 2.0)

        self.state = State.NAVIGATING

        if not self.nav_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().warn("Nav2 action server not available")
            self._try_next_candidate()
            return

        future = self.nav_client.send_goal_async(goal)
        future.add_done_callback(self._nav_goal_response)

    def _nav_goal_response(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn("Nav goal rejected, trying next candidate")
            self._try_next_candidate()
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
            self.get_logger().warn(f"Navigation failed (status={status}), trying next candidate")
            if self.state == State.NAVIGATING:
                self._try_next_candidate()

    def _cancel_nav_if_active(self):
        if self.nav_goal_handle is not None:
            self.get_logger().info("Cancelling active nav goal")
            self.nav_goal_handle.cancel_goal_async()
            self.nav_goal_handle = None

    def _publish_coverage_grid(self):
        """Publish coverage as OccupancyGrid for RViz visualization."""
        msg = OccupancyGrid()
        msg.header.frame_id = self.map_frame
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.info.resolution = self.grid_resolution
        msg.info.width = self.grid_cells
        msg.info.height = self.grid_cells
        msg.info.origin.position.x = self.grid_origin_x
        msg.info.origin.position.y = self.grid_origin_y
        msg.info.origin.orientation.w = 1.0

        # OccupancyGrid values: 0 = free (seen), 100 = occupied (unseen), -1 = unknown
        data = np.where(self.coverage == 1, 0, 100).astype(np.int8)
        msg.data = data.flatten().tolist()
        self.grid_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = CameraCoverageExplorer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
