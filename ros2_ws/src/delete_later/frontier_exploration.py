#!/usr/bin/env python3
# Frontier exploration (tidybot-adapted)

import rclpy  
import scipy
from rclpy.node import Node     
from asl_tb3_lib.navigation import BaseNavigator, TrajectoryPlan
from asl_tb3_lib.math_utils import wrap_angle
from asl_tb3_lib.tf_utils import quaternion_to_yaw
from scipy.interpolate import splrep, splev
from geometry_msgs.msg import Pose2D
import numpy as np
from scipy import linalg
from std_msgs.msg import Bool, String
from asl_tb3_lib.grids import snap_to_grid, StochOccupancyGrid2D  #keep, this is robot-agnostic
from nav_msgs.msg import OccupancyGrid, Path
from scipy.signal import convolve2d

class FrontierExplorer(Node):
    def __init__(self):
        super().__init__("frontier_explorer")

        # subscribe to estimated pose published by the state estimator
        self.state_sub = self.create_subscription(Pose2D, "/odom_est_pose", self.state_callback, 10)
        self.map_sub = self.create_subscription(OccupancyGrid, "/map", self.map_callback, 10)
        self.nav_success_sub = self.create_subscription(Bool, "/nav_success", self.nav_success_callback, 10)
        self.cmd_nav_pub = self.create_publisher(Pose2D, "/cmd_nav", 10)
        self.detection_callback = self.create_subscription(Bool, "/detector_bool", self.detect_callback, 10)

        self.current_state = None
        self.occupancy = None
        self.nav_success = True
        self.window_size = 13
        self.image_detected = False
        # timer added
        self.stop_timer = None

    def state_callback(self, msg: Pose2D):
        self.current_state = np.array([msg.x, msg.y, msg.theta])

    def detect_callback(self, msg: Bool):
        if msg.data:
            if not self.image_detected:
                self.image_detected = True
                goal = Pose2D(x=float(self.current_state[0]), y=float(self.current_state[1]), theta=float(self.current_state[2]))
                self.cmd_nav_pub.publish(goal)
                # five second timer before resuming:
                self.stop_timer = self.create_timer(5.0, self.resume_after_stop)
            else:
                self.get_logger().info("Resuming exploration 2.")
                self.frontier(self.occupancy)
                self.image_detected = False
                self.nav_success = False
                
    def resume_after_stop(self):
        # restart exploring
        self.frontier(self.occupancy)
        self.nav_success = False
        self.image_detected = False
        # kill timer
        if self.stop_timer is not None:
            self.stop_timer.cancel()
            self.stop_timer = None
                
    def map_callback(self, msg: OccupancyGrid):
        self.occupancy = StochOccupancyGrid2D(
            resolution=msg.info.resolution,
            size_xy=np.array([msg.info.width, msg.info.height]),
            origin_xy=np.array([msg.info.origin.position.x, msg.info.origin.position.y]),
            window_size=self.window_size,
            probs=msg.data,
        )
        if self.image_detected:
            return
        if self.nav_success and not self.image_detected:
            self.frontier(self.occupancy)
            self.nav_success = False

    def nav_success_callback(self, msg: Bool):
        self.nav_success = True

    def frontier(self, occupancy: StochOccupancyGrid2D):
        if self.occupancy is None or self.current_state is None:
            return

        probs = occupancy.probs
        unknown_mask = (probs < 0)
        known_and_occupied_mask = (probs >= occupancy.thresh)
        known_and_unoccupied_mask = (probs < occupancy.thresh) & (probs >= 0)

        kernel = np.ones((self.window_size, self.window_size), dtype=int)
        unknown_count = convolve2d(unknown_mask.astype(int), kernel, mode='same', boundary='fill', fillvalue=0)
        known_and_occupied_count = convolve2d(known_and_occupied_mask.astype(int), kernel, mode='same', boundary='fill', fillvalue=0)
        known_and_unoccupied_count = convolve2d(known_and_unoccupied_mask.astype(int), kernel, mode='same', boundary='fill', fillvalue=0)

        total_cells_in_window = self.window_size * self.window_size
        percent_unknown = unknown_count / total_cells_in_window
        percent_unoccupied = known_and_unoccupied_count / total_cells_in_window

        frontier_mask = (percent_unknown >= 0.2) & (known_and_occupied_count == 0) & (percent_unoccupied >= 0.3)
        frontier_mask = frontier_mask & (probs >= 0)

        y, x = np.where(frontier_mask)
        grid_xy = np.vstack([x, y]).T
        frontier_states = occupancy.grid2state(grid_xy)

        if len(frontier_states) == 0:
            return
        
        distances = np.linalg.norm(frontier_states - np.array([self.current_state[0], self.current_state[1]]), axis=1)
        idx = np.argmin(distances)
        frontier = frontier_states[idx]

        goal = Pose2D(x=float(frontier[0]), y=float(frontier[1]), theta=0.0)
        self.cmd_nav_pub.publish(goal)
        self.get_logger().info(f"Publishing frontier goal to /cmd_nav: ({goal.x:.2f}, {goal.y:.2f})")

def main(argv=None):
    rclpy.init(args=argv)
    node = FrontierExplorer()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
