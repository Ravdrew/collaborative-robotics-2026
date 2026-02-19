#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped

from cv_bridge import CvBridge

import cv2
import numpy as np
import mediapipe as mp
import pyrealsense2 as rs

from message_filters import Subscriber, ApproximateTimeSynchronizer


class HandPlaceTargetNode(Node):

    def __init__(self):
        super().__init__('hand_place_target_node')

        # Subscribers (RGB + aligned depth)
        self.rgb_sub = Subscriber(
            self, Image,
            '/camera/color/image_raw'
        )

        self.depth_sub = Subscriber(
            self, Image,
            '/camera/depth/image_raw' # changed from aligned_depth_to_color
        )

        self.sync = ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub],
            queue_size=10,
            slop=0.1
        )
        self.sync.registerCallback(self.image_callback)

        # Publisher
        self.target_pub = self.create_publisher(
            PointStamped,
            '/place_target_local',
            10
        )

        # CV bridge
        self.bridge = CvBridge()
        
        self.get_logger().info("Initializing MediaPipe Hands...")

        # MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )

        self.get_logger().info("Hand place target node started.")

    # --------------------------------------------------

    def image_callback(self, rgb_msg, depth_msg):

        # Callback entry
        self.get_logger().info("image_callback: received messages")
        self.get_logger().debug("image_callback called")

        # Convert ROS → OpenCV
        try:
            rgb = self.bridge.imgmsg_to_cv2(rgb_msg, 'bgr8')
            depth = self.bridge.imgmsg_to_cv2(depth_msg, 'passthrough')
        except Exception as e:
            self.get_logger().info(f"Failed to convert images: {e}")
            self.get_logger().error(f"Failed to convert images: {e}")
            return

        rgb_rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        results = self.hands.process(rgb_rgb)
        self.get_logger().info(f"MediaPipe results: {bool(results and results.multi_hand_landmarks)}")

        if not results or not results.multi_hand_landmarks:
            self.get_logger().info("No hands detected")
            return

        hand_landmarks = results.multi_hand_landmarks[0]

        h, w, _ = rgb.shape

        landmark_3d_points = []
        skipped_depth_zero = 0
        skipped_oob = 0

        for lm in hand_landmarks.landmark:

            # Pixel coordinates
            px = int(lm.x * w)
            py = int(lm.y * h)

            # Check bounds
            if px < 0 or px >= w or py < 0 or py >= h:
                skipped_oob += 1
                self.get_logger().info(f"Landmark out of bounds: px={px}, py={py}, w={w}, h={h}")
                continue

            # Depth in meters (RealSense usually mm → convert)
            depth_raw = depth[py, px]
            try:
                depth_val = float(depth_raw) * 0.001
            except Exception as e:
                self.get_logger().info(f"Depth access error at ({py},{px}): {e}")
                continue

            if depth_val == 0:
                skipped_depth_zero += 1
                continue

            # Back-project to 3D (camera frame)
            try:
                X, Y, Z = self.deproject(px, py, depth_val, w, h)
            except Exception as e:
                self.get_logger().info(f"Deproject failed for px={px},py={py},d={depth_val}: {e}")
                continue

            landmark_3d_points.append([X, Y, Z])

        self.get_logger().info(f"Collected 3D landmarks: {len(landmark_3d_points)} (skipped_depth_zero={skipped_depth_zero}, skipped_oob={skipped_oob})")
        if len(landmark_3d_points) < 21:
            self.get_logger().info("Not enough valid 3D landmarks to publish target")
            return

        landmark_3d_points = np.array(landmark_3d_points)

        # Palm indices
        palm_ids = [0, 5, 9, 13, 17]

        palm_points = landmark_3d_points[palm_ids]

        palm_center = np.mean(palm_points, axis=0)

        self.get_logger().info(f"Palm center (camera frame): {palm_center}")
        self.publish_target(palm_center, rgb_msg.header)

    # --------------------------------------------------

    def deproject(self, px, py, depth, w, h):
        """
        Simple pinhole back-projection.
        Replace intrinsics with CameraInfo in production.
        """

        # Deproject manually
        fx = 600.0
        fy = 600.0
        cx = w / 2.0
        cy = h / 2.0

        X = (px - cx) * depth / fx
        Y = (py - cy) * depth / fy
        Z = depth

        # Debug small sanity check
        if not np.isfinite(X) or not np.isfinite(Y) or not np.isfinite(Z):
            self.get_logger().info(f"Deproject produced non-finite: px={px},py={py},depth={depth},X={X},Y={Y},Z={Z}")

        # Deproject with RS
        # point_3d = rs.rs2_deproject_pixel_to_point(
        #     intrinsics,
        #     [px, py],
        #     depth
        # )
        # X, Y, Z = point_3d

        return X, Y, Z

    # --------------------------------------------------

    def publish_target(self, point, header):

        msg = PointStamped()

        msg.header = header
        msg.header.frame_id = "camera_color_optical_frame"

        msg.point.x = float(point[0])
        msg.point.y = float(point[1])
        msg.point.z = float(point[2])

        self.get_logger().info(f"Publishing target: x={msg.point.x:.3f}, y={msg.point.y:.3f}, z={msg.point.z:.3f}")
        self.target_pub.publish(msg)


# ------------------------------------------------------

def main(args=None):

    rclpy.init(args=args)

    node = HandPlaceTargetNode()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()