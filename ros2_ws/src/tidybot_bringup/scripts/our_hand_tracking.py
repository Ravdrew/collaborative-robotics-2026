#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped

from cv_bridge import CvBridge

import cv2
import numpy as np
import mediapipe as mp
import realsense as rs

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
            '/camera/depth/image_raw'
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

        # Convert ROS → OpenCV
        rgb = self.bridge.imgmsg_to_cv2(rgb_msg, 'bgr8')
        depth = self.bridge.imgmsg_to_cv2(depth_msg, 'passthrough')

        rgb_rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        results = self.hands.process(rgb_rgb)

        if not results.multi_hand_landmarks:
            return

        hand_landmarks = results.multi_hand_landmarks[0]

        h, w, _ = rgb.shape

        landmark_3d_points = []

        for lm in hand_landmarks.landmark:

            # Pixel coordinates
            px = int(lm.x * w)
            py = int(lm.y * h)

            # Depth in meters (RealSense usually mm → convert)
            depth_val = depth[py, px] * 0.001

            if depth_val == 0:
                continue

            # Back-project to 3D (camera frame)
            X, Y, Z = self.deproject(px, py, depth_val, w, h)

            landmark_3d_points.append([X, Y, Z])

        if len(landmark_3d_points) < 21:
            return

        landmark_3d_points = np.array(landmark_3d_points)

        # Palm indices
        palm_ids = [0, 5, 9, 13, 17]

        palm_points = landmark_3d_points[palm_ids]

        palm_center = np.mean(palm_points, axis=0)

        self.publish_target(palm_center, rgb_msg.header)

    # --------------------------------------------------

    def deproject(self, px, py, depth, w, h):
        """
        Simple pinhole back-projection.
        Replace intrinsics with CameraInfo in production.
        """

        # Deproject manually
        # fx = 600.0
        # fy = 600.0
        # cx = w / 2.0
        # cy = h / 2.0

        # X = (px - cx) * depth / fx
        # Y = (py - cy) * depth / fy
        # Z = depth

        # Deproject with RS
        point_3d = rs.rs2_deproject_pixel_to_point(
            intrinsics,
            [px, py],
            depth
        )
        X, Y, Z = point_3d

        return X, Y, Z

    # --------------------------------------------------

    def publish_target(self, point, header):

        msg = PointStamped()

        msg.header = header
        msg.header.frame_id = "camera_color_optical_frame"

        msg.point.x = float(point[0])
        msg.point.y = float(point[1])
        msg.point.z = float(point[2])

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
