#!/usr/bin/env python3
"""
YOLO object detection using RealSense RGB + aligned depth.

- Subscribes:
  /camera/color/image_raw
  /camera/aligned_depth_to_color/image_raw

- Publishes:
  /pick_target_local   (Pose)  -> best apple/banana
  /place_target_local  (Pose)  -> best book

Behavior:
- Runs YOLO on RGB image
- Finds best detection among pick classes {"apple", "banana"}
- Finds best detection among place classes {"book"}
- Publishes both independently if both exist
"""

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped, Pose
from std_msgs.msg import Bool

from cv_bridge import CvBridge
import cv2
import numpy as np

from message_filters import Subscriber, ApproximateTimeSynchronizer
from ultralytics import YOLO


class FruitTargetNode(Node):
    def __init__(self):
        super().__init__("fruit_target_node")

        self.rgb_sub = Subscriber(self, Image, "/camera/color/image_raw")
        self.depth_sub = Subscriber(self, Image, "/camera/aligned_depth_to_color/image_raw")

        self.sync = ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub],
            queue_size=10,
            slop=0.1
        )
        self.sync.registerCallback(self.image_callback)

        self.target_pub = self.create_publisher(PointStamped, "/fruit_target_local", 10)
        self.pick_target_pub = self.create_publisher(Pose, "/pick_target_local", 10)
        self.place_target_pub = self.create_publisher(Pose, "/place_target_local", 10)

        self.bridge = CvBridge()
        self.model = YOLO("yolov8n.pt")

        self.target_pick_classes = {"banana"}
        self.target_place_classes = {"book"}

        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            "/camera/color/camera_info",
            self.camera_info_callback,
            10
        )
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None

        self.depth_window_radius = 2

        # Detection gating: only publish when enabled (robot stationary)
        self.detection_enabled = True
        self.detection_cooldown = False  # brief delay after re-enabling
        self.create_subscription(Bool, "/detection_enabled", self._on_detection_enabled, 10)

        self.get_logger().info("Fruit target node started (YOLO: apple/banana + book).")

    def _on_detection_enabled(self, msg: Bool):
        was_disabled = not self.detection_enabled
        self.detection_enabled = msg.data
        if msg.data and was_disabled:
            # Re-create the sync to flush stale buffered images
            self.get_logger().info("Detection re-enabled — flushing buffer, cooldown 1s")
            self.sync = ApproximateTimeSynchronizer(
                [self.rgb_sub, self.depth_sub],
                queue_size=10,
                slop=0.1
            )
            self.sync.registerCallback(self.image_callback)
            # Brief cooldown so fresh images fill the pipeline
            self.detection_cooldown = True
            self.create_timer(1.0, self._end_cooldown)

    def _end_cooldown(self):
        self.detection_cooldown = False
        self.get_logger().info("Detection cooldown ended — publishing detections")

    def image_callback(self, rgb_msg, depth_msg):
        if not self.detection_enabled or self.detection_cooldown:
            return
        self.get_logger().debug("image_callback: received messages")

        try:
            rgb = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
            depth = self.bridge.imgmsg_to_cv2(depth_msg, "passthrough")
        except Exception as e:
            self.get_logger().error(f"Failed to convert images: {e}")
            return

        h, w, _ = rgb.shape

        try:
            results = self.model.predict(
                source=rgb,
                verbose=False,
                conf=0.4,
                iou=0.5
            )
        except Exception as e:
            self.get_logger().error(f"YOLO inference failed: {e}")
            return

        if not results or len(results) == 0:
            self.get_logger().info("YOLO: no results returned")
            return

        r0 = results[0]
        if r0.boxes is None or len(r0.boxes) == 0:
            self.get_logger().info("YOLO: no detections")
            return

        names = self.model.names
        best_pick = None
        best_place = None

        for b in r0.boxes:
            cls_id = int(b.cls.item())
            conf = float(b.conf.item())
            cls_name = names[cls_id] if isinstance(names, (list, tuple)) else names.get(cls_id, str(cls_id))
            x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().tolist()

            if cls_name in self.target_pick_classes:
                if best_pick is None or conf > best_pick[0]:
                    best_pick = (conf, cls_name, (x1, y1, x2, y2))

            elif cls_name in self.target_place_classes:
                if best_place is None or conf > best_place[0]:
                    best_place = (conf, cls_name, (x1, y1, x2, y2))

        if best_pick is None and best_place is None:
            self.get_logger().info("YOLO: detections found, but none were apple/banana/book")
            return

        if best_pick is not None:
            self.process_and_publish_detection(
                rgb=rgb,
                depth=depth,
                detection=best_pick,
                action="pick",
                header=rgb_msg.header
            )

        if best_place is not None:
            self.process_and_publish_detection(
                rgb=rgb,
                depth=depth,
                detection=best_place,
                action="place",
                header=rgb_msg.header
            )

    def process_and_publish_detection(self, rgb, depth, detection, action, header):
        conf, cls_name, (x1, y1, x2, y2) = detection
        h, w, _ = rgb.shape

        width = x2 - x1
        height = y2 - y1
        orientation = 1.0 if width < height else -1.0

        px = int((x1 + x2) * 0.5)
        py = int((y1 + y2) * 0.5)

        px = int(np.clip(px, 0, w - 1))
        py = int(np.clip(py, 0, h - 1))

        depth_m = self.get_depth_median_meters(depth, px, py)
        if depth_m is None or depth_m <= 0.0:
            self.get_logger().info(f"{cls_name}: depth invalid at center ({px},{py}); skipping")
            return

        try:
            X, Y, Z = self.deproject(px, py, depth_m, w, h)
        except Exception as e:
            self.get_logger().error(f"Deproject failed for {cls_name}: {e}")
            return

        self.get_logger().info(
            f"{action.upper()} target: {cls_name} conf={conf:.2f} "
            f"px=({px},{py}) depth={depth_m:.3f}m -> "
            f"XYZ=({X:.3f},{Y:.3f},{Z:.3f})"
        )

        self.publish_target(action, (X, Y, Z), orientation, header)

    def get_depth_median_meters(self, depth_img, px, py):
        r = self.depth_window_radius
        h, w = depth_img.shape[:2]

        x0 = max(px - r, 0)
        x1 = min(px + r, w - 1)
        y0 = max(py - r, 0)
        y1 = min(py + r, h - 1)

        window = depth_img[y0:y1 + 1, x0:x1 + 1].astype(np.float32).reshape(-1)
        window = window[np.isfinite(window)]
        window = window[window > 0]

        if window.size == 0:
            return None

        med = float(np.median(window))
        if med > 10.0:
            med *= 0.001

        return med

    def camera_info_callback(self, msg: CameraInfo):
        self.fx = msg.k[0]
        self.fy = msg.k[4]
        self.cx = msg.k[2]
        self.cy = msg.k[5]
        self.get_logger().info(
            f"Camera intrinsics received: fx={self.fx:.1f} fy={self.fy:.1f} "
            f"cx={self.cx:.1f} cy={self.cy:.1f}",
            once=True
        )

    def deproject(self, px, py, depth, w, h):
        if self.fx is not None:
            fx = self.fx
            fy = self.fy
            cx = self.cx
            cy = self.cy
        else:
            self.get_logger().warn("Camera intrinsics not yet received, using fallback values")
            fx = 600.0
            fy = 600.0
            cx = w / 2.0
            cy = h / 2.0

        X = (px - cx) * depth / fx
        Y = (py - cy) * depth / fy
        Z = depth

        if not np.isfinite(X) or not np.isfinite(Y) or not np.isfinite(Z):
            raise ValueError(f"Non-finite deproject: X={X}, Y={Y}, Z={Z}")

        return X, Y, Z

    def publish_target(self, action, point_xyz, orientation, header):
        pose_msg = Pose()
        pose_msg.position.x = float(point_xyz[0])
        pose_msg.position.y = float(point_xyz[1])
        pose_msg.position.z = float(point_xyz[2])
        pose_msg.orientation.w = orientation
        pose_msg.orientation.x = 0.0
        pose_msg.orientation.y = 0.0
        pose_msg.orientation.z = 0.0

        if action == "pick":
            self.pick_target_pub.publish(pose_msg)
        elif action == "place":
            self.place_target_pub.publish(pose_msg)


def main(args=None):
    rclpy.init(args=args)
    node = FruitTargetNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()