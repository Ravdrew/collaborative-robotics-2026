#!/usr/bin/env python3
"""
YOLO object detection using RealSense RGB + aligned depth.

- Subscribes:
  /camera/color/image_raw
  /camera/aligned_depth_to_color/image_raw

- Publishes:
  /pick_target_local   (Pose)  -> best banana
  /place_target_local  (Pose)  -> best book

Behavior:
- Runs YOLO on RGB image
- Finds best detection among pick classes {"banana"}
- Finds best detection among place classes {"book"}
- Publishes both independently if both exist

Notes:
- Uses Ultralytics YOLO26s directly by model name, so no local model path is needed.
- For book detections, depth is the average over the INNER region of the bounding box.
"""

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Pose
from std_msgs.msg import Bool

from cv_bridge import CvBridge
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

        self.pick_target_pub = self.create_publisher(Pose, "/pick_target_local", 10)
        self.place_target_pub = self.create_publisher(Pose, "/place_target_local", 10)

        self.bridge = CvBridge()

        # Better pretrained model, loaded by name only.
        # Auto-downloads on first use.
        self.model = YOLO("yolo26s.pt")

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

        # For book depth averaging: use inner crop to avoid edge/background contamination.
        # Example: 0.2 means trim 20% from each side, leaving central 60% region.
        self.inner_bbox_margin_ratio = 0.2

        # Detection gating: only publish when enabled
        self.detection_enabled = True
        self.create_subscription(Bool, "/detection_enabled", self._on_detection_enabled, 10)

        self.get_logger().info("Fruit target node started (YOLO26s: banana + book).")

    def _on_detection_enabled(self, msg: Bool):
        self.detection_enabled = msg.data

    def image_callback(self, rgb_msg, depth_msg):
        if not self.detection_enabled:
            return

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

        if not results:
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
            self.get_logger().info("YOLO: detections found, but none were banana/book")
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

        # Use bbox center pixel for XY projection
        px = int((x1 + x2) * 0.5)
        py = int((y1 + y2) * 0.5)
        px = int(np.clip(px, 0, w - 1))
        py = int(np.clip(py, 0, h - 1))

        # Different depth logic by action
        if action == "place" and cls_name == "book":
            depth_m = self.get_depth_inner_bbox_average_meters(depth, x1, y1, x2, y2)
            if depth_m is None or depth_m <= 0.0:
                self.get_logger().info(f"{cls_name}: inner-bbox average depth invalid; skipping")
                return
        else:
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

        self.publish_target(action, (X, Y, Z), orientation)

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

    def get_depth_inner_bbox_average_meters(self, depth_img, x1, y1, x2, y2):
        """
        Average depth over the INNER region of the bounding box.
        This reduces contamination from bbox edges and background.
        Intended specifically for book detections.
        """
        h, w = depth_img.shape[:2]

        x1 = float(np.clip(x1, 0, w - 1))
        x2 = float(np.clip(x2, 0, w - 1))
        y1 = float(np.clip(y1, 0, h - 1))
        y2 = float(np.clip(y2, 0, h - 1))

        if x2 <= x1 or y2 <= y1:
            return None

        bw = x2 - x1
        bh = y2 - y1
        mx = self.inner_bbox_margin_ratio * bw
        my = self.inner_bbox_margin_ratio * bh

        ix1 = int(np.clip(np.floor(x1 + mx), 0, w - 1))
        ix2 = int(np.clip(np.ceil(x2 - mx), 0, w - 1))
        iy1 = int(np.clip(np.floor(y1 + my), 0, h - 1))
        iy2 = int(np.clip(np.ceil(y2 - my), 0, h - 1))

        # Fallback if inner crop collapses
        if ix2 <= ix1 or iy2 <= iy1:
            ix1 = int(np.clip(np.floor(x1), 0, w - 1))
            ix2 = int(np.clip(np.ceil(x2), 0, w - 1))
            iy1 = int(np.clip(np.floor(y1), 0, h - 1))
            iy2 = int(np.clip(np.ceil(y2), 0, h - 1))

            if ix2 <= ix1 or iy2 <= iy1:
                return None

        region = depth_img[iy1:iy2 + 1, ix1:ix2 + 1].astype(np.float32).reshape(-1)
        region = region[np.isfinite(region)]
        region = region[region > 0]

        if region.size == 0:
            return None

        avg = float(np.mean(region))
        if avg > 10.0:
            avg *= 0.001

        return avg

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

    def publish_target(self, action, point_xyz, orientation):
        pose_msg = Pose()
        pose_msg.position.x = float(point_xyz[0])
        pose_msg.position.y = float(point_xyz[1])
        pose_msg.position.z = float(point_xyz[2])
        pose_msg.orientation.w = float(orientation)
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
