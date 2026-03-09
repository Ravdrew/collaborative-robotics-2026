#!/usr/bin/env python3
"""
YOLO object detection (apples + bananas) using RealSense RGB + aligned depth.

- Subscribes:
  /camera/color/image_raw
  /camera/aligned_depth_to_color/image_raw

- Publishes (PointStamped):
  /fruit_target_local

Behavior:
- Runs YOLO on the RGB image
- Filters detections to {apple, banana} (COCO class names)
- Picks the highest-confidence detection among those classes
- Uses aligned depth at the bbox center (median in a small window) to get Z
- Deprojects (using the same simple pinhole model as your example) to X,Y,Z
"""

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped

from cv_bridge import CvBridge
import cv2
import numpy as np

from message_filters import Subscriber, ApproximateTimeSynchronizer

# ---- YOLO: Ultralytics ----
# pip install ultralytics
from ultralytics import YOLO


class FruitTargetNode(Node):
    def __init__(self):
        super().__init__("fruit_target_node")

        # Subscribers (RGB + aligned depth)
        self.rgb_sub = Subscriber(self, Image, "/camera/color/image_raw")
        self.depth_sub = Subscriber(self, Image, "/camera/aligned_depth_to_color/image_raw")

        self.sync = ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub],
            queue_size=10,
            slop=0.1
        )
        self.sync.registerCallback(self.image_callback)

        # Publisher
        self.target_pub = self.create_publisher(PointStamped, "/fruit_target_local", 10)

        self.bridge = CvBridge()

        # Load YOLO model
        # Default: lightweight COCO model (has "banana" and "apple")
        # You can change to "yolov8m.pt" etc. if you want better accuracy.
        model_name = "yolov8n.pt"
        self.get_logger().info(f"Loading YOLO model: {model_name}")
        self.model = YOLO(model_name)

        # COCO class names we want
        self.target_classes = {"banana", "apple"}

        # Subscribe to camera intrinsics
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

        # Small window for robust depth read at bbox center
        self.depth_window_radius = 2  # 5x5 window

        self.get_logger().info("Fruit target node started (YOLO: apple/banana).")

    # --------------------------------------------------

    def image_callback(self, rgb_msg, depth_msg):
        self.get_logger().debug("image_callback: received messages")

        # Convert ROS → OpenCV
        try:
            rgb = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
            depth = self.bridge.imgmsg_to_cv2(depth_msg, "passthrough")
        except Exception as e:
            self.get_logger().error(f"Failed to convert images: {e}")
            return

        h, w, _ = rgb.shape

        # Run YOLO (Ultralytics expects BGR numpy arrays fine)
        try:
            results = self.model.predict(
                source=rgb,
                verbose=False,
                conf=0.4,     # tune as needed
                iou=0.5       # tune as needed
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

        # Find best detection among apples/bananas
        best = None  # (conf, cls_name, (x1,y1,x2,y2))
        names = self.model.names  # dict or list mapping class id -> name

        for b in r0.boxes:
            cls_id = int(b.cls.item())
            conf = float(b.conf.item())

            cls_name = names[cls_id] if isinstance(names, (list, tuple)) else names.get(cls_id, str(cls_id))
            if cls_name not in self.target_classes:
                continue

            x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().tolist()
            if best is None or conf > best[0]:
                best = (conf, cls_name, (x1, y1, x2, y2))

        if best is None:
            self.get_logger().info("YOLO: detections found, but none were apple/banana")
            return

        conf, cls_name, (x1, y1, x2, y2) = best

        # Compute bbox center (pixel)
        cx = int((x1 + x2) * 0.5)
        cy = int((y1 + y2) * 0.5)

        # Clamp center to image bounds
        cx = int(np.clip(cx, 0, w - 1))
        cy = int(np.clip(cy, 0, h - 1))

        # Robust depth at center using a small window median
        depth_m = self.get_depth_median_meters(depth, cx, cy)
        if depth_m is None or depth_m <= 0.0:
            self.get_logger().info(f"{cls_name}: depth invalid at center ({cx},{cy}); skipping")
            return

        # Deproject to 3D (camera frame)
        try:
            X, Y, Z = self.deproject(cx, cy, depth_m, w, h)
        except Exception as e:
            self.get_logger().error(f"Deproject failed: {e}")
            return

        self.get_logger().info(
            f"Best target: {cls_name} conf={conf:.2f} px=({cx},{cy}) depth={depth_m:.3f}m -> "
            f"XYZ=({X:.3f},{Y:.3f},{Z:.3f})"
        )

        self.publish_target((X, Y, Z), rgb_msg.header)

        # Optional: visualize (comment out if headless)
        # self.debug_viz(rgb, (x1, y1, x2, y2), cls_name, conf, (cx, cy), depth_m)

    # --------------------------------------------------

    def get_depth_median_meters(self, depth_img, px, py):
        """
        Returns median depth (meters) in a small window around (px,py).
        Handles common RealSense encodings:
          - uint16 in millimeters
          - float32 in meters
        """
        r = self.depth_window_radius
        h, w = depth_img.shape[:2]

        x0 = max(px - r, 0)
        x1 = min(px + r, w - 1)
        y0 = max(py - r, 0)
        y1 = min(py + r, h - 1)

        window = depth_img[y0:y1 + 1, x0:x1 + 1].astype(np.float32).reshape(-1)

        # Remove zeros / invalid
        window = window[np.isfinite(window)]
        window = window[window > 0]

        if window.size == 0:
            return None

        # Heuristic: if values look like mm (typical uint16), convert to meters
        # Many aligned depth topics are uint16 mm.
        med = float(np.median(window))
        if med > 10.0:  # >10 meters is unlikely for typical scenes, suggests mm scale
            med *= 0.001

        # If original image is uint16, it's almost certainly mm anyway, so this is safe.
        return med

    # --------------------------------------------------

    def camera_info_callback(self, msg: CameraInfo):
        """
        Extract pinhole intrinsics from the CameraInfo K matrix.
        K = [fx  0 cx]
            [ 0 fy cy]
            [ 0  0  1]
        """
        self.fx = msg.k[0]
        self.fy = msg.k[4]
        self.cx = msg.k[2]
        self.cy = msg.k[5]
        self.get_logger().info(
            f"Camera intrinsics received: fx={self.fx:.1f} fy={self.fy:.1f} "
            f"cx={self.cx:.1f} cy={self.cy:.1f}",
            once=True
        )

    # --------------------------------------------------

    def deproject(self, px, py, depth, w, h):
        """
        Pinhole back-projection using real camera intrinsics from CameraInfo.
        Falls back to simple estimates if intrinsics haven't arrived yet.
        """
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

    # --------------------------------------------------

    def publish_target(self, point_xyz, header):
        msg = PointStamped()
        msg.header = header
        msg.header.frame_id = "camera_color_optical_frame"

        msg.point.x = float(point_xyz[0])
        msg.point.y = float(point_xyz[1])
        msg.point.z = float(point_xyz[2])

        self.target_pub.publish(msg)

    # --------------------------------------------------

    def debug_viz(self, rgb, bbox, cls_name, conf, center, depth_m):
        x1, y1, x2, y2 = [int(v) for v in bbox]
        cx, cy = center

        vis = rgb.copy()
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(vis, (cx, cy), 4, (0, 0, 255), -1)
        txt = f"{cls_name} {conf:.2f} z={depth_m:.2f}m"
        cv2.putText(vis, txt, (x1, max(y1 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow("fruit_yolo", vis)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = FruitTargetNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
