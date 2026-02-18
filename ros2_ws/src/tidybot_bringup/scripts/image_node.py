#!/usr/bin/env python3
"""
TidyBot2 Image Processing Node

Subscribes to /pick_target and /place_target from audio_processing_node,
uses the RealSense D435 camera (RGB + Depth) to detect and localize objects
via YOLOv8 (ultralytics), and publishes the 3-D position.

Performs RGB-Depth alignment using camera intrinsics/extrinsics (the D435
has separate RGB and depth sensors with a physical baseline).

Valid pick targets : apple, banana
Valid place targets: none, basket, hand

Subscribed Topics:
    /pick_target              (std_msgs/String)        - object to pick
    /place_target             (std_msgs/String)        - place destination
    /camera/color/image_raw   (sensor_msgs/Image)      - RGB feed (rgb8, 640x480)
    /camera/depth/image_raw   (sensor_msgs/Image)      - Depth feed (16UC1 mm, 640x480)
    /camera/color/camera_info (sensor_msgs/CameraInfo)  - RGB camera intrinsics
    /camera/depth/camera_info (sensor_msgs/CameraInfo)  - Depth camera intrinsics (real HW)

Published Topics:
    /pick_target_global   (geometry_msgs/PointStamped) - 3-D coordinates of pick target
    /place_target_global  (geometry_msgs/PointStamped) - 3-D coordinates of place target

Usage:
    # Terminal 1 - launch sim
    ros2 launch tidybot_bringup sim.launch.py use_rviz:=false

    # Terminal 2
    ros2 run tidybot_bringup image_node.py
"""

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String
from geometry_msgs.msg import PointStamped

from cv_bridge import CvBridge

# YOLOv8
from ultralytics import YOLO


# -- Valid targets -----------------------------------------------------------
VALID_PICK_TARGETS = {"apple", "banana"}
VALID_PLACE_TARGETS = {"none", "basket", "hand"}

# Mapping from our target names -> COCO class names that YOLO recognises.
# YOLO COCO classes: 'apple' (id 47), 'banana' (id 46), 'bottle' (id 39),
# 'person' (id 0), 'bowl' (id 45) — we use 'person' for hand detection.
TARGET_TO_YOLO_CLASS = {
    "apple":  "apple",
    "banana": "banana",
    "basket": "bowl",       # closest COCO class
    "hand":   "person",    # detect person, then the hand is near them
}


# ===========================================================================
class ImageNode(Node):
    """RealSense-based image processing with RGB-Depth alignment + YOLOv8."""

    def __init__(self):
        super().__init__("image_node")

        # -- Parameters -------------------------------------------------------
        self.declare_parameter("pick_topic", "/pick_target")
        self.declare_parameter("place_topic", "/place_target")
        self.declare_parameter("rgb_topic", "/camera/color/image_raw")
        self.declare_parameter("depth_topic", "/camera/depth/image_raw")
        self.declare_parameter("rgb_info_topic", "/camera/color/camera_info")
        self.declare_parameter("depth_info_topic", "/camera/depth/camera_info")
        self.declare_parameter("detection_rate", 5.0)        # Hz - YOLO is local, can be fast
        self.declare_parameter("yolo_model", "yolov8n.pt")   # nano model for speed
        self.declare_parameter("yolo_conf", 0.4)              # confidence threshold

        pick_topic = self.get_parameter("pick_topic").value
        place_topic = self.get_parameter("place_topic").value
        rgb_topic = self.get_parameter("rgb_topic").value
        depth_topic = self.get_parameter("depth_topic").value
        rgb_info_topic = self.get_parameter("rgb_info_topic").value
        depth_info_topic = self.get_parameter("depth_info_topic").value
        detection_rate = self.get_parameter("detection_rate").value
        yolo_model_name = self.get_parameter("yolo_model").value
        self.yolo_conf = self.get_parameter("yolo_conf").value

        # Load YOLO model (downloads weights on first run)
        self.get_logger().info(f"Loading YOLO model: {yolo_model_name} ...")
        self.yolo_model = YOLO(yolo_model_name)
        self.get_logger().info("YOLO model loaded.")

        # -- Internal state ----------------------------------------------------
        self.cv_bridge = CvBridge()
        self.pick_target = None   # str | None
        self.place_target = None  # str | None
        self.latest_rgb = None    # np.ndarray | None
        self.latest_depth = None  # np.ndarray | None

        # Detection results are computed synchronously each loop (YOLO is fast)

        # Camera intrinsics - populated from CameraInfo messages
        self.rgb_intrinsics = None   # dict | None
        self.depth_intrinsics = None # dict | None

        # Extrinsics: rigid transform depth -> RGB.
        # D435 has ~25 mm baseline in X; in sim the cameras are co-located.
        self.depth_to_rgb_rotation = np.eye(3)
        self.depth_to_rgb_translation = np.zeros(3)  # metres

        # -- Subscribers -------------------------------------------------------
        self.pick_sub = self.create_subscription(
            String, pick_topic, self._cb_pick, 10
        )
        self.place_sub = self.create_subscription(
            String, place_topic, self._cb_place, 10
        )

        img_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE)

        self.rgb_sub = self.create_subscription(
            Image, rgb_topic, self._cb_rgb, img_qos
        )
        self.depth_sub = self.create_subscription(
            Image, depth_topic, self._cb_depth, img_qos
        )
        self.rgb_info_sub = self.create_subscription(
            CameraInfo, rgb_info_topic, self._cb_rgb_info, 10
        )
        self.depth_info_sub = self.create_subscription(
            CameraInfo, depth_info_topic, self._cb_depth_info, 10
        )

        # -- Publishers --------------------------------------------------------
        self.pick_global_pub = self.create_publisher(
            PointStamped, "/pick_target_global", 10
        )
        self.place_global_pub = self.create_publisher(
            PointStamped, "/place_target_global", 10
        )

        # -- Periodic detection loop -------------------------------------------
        self.timer = self.create_timer(1.0 / detection_rate, self._detection_loop)

        self.get_logger().info("=" * 55)
        self.get_logger().info("  TidyBot2 Image Processing Node (YOLOv8)")
        self.get_logger().info("=" * 55)
        self.get_logger().info(f"  RGB topic  : {rgb_topic}")
        self.get_logger().info(f"  Depth topic: {depth_topic}")
        self.get_logger().info(f"  Pick topic : {pick_topic}")
        self.get_logger().info(f"  Place topic: {place_topic}")
        self.get_logger().info(f"  YOLO model : {yolo_model_name}")
        self.get_logger().info(f"  Confidence : {self.yolo_conf}")
        self.get_logger().info(f"  Rate       : {detection_rate} Hz")
        self.get_logger().info("  Valid picks : apple, banana")
        self.get_logger().info("  Valid places: none, basket, hand")
        self.get_logger().info("  Waiting for camera feed and targets ...")

    # ======================== CALLBACKS =====================================

    def _cb_pick(self, msg):
        target = msg.data.strip().lower()
        if target not in VALID_PICK_TARGETS:
            self.get_logger().warn(
                f"Ignoring unknown pick target '{target}'. "
                f"Valid: {VALID_PICK_TARGETS}"
            )
            return
        self.pick_target = target
        self.get_logger().info(f"Pick target set -> '{target}'")

    def _cb_place(self, msg):
        target = msg.data.strip().lower()
        if target not in VALID_PLACE_TARGETS:
            self.get_logger().warn(
                f"Ignoring unknown place target '{target}'. "
                f"Valid: {VALID_PLACE_TARGETS}"
            )
            return
        self.place_target = target
        self.get_logger().info(f"Place target set -> '{target}'")

    def _cb_rgb(self, msg):
        try:
            self.latest_rgb = self.cv_bridge.imgmsg_to_cv2(msg, "rgb8")
        except Exception as e:
            self.get_logger().warn(f"RGB conversion failed: {e}")

    def _cb_depth(self, msg):
        try:
            self.latest_depth = self.cv_bridge.imgmsg_to_cv2(msg, "passthrough")
        except Exception as e:
            self.get_logger().warn(f"Depth conversion failed: {e}")

    def _cb_rgb_info(self, msg):
        if self.rgb_intrinsics is not None:
            return
        self.rgb_intrinsics = self._extract_intrinsics(msg)
        ci = self.rgb_intrinsics
        self.get_logger().info(
            f"RGB intrinsics: fx={ci['fx']:.1f} fy={ci['fy']:.1f} "
            f"cx={ci['cx']:.1f} cy={ci['cy']:.1f} ({ci['width']}x{ci['height']})"
        )

    def _cb_depth_info(self, msg):
        if self.depth_intrinsics is not None:
            return
        self.depth_intrinsics = self._extract_intrinsics(msg)
        ci = self.depth_intrinsics
        # Recover baseline from projection matrix P[0,3] = -fx*baseline
        if len(msg.p) >= 4 and msg.p[3] != 0.0:
            fx = msg.p[0]
            if fx != 0.0:
                baseline = -msg.p[3] / fx
                self.depth_to_rgb_translation[0] = baseline
                self.get_logger().info(
                    f"Depth->RGB baseline: {baseline * 1000:.1f} mm"
                )
        self.get_logger().info(
            f"Depth intrinsics: fx={ci['fx']:.1f} fy={ci['fy']:.1f} "
            f"cx={ci['cx']:.1f} cy={ci['cy']:.1f} ({ci['width']}x{ci['height']})"
        )

    # ==================== INTRINSICS HELPERS ================================

    @staticmethod
    def _extract_intrinsics(msg):
        """Pull fx, fy, cx, cy, K, D from a CameraInfo message.

        K is the 3x3 row-major intrinsic matrix:
            K = [[fx,  0, cx],
                 [ 0, fy, cy],
                 [ 0,  0,  1]]
        """
        K = np.array(msg.k).reshape(3, 3)
        D = np.array(msg.d) if len(msg.d) > 0 else np.zeros(5)
        return {
            "fx": K[0, 0], "fy": K[1, 1],
            "cx": K[0, 2], "cy": K[1, 2],
            "width": msg.width, "height": msg.height,
            "K": K, "D": D,
        }

    def _default_intrinsics(self):
        """Fallback D435 intrinsics at 640x480, fovy=42deg.

        fy = height / (2*tan(fovy/2)) = 480 / (2*tan(21deg)) ~ 625.2
        fx = fy  (square pixels)
        cx = 320, cy = 240
        """
        fy = 480.0 / (2.0 * np.tan(np.radians(21.0)))
        fx = fy
        K = np.array([[fx, 0, 320.0],
                       [0, fy, 240.0],
                       [0,  0,   1.0]])
        return {
            "fx": fx, "fy": fy, "cx": 320.0, "cy": 240.0,
            "width": 640, "height": 480, "K": K, "D": np.zeros(5),
        }

    # ==================== RGB-DEPTH ALIGNMENT ==============================

    def align_depth_to_rgb(self, depth_image):
        """Align a depth image to the RGB camera frame.

        The RealSense D435 has physically separate depth and RGB sensors.
        Their images do NOT overlap pixel-for-pixel because:

          - Each sensor has its own *intrinsics* (focal length, principal
            point, distortion coefficients).
          - There is a rigid *extrinsic* transform (rotation R and
            translation t) between them -- mostly a ~25 mm baseline in X.

        Algorithm (per-pixel reprojection):

        For every valid depth pixel (u_d, v_d) with depth z:

        1. Back-project into 3-D using the depth intrinsics:
               X_d = (u_d - cx_d) * z / fx_d
               Y_d = (v_d - cy_d) * z / fy_d
               Z_d = z

        2. Transform from depth frame to RGB frame:
               P_rgb = R @ P_depth + t

        3. Project onto RGB image plane using RGB intrinsics:
               u_rgb = fx_rgb * X_rgb / Z_rgb + cx_rgb
               v_rgb = fy_rgb * Y_rgb / Z_rgb + cy_rgb

        4. Z-buffer write (closer values win).

        In simulation (R=I, t=0, same intrinsics) this reduces to a copy.

        Args:
            depth_image: (H_d x W_d) uint16 array, depth in millimetres.

        Returns:
            Aligned depth image (H_rgb x W_rgb) uint16 in millimetres.
        """
        rgb_intr = self.rgb_intrinsics or self._default_intrinsics()
        depth_intr = self.depth_intrinsics or rgb_intr

        R = self.depth_to_rgb_rotation
        t = self.depth_to_rgb_translation

        h_d, w_d = depth_image.shape[:2]
        h_rgb = rgb_intr["height"]
        w_rgb = rgb_intr["width"]

        aligned = np.zeros((h_rgb, w_rgb), dtype=np.uint16)

        # --- Fast path: trivial alignment (simulation) ---
        same_intr = (
            abs(rgb_intr["fx"] - depth_intr["fx"]) < 1.0
            and abs(rgb_intr["fy"] - depth_intr["fy"]) < 1.0
            and abs(rgb_intr["cx"] - depth_intr["cx"]) < 1.0
            and abs(rgb_intr["cy"] - depth_intr["cy"]) < 1.0
        )
        no_offset = np.allclose(t, 0.0) and np.allclose(R, np.eye(3))

        if same_intr and no_offset:
            min_h, min_w = min(h_d, h_rgb), min(w_d, w_rgb)
            aligned[:min_h, :min_w] = depth_image[:min_h, :min_w]
            return aligned

        # --- Full reprojection (real hardware) ---
        v_coords, u_coords = np.mgrid[0:h_d, 0:w_d]
        valid = depth_image > 0
        u_d = u_coords[valid].astype(np.float64)
        v_d = v_coords[valid].astype(np.float64)
        z_m = depth_image[valid].astype(np.float64) / 1000.0

        if z_m.size == 0:
            return aligned

        # 1. Back-project to 3-D in depth frame
        x_d = (u_d - depth_intr["cx"]) * z_m / depth_intr["fx"]
        y_d = (v_d - depth_intr["cy"]) * z_m / depth_intr["fy"]
        pts_depth = np.stack([x_d, y_d, z_m], axis=1)

        # 2. Rigid transform depth -> RGB frame
        pts_rgb = (R @ pts_depth.T).T + t

        # 3. Project onto RGB image plane
        z_rgb = pts_rgb[:, 2]
        ok = z_rgb > 1e-3
        u_rgb = np.round(
            rgb_intr["fx"] * pts_rgb[ok, 0] / z_rgb[ok] + rgb_intr["cx"]
        ).astype(np.int32)
        v_rgb = np.round(
            rgb_intr["fy"] * pts_rgb[ok, 1] / z_rgb[ok] + rgb_intr["cy"]
        ).astype(np.int32)
        z_mm = (z_rgb[ok] * 1000.0).astype(np.uint16)

        # 4. Z-buffer write (farthest first so nearer overwrites)
        in_bounds = (
            (u_rgb >= 0) & (u_rgb < w_rgb) &
            (v_rgb >= 0) & (v_rgb < h_rgb)
        )
        u_rgb = u_rgb[in_bounds]
        v_rgb = v_rgb[in_bounds]
        z_mm = z_mm[in_bounds]
        order = np.argsort(-z_mm)
        aligned[v_rgb[order], u_rgb[order]] = z_mm[order]

        return aligned

    # ================= YOLOv8 OBJECT DETECTION =============================

    def _detect_target(self, rgb_image, target_name):
        """Run YOLOv8 on the image and find the best detection for target_name.

        Args:
            rgb_image:   (H, W, 3) uint8 RGB image.
            target_name: one of the keys in TARGET_TO_YOLO_CLASS.

        Returns:
            dict with {found, bbox [x1,y1,x2,y2], confidence} or None.
        """
        yolo_class = TARGET_TO_YOLO_CLASS.get(target_name)
        if yolo_class is None:
            return {"found": False, "bbox": [0, 0, 0, 0], "confidence": 0.0}

        # YOLO expects BGR
        bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        results = self.yolo_model(bgr, conf=self.yolo_conf, verbose=False)

        best_box = None
        best_conf = 0.0

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                cls_name = self.yolo_model.names[cls_id]
                conf = float(box.conf[0])

                if cls_name == yolo_class and conf > best_conf:
                    best_conf = conf
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    best_box = [int(x1), int(y1), int(x2), int(y2)]

        if best_box is not None:
            return {
                "found": True,
                "bbox": best_box,
                "confidence": best_conf,
            }
        return {"found": False, "bbox": [0, 0, 0, 0], "confidence": 0.0}

    # ==================== 3-D POSITION =====================================

    def pixel_to_3d(self, u, v, aligned_depth):
        """Convert (u, v) pixel + aligned depth -> 3-D point in camera frame.

        Uses RGB intrinsics (depth already aligned to RGB).
        Samples a small window and takes median depth for robustness.

        Returns np.array([x, y, z]) in metres, or None.
        """
        intr = self.rgb_intrinsics or self._default_intrinsics()

        half = 5
        v0 = max(0, v - half)
        v1 = min(aligned_depth.shape[0], v + half + 1)
        u0 = max(0, u - half)
        u1 = min(aligned_depth.shape[1], u + half + 1)

        roi = aligned_depth[v0:v1, u0:u1]
        valid = roi[roi > 0]
        if valid.size == 0:
            return None

        z_m = float(np.median(valid)) / 1000.0
        x = (u - intr["cx"]) * z_m / intr["fx"]
        y = (v - intr["cy"]) * z_m / intr["fy"]
        return np.array([x, y, z_m])

    # ==================== MAIN DETECTION LOOP ==============================

    def _detection_loop(self):
        """Periodic loop: align -> detect (YOLO) -> localise -> publish."""
        if self.latest_rgb is None or self.latest_depth is None:
            return
        if self.pick_target is None:
            return

        # 1. Align depth to RGB ------------------------------------------------
        aligned_depth = self.align_depth_to_rgb(self.latest_depth)

        rgb = self.latest_rgb.copy()
        now = self.get_clock().now().to_msg()

        # 2. Detect pick target ------------------------------------------------
        pick_det = self._detect_target(rgb, self.pick_target)
        if pick_det and pick_det["found"]:
            point_3d = self._bbox_to_3d(pick_det["bbox"], rgb.shape, aligned_depth)
            if point_3d is not None:
                msg = PointStamped()
                msg.header.stamp = now
                msg.header.frame_id = "camera_color_optical_frame"
                msg.point.x = float(point_3d[0])
                msg.point.y = float(point_3d[1])
                msg.point.z = float(point_3d[2])
                self.pick_global_pub.publish(msg)
                self.get_logger().info(
                    f"pick_target_global: '{self.pick_target}' "
                    f"({point_3d[0]:.3f}, {point_3d[1]:.3f}, {point_3d[2]:.3f})m "
                    f"[conf={pick_det['confidence']:.2f}]"
                )
            else:
                self.get_logger().warn(
                    f"'{self.pick_target}' detected but no valid depth"
                )
        else:
            self.get_logger().debug(f"YOLO: '{self.pick_target}' not found")

        # 3. Detect place target -----------------------------------------------
        if self.place_target and self.place_target != "none":
            place_det = self._detect_target(rgb, self.place_target)
            if place_det and place_det["found"]:
                point_3d = self._bbox_to_3d(place_det["bbox"], rgb.shape, aligned_depth)
                if point_3d is not None:
                    msg = PointStamped()
                    msg.header.stamp = now
                    msg.header.frame_id = "camera_color_optical_frame"
                    msg.point.x = float(point_3d[0])
                    msg.point.y = float(point_3d[1])
                    msg.point.z = float(point_3d[2])
                    self.place_global_pub.publish(msg)
                    self.get_logger().info(
                        f"place_target_global: '{self.place_target}' "
                        f"({point_3d[0]:.3f}, {point_3d[1]:.3f}, {point_3d[2]:.3f})m "
                        f"[conf={place_det['confidence']:.2f}]"
                    )
                else:
                    self.get_logger().warn(
                        f"'{self.place_target}' detected but no valid depth"
                    )

    def _bbox_to_3d(self, bbox, img_shape, aligned_depth):
        """Convert a bbox [x1,y1,x2,y2] to a 3-D point via depth lookup."""
        x1, y1, x2, y2 = bbox
        h, w = img_shape[:2]
        x1 = max(0, min(int(x1), w - 1))
        y1 = max(0, min(int(y1), h - 1))
        x2 = max(0, min(int(x2), w - 1))
        y2 = max(0, min(int(y2), h - 1))
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        return self.pixel_to_3d(cx, cy, aligned_depth)


# ===========================================================================
def main(args=None):
    rclpy.init(args=args)
    node = ImageNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
