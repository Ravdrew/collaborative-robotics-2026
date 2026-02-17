#!/usr/bin/env python3
"""
Naive depth estimator for TidyBot MuJoCo simulation.

What this node does:
1) Subscribes to /camera/depth/image_raw (16UC1 mm in sim bridge).
2) Publishes plausible depth CameraInfo on /camera/depth/camera_info
   using MuJoCo d435_depth settings (640x480, fovy=57deg).
3) Converts a horizontal depth band into a naive LaserScan-like signal.

This is intentionally simple and not a full mapping/perception pipeline.
"""

import math
import warnings

import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image, LaserScan
from std_msgs.msg import Header


class DepthEstimator(Node):
    def __init__(self) -> None:
        super().__init__("our_depth_estimator")

        # ROS topics
        self.declare_parameter("depth_topic", "/camera/depth/image_raw")
        self.declare_parameter("depth_info_topic", "/camera/depth/camera_info")
        self.declare_parameter("scan_topic", "/obstacles/scan")

        # d435_depth camera model from MuJoCo XML
        self.declare_parameter("width", 640)
        self.declare_parameter("height", 480)
        self.declare_parameter("fovy_deg", 57.0)
        self.declare_parameter("frame_id", "camera_depth_optical_frame")

        # Naive scan extraction settings
        self.declare_parameter("range_min", 0.25)
        self.declare_parameter("range_max", 6.0)
        self.declare_parameter("band_half_height_px", 4)
        self.declare_parameter("sample_step_px", 2)
        self.declare_parameter("nan_as_inf", True)

        self.depth_topic = str(self.get_parameter("depth_topic").value)
        self.depth_info_topic = str(self.get_parameter("depth_info_topic").value)
        self.scan_topic = str(self.get_parameter("scan_topic").value)

        self.width = int(self.get_parameter("width").value)
        self.height = int(self.get_parameter("height").value)
        self.fovy_deg = float(self.get_parameter("fovy_deg").value)
        self.frame_id = str(self.get_parameter("frame_id").value)

        self.range_min = float(self.get_parameter("range_min").value)
        self.range_max = float(self.get_parameter("range_max").value)
        self.band_half_height_px = int(self.get_parameter("band_half_height_px").value)
        self.sample_step_px = max(1, int(self.get_parameter("sample_step_px").value))
        self.nan_as_inf = bool(self.get_parameter("nan_as_inf").value)

        self.bridge = CvBridge()
        self.depth_msgs_seen = 0
        self.warned_encoding = False

        # Build intrinsics once and reuse.
        self._camera_info_template = self._build_camera_info_template()
        self.fx = float(self._camera_info_template.k[0])
        self.cx = float(self._camera_info_template.k[2])

        self.info_pub = self.create_publisher(CameraInfo, self.depth_info_topic, 10)
        self.scan_pub = self.create_publisher(LaserScan, self.scan_topic, 10)
        self.create_subscription(Image, self.depth_topic, self.on_depth, 10)

        self.create_timer(2.0, self._health_log)

        self.get_logger().info(
            "our_depth_estimator started.\n"
            f"  Subscribing: {self.depth_topic}\n"
            f"  Publishing:  {self.depth_info_topic} (CameraInfo)\n"
            f"  Publishing:  {self.scan_topic} (LaserScan)\n"
            f"  Frame:       {self.frame_id}"
        )

    def _health_log(self) -> None:
        if self.depth_msgs_seen == 0:
            self.get_logger().warn(
                "No depth images received yet. Check sim and topic "
                f"'{self.depth_topic}'."
            )

    def _build_camera_info_template(self) -> CameraInfo:
        fy = (self.height / 2.0) / math.tan(math.radians(self.fovy_deg) / 2.0)
        fx = fy
        cx = self.width / 2.0
        cy = self.height / 2.0

        info = CameraInfo()
        info.width = self.width
        info.height = self.height
        info.distortion_model = "plumb_bob"
        info.d = [0.0, 0.0, 0.0, 0.0, 0.0]
        info.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
        info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        info.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
        return info

    def _make_camera_info(self, stamp) -> CameraInfo:
        info = CameraInfo()
        info.header = Header(stamp=stamp, frame_id=self.frame_id)
        info.width = self._camera_info_template.width
        info.height = self._camera_info_template.height
        info.distortion_model = self._camera_info_template.distortion_model
        info.d = list(self._camera_info_template.d)
        info.k = list(self._camera_info_template.k)
        info.r = list(self._camera_info_template.r)
        info.p = list(self._camera_info_template.p)
        return info

    def on_depth(self, msg: Image) -> None:
        self.depth_msgs_seen += 1

        info = self._make_camera_info(msg.header.stamp)
        self.info_pub.publish(info)

        depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        depth = np.asarray(depth)

        if depth.ndim != 2:
            self.get_logger().warn("Depth image is not single-channel (HxW). Skipping frame.")
            return

        if depth.dtype == np.uint16 or msg.encoding == "16UC1":
            depth_m = depth.astype(np.float32) * 0.001
        elif msg.encoding == "32FC1" or depth.dtype in (np.float32, np.float64):
            depth_m = depth.astype(np.float32)
        else:
            if not self.warned_encoding:
                self.get_logger().warn(
                    f"Unexpected depth encoding='{msg.encoding}', dtype={depth.dtype}; "
                    "attempting float32 meters."
                )
                self.warned_encoding = True
            depth_m = depth.astype(np.float32)

        # Drop non-positive and non-finite readings before median reduction.
        depth_m = np.where(np.isfinite(depth_m) & (depth_m > 0.0), depth_m, np.nan)

        h, w = depth_m.shape
        band = self.band_half_height_px
        step = self.sample_step_px
        center_row = h // 2
        r0 = max(0, center_row - band)
        r1 = min(h, center_row + band + 1)

        band_image = depth_m[r0:r1, ::step]
        if band_image.size == 0:
            return

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            depth_slice = np.nanmedian(band_image, axis=0)

        u = np.arange(0, w, step, dtype=np.float32)
        if len(u) != len(depth_slice):
            n = min(len(u), len(depth_slice))
            u = u[:n]
            depth_slice = depth_slice[:n]
            if n == 0:
                return

        angles = np.arctan((u - self.cx) / self.fx)
        with np.errstate(divide="ignore", invalid="ignore"):
            ranges = depth_slice / np.cos(angles)

        if self.nan_as_inf:
            ranges = np.where(np.isfinite(ranges), ranges, np.inf)
        else:
            ranges = np.where(np.isfinite(ranges), ranges, 0.0)

        ranges = np.where(
            (ranges >= self.range_min) & (ranges <= self.range_max),
            ranges,
            np.inf,
        )

        scan = LaserScan()
        scan.header = Header(stamp=msg.header.stamp, frame_id=self.frame_id)
        scan.angle_min = float(angles[0])
        scan.angle_max = float(angles[-1])
        scan.angle_increment = float(
            (scan.angle_max - scan.angle_min) / max(1, len(angles) - 1)
        )
        scan.time_increment = 0.0
        scan.scan_time = 0.0
        scan.range_min = self.range_min
        scan.range_max = self.range_max
        scan.ranges = ranges.astype(np.float32).tolist()

        self.scan_pub.publish(scan)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DepthEstimator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
