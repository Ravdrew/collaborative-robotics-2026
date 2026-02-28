#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseWithCovarianceStamped
from apriltag_msgs.msg import AprilTagDetectionArray
import yaml
import tf_transformations
import math


class TagGlobalPoseNode(Node):

    def __init__(self):
        super().__init__('tag_global_pose_node')

        # Load tag map
        config_path = self.declare_parameter(
            'tag_map_file',
            'config/tag_locations.yaml'
        ).value

        with open(config_path, 'r') as f:
            self.tag_map = yaml.safe_load(f)['tag_locations']

        self.sub = self.create_subscription(
            AprilTagDetectionArray,
            '/tag_detections',
            self.tag_callback,
            10
        )

        self.pub = self.create_publisher(
            PoseWithCovarianceStamped,
            '/tag_global_pose',
            10
        )

        self.get_logger().info("Tag Global Pose Node Started")

    def tag_callback(self, msg):

        if len(msg.detections) == 0:
            return

        detection = msg.detections[0]
        tag_id = str(detection.id[0])

        if tag_id not in self.tag_map:
            self.get_logger().warn(f"Tag {tag_id} not in map")
            return

        tag_global = self.tag_map[tag_id]

        # Tag pose in camera frame
        pose_cam = detection.pose.pose.pose

        # Convert quaternion to yaw
        q = pose_cam.orientation
        _, _, yaw_cam = tf_transformations.euler_from_quaternion(
            [q.x, q.y, q.z, q.w]
        )

        # Tag position relative to camera
        tx = pose_cam.position.x
        ty = pose_cam.position.y

        # Known global tag pose
        gx = tag_global['x']
        gy = tag_global['y']
        gyaw = tag_global['yaw']

        # Compute robot global pose
        robot_x = gx - (math.cos(gyaw)*tx - math.sin(gyaw)*ty)
        robot_y = gy - (math.sin(gyaw)*tx + math.cos(gyaw)*ty)
        robot_yaw = gyaw - yaw_cam

        q_out = tf_transformations.quaternion_from_euler(0, 0, robot_yaw)

        pose_msg = PoseWithCovarianceStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = "map"

        pose_msg.pose.pose.position.x = robot_x
        pose_msg.pose.pose.position.y = robot_y
        pose_msg.pose.pose.orientation.x = q_out[0]
        pose_msg.pose.pose.orientation.y = q_out[1]
        pose_msg.pose.pose.orientation.z = q_out[2]
        pose_msg.pose.pose.orientation.w = q_out[3]

        # Covariance (tune later)
        pose_msg.pose.covariance[0] = 0.05
        pose_msg.pose.covariance[7] = 0.05
        pose_msg.pose.covariance[35] = 0.1

        self.pub.publish(pose_msg)


def main(args=None):
    rclpy.init(args=args)
    node = TagGlobalPoseNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
