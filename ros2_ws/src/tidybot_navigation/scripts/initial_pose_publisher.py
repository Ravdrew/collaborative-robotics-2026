#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import OccupancyGrid

class InitialPosePublisher(Node):

    def __init__(self):
        super().__init__('initial_pose_publisher')

        # Publisher to Nav2
        self.pose_pub = self.create_publisher(
            PoseWithCovarianceStamped,
            '/initialpose',
            10
        )

        # Wait for map before publishing
        self.map_received = False

        self.map_sub = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            10
        )

        self.timer = self.create_timer(0.5, self.publish_initial_pose)

        self.get_logger().info("Waiting for /map before publishing initial pose...")

    def map_callback(self, msg):
        if not self.map_received:
            self.get_logger().info("Map received — ready to initialize pose")
        self.map_received = True

    def publish_initial_pose(self):

        if not self.map_received:
            return

        msg = PoseWithCovarianceStamped()

        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()

        # Starting pose (adjust if needed)
        msg.pose.pose.position.x = 0.0
        msg.pose.pose.position.y = 0.0
        msg.pose.pose.position.z = 0.0

        msg.pose.pose.orientation.x = 0.0
        msg.pose.pose.orientation.y = 0.0
        msg.pose.pose.orientation.z = 0.0
        msg.pose.pose.orientation.w = 1.0

        # Covariance (confidence)
        msg.pose.covariance[0] = 0.25
        msg.pose.covariance[7] = 0.25
        msg.pose.covariance[35] = 0.068

        self.pose_pub.publish(msg)

        self.get_logger().info("Initial pose published!")

        # Only run once
        self.timer.cancel()


def main(args=None):
    rclpy.init(args=args)
    node = InitialPosePublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()