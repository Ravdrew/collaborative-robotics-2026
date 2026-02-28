#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

class InitialMover(Node):
    def __init__(self):
        super().__init__('initial_mover')

        self.publisher1 = self.create_publisher(Twist, '/cmd_vel', 10)
        self.publisher2 = self.create_publisher(Twist, '/cmd_vel_nav', 10)

        self.moved = False
        self.move_timer = self.create_timer(0.1, self.move_once)
        self.stop_timer = None

    def move_once(self):
        if not self.moved:
            twist = Twist()
            twist.linear.x = 0.1

            self.publisher1.publish(twist)
            self.publisher2.publish(twist)

            self.get_logger().info('Moving forward into costmap')

            self.moved = True

            # cancel move timer so it only runs once
            self.move_timer.cancel()

            # create stop timer
            self.stop_timer = self.create_timer(1.0, self.stop_robot)

    def stop_robot(self):
        twist = Twist()

        self.publisher1.publish(twist)
        self.publisher2.publish(twist)

        self.get_logger().info('Stopped robot')

        # cancel stop timer so it only runs once
        self.stop_timer.cancel()

def main(args=None):
    rclpy.init(args=args)
    node = InitialMover()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()