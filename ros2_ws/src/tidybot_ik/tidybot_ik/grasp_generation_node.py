#!/usr/bin/env python3
"""
TidyBot2 Grasp Generation Node

Given an object location in the camera frame, computes and publishes the
end-effector pose for a top-down grasp in the robot (base_link) frame.

The output pose is ready to pass directly to the /plan_to_target service
(see test_planner_sim.py for usage).

Topics:
- Subscribes: /object_pose_in_camera (geometry_msgs/Pose)
    Object position in camera_color_optical_frame. Only the position is used;
    the orientation field is ignored — the grasp orientation is always top-down.

- Publishes: /EEF_pose_command (geometry_msgs/Pose)
    End-effector pose in base_link frame, offset above the object with a
    fingers-pointing-down orientation suitable for a top-down grasp.

Parameters:
- camera_frame (str, default: 'camera_color_optical_frame')
    TF frame in which the incoming object pose is expressed.
- robot_frame (str, default: 'base_link')
    TF frame in which the output EEF pose is expressed.
- grasp_height_offset (float, default: 0.10)
    Height (metres) added above the object position along +Z (base_link up).

Usage:
    # Terminal 1: Start simulation
    ros2 launch tidybot_bringup sim.launch.py

    # Terminal 2: Start this node
    ros2 run tidybot_ik grasp_generation_node

    # Terminal 3: Publish a test object pose
    ros2 topic pub /object_pose_in_camera geometry_msgs/msg/Pose \\
        "{position: {x: 0.0, y: 0.3, z: 0.5}, orientation: {w: 1.0}}"

    # Terminal 4: Watch the resulting EEF pose
    ros2 topic echo /EEF_pose_command
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, PoseStamped

import tf2_ros
import tf2_geometry_msgs  # noqa: F401 — registers transform functions


# ---------------------------------------------------------------------------
# Top-down grasp orientation in base_link frame (quaternion: w, x, y, z)
#
# Ry(π/2) rotates the pinch_site x-axis (gripper finger direction) to point
# along -Z (straight down).  This matches ORIENT_FINGERS_DOWN in
# test_planner_sim.py.
# ---------------------------------------------------------------------------
_FINGERS_DOWN_HORIZONTAL_QW = 0.5
_FINGERS_DOWN_HORIZONTAL_QX = 0.5
_FINGERS_DOWN_HORIZONTAL_QY = 0.5
_FINGERS_DOWN_HORIZONTAL_QZ = -0.5
_FINGERS_DOWN_VERTICAL_QW = 0.707107
_FINGERS_DOWN_VERTICAL_QX = 0.0
_FINGERS_DOWN_VERTICAL_QY = 0.707107
_FINGERS_DOWN_VERTICAL_QZ = 0.0


class GraspGenerationNode(Node):
    """Reactive node: object pose in camera frame → top-down EEF pose in base_link."""

    def __init__(self):
        super().__init__('grasp_generation')

        # Parameters
        self.declare_parameter('camera_frame', 'camera_color_optical_frame')
        self.declare_parameter('robot_frame', 'base_link')
        self.declare_parameter('grasp_height_offset', 0.10)

        self.camera_frame = (
            self.get_parameter('camera_frame').get_parameter_value().string_value
        )
        self.robot_frame = (
            self.get_parameter('robot_frame').get_parameter_value().string_value
        )
        self.grasp_height_offset = (
            self.get_parameter('grasp_height_offset').get_parameter_value().double_value
        )

        # TF2
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Subscriber
        self.create_subscription(
            Pose,
            '/object_pose_in_camera',
            self._object_pose_callback,
            10,
        )

        # Publisher
        self._pub = self.create_publisher(Pose, '/EEF_pose_command', 10)

        self.get_logger().info('=' * 50)
        self.get_logger().info('Grasp Generation Node')
        self.get_logger().info('=' * 50)
        self.get_logger().info(f'  Camera frame    : {self.camera_frame}')
        self.get_logger().info(f'  Robot frame     : {self.robot_frame}')
        self.get_logger().info(f'  Height offset   : {self.grasp_height_offset:.3f} m')
        self.get_logger().info('Waiting for /object_pose_in_camera ...')

    # ------------------------------------------------------------------
    # Callback
    # ------------------------------------------------------------------

    def _object_pose_callback(self, msg: Pose) -> None:
        """Transform object pose to robot frame and publish top-down EEF pose."""

        # Wrap bare Pose in a stamped message so tf2 can work with it.
        # Use Time(0) so lookup_transform returns the latest available transform.
        stamped = PoseStamped()
        stamped.header.stamp = rclpy.time.Time().to_msg()
        stamped.header.frame_id = self.camera_frame
        stamped.pose = msg

        # Look up and apply the transform
        try:
            transform = self.tf_buffer.lookup_transform(
                self.robot_frame,
                self.camera_frame,
                rclpy.time.Time(),
            )
        except tf2_ros.LookupException as exc:
            self.get_logger().warn(f'TF lookup failed: {exc}')
            return
        except tf2_ros.ConnectivityException as exc:
            self.get_logger().warn(f'TF connectivity error: {exc}')
            return
        except tf2_ros.ExtrapolationException as exc:
            self.get_logger().warn(f'TF extrapolation error: {exc}')
            return

        # do_transform_pose_stamped accepts PoseStamped and returns PoseStamped
        pose_in_robot: PoseStamped = tf2_geometry_msgs.do_transform_pose_stamped(
            stamped, transform
        )

        # Build top-down EEF pose:
        #   - position: object XY in base_link + fixed height offset in Z
        #   - orientation: fingers pointing straight down (see module docstring)
        eef_pose = Pose()
        eef_pose.position.x = pose_in_robot.pose.position.x
        eef_pose.position.y = pose_in_robot.pose.position.y
        eef_pose.position.z = pose_in_robot.pose.position.z + self.grasp_height_offset
        if msg.orientation.w == 1.0:
            eef_pose.orientation.w = _FINGERS_DOWN_HORIZONTAL_QW
            eef_pose.orientation.x = _FINGERS_DOWN_HORIZONTAL_QX
            eef_pose.orientation.y = _FINGERS_DOWN_HORIZONTAL_QY
            eef_pose.orientation.z = _FINGERS_DOWN_HORIZONTAL_QZ
        else:
            eef_pose.orientation.w = _FINGERS_DOWN_VERTICAL_QW
            eef_pose.orientation.x = _FINGERS_DOWN_VERTICAL_QX
            eef_pose.orientation.y = _FINGERS_DOWN_VERTICAL_QY
            eef_pose.orientation.z = _FINGERS_DOWN_VERTICAL_QZ

        self._pub.publish(eef_pose)

        self.get_logger().info(
            f'Object in {self.robot_frame}: '
            f'({pose_in_robot.pose.position.x:.3f}, '
            f'{pose_in_robot.pose.position.y:.3f}, '
            f'{pose_in_robot.pose.position.z:.3f})'
        )
        self.get_logger().info(
            f'EEF pose published: '
            f'({eef_pose.position.x:.3f}, '
            f'{eef_pose.position.y:.3f}, '
            f'{eef_pose.position.z:.3f})'
        )


def main(args=None):
    rclpy.init(args=args)
    node = GraspGenerationNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
