"""
Nav2 navigation launch for TidyBot2.

This is an OVERLAY launch file — run it on top of the existing sim or real launch.
It starts depthimage_to_laserscan, SLAM Toolbox, and the full Nav2 stack.

Usage:
  # Simulation (on top of sim.launch.py)
  ros2 launch tidybot_navigation navigation.launch.py sim:=true

  # Real hardware with depth camera (on top of real.launch.py)
  ros2 launch tidybot_navigation navigation.launch.py

  # Real hardware with LiDAR (skip depthimage_to_laserscan)
  ros2 launch tidybot_navigation navigation.launch.py scan_source:=lidar

NOTE: The D435 is on a pan-tilt mount. Keep camera at pan=0, tilt=0 during
navigation, otherwise the laser scan plane rotates and corrupts the costmap.
This limitation goes away once a real LiDAR is available (scan_source:=lidar).
"""

import os
# import sys

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction, ExecuteProcess, TimerAction
from launch.conditions import IfCondition, LaunchConfigurationEquals
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, Command, TextSubstitution
from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import Node


def launch_setup(context, *args, **kwargs):
    pkg_dir = get_package_share_directory('tidybot_navigation')
    nav2_bringup_dir = get_package_share_directory('nav2_bringup')

    use_sim_time = LaunchConfiguration('use_sim_time')
    use_rviz = LaunchConfiguration('use_rviz')
    is_sim = LaunchConfiguration('sim').perform(context) == 'true'

    # In simulation, use un-flipped depth on nav topics
    # On real hardware, use standard RealSense topics
    # Both use camera_link as output frame (X-forward, Y-left for correct 2D SLAM)
    if is_sim:
        depth_topic = '/camera/depth/image_nav'
        depth_info_topic = '/camera/depth/camera_info_nav'
    else:
        depth_topic = '/camera/depth/image_raw'
        depth_info_topic = '/camera/depth/camera_info'
    scan_output_frame = 'camera_link'  # from URDF, not flipped like camera_link

    # depthimage_to_laserscan — only when scan_source:=depth
    depth_to_scan_node = Node(
        condition=LaunchConfigurationEquals('scan_source', 'depth'),
        package='depthimage_to_laserscan',
        executable='depthimage_to_laserscan_node',
        name='depthimage_to_laserscan',
        parameters=[{
            'use_sim_time': use_sim_time,
            'scan_time': 0.033,
            'range_min': 0.28,
            'range_max': 5.0,
            'scan_height': 100,
            'output_frame': scan_output_frame,
        }],
        remappings=[
            ('depth', depth_topic),
            ('depth_camera_info', depth_info_topic),
            ('scan', '/scan'),
        ],
    )

    # SLAM Toolbox (online async)
    slam_toolbox_node = Node(
        package='slam_toolbox',
        executable='async_slam_toolbox_node',
        name='slam_toolbox',
        output='screen',
        parameters=[
            os.path.join(pkg_dir, 'config', 'slam_toolbox_params.yaml'),
            {'use_sim_time': use_sim_time},
        ],
    )

    # Nav2 bringup (controller, planner, behavior, bt_navigator, etc.)
    nav2_bringup = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(nav2_bringup_dir, 'launch', 'navigation_launch.py')
        ),
        launch_arguments={
            'use_sim_time': use_sim_time,
            'params_file': os.path.join(pkg_dir, 'config', 'nav2_params.yaml'),
        }.items(),
    )

    # this calls the script that moves the robot forward on startup into the cost map
    initial_mover_node = Node(
        package='tidybot_navigation',
        executable='init_forward.py',
        name='initial_mover',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}],
    )

    # RViz with navigation config
    rviz_node = Node(
        condition=IfCondition(use_rviz),
        package='rviz2',
        executable='rviz2',
        name='rviz2_nav',
        arguments=['-d', os.path.join(pkg_dir, 'rviz', 'navigation.rviz')],
        parameters=[{'use_sim_time': use_sim_time}],
    )

    # # AprilTag parameters, from apriltag_ros
    # apriltag_node = Node(
    # package='apriltag_ros',
    # executable='apriltag_node',
    # name='apriltag_detector',
    # parameters=[{
    #     'image_transport': 'raw',
    #     'family': 'tag36h11',
    #     'size': 0.162,   # <-- tag size
    # }],
    # remappings=[('image_rect', '/camera/color/image_raw'),('camera_info', '/camera/color/camera_info'),],
    # )

    # # our personal tag localization node
    # tag_localization_node = Node(
    # package='tidybot_navigation',
    # executable='tag_global_pose_node',
    # name='tag_global_pose',
    # parameters=[os.path.join(pkg_dir, 'config', 'tag_locations.yaml')],
    # )

    # # EKF to smoothly integrate the global strapdown from the apriltags to the continuous slam happening, relies on robot_localization package
    # ekf_node = Node(
    # package='robot_localization',
    # executable='ekf_node',
    # name='ekf_filter_node',
    # output='screen',
    # parameters=[
    #     os.path.join(pkg_dir, 'config', 'ekf_localization.yaml'),
    #     {'use_sim_time': use_sim_time},
    # ],
    # )

    # # this assume installation of m-explore-ros2
    # # it subscribes to /map and publishes /goal_pose
    # explore_node = Node(
    # package='explore_lite',
    # executable='explore',
    # name='explore',
    # output='screen',
    # parameters=[{
    #     'use_sim_time': use_sim_time,
    #     'costmap_topic': '/global_costmap/costmap',
    #     'map_topic': '/map',
    #     'goal_topic': '/goal_pose',
    #     'visualize': True
    # }]
    # )

    return [
        depth_to_scan_node,
        slam_toolbox_node,          # create map
        nav2_bringup,               # start nav stack
        initial_mover_node,         # move robot into map
        # apriltag_node,            # detect tags that may be in view
        # tag_localization_node,    # integrate tag-based pose
        # ekf_node,                 # fuse localization
        # explore_node,             # frontier explore
        rviz_node
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time', default_value='false',
            description='Use simulation clock (requires /clock topic)'),
        DeclareLaunchArgument(
            'sim', default_value='false',
            description='Running in simulation (uses nav depth topics with correct geometry)'),
        DeclareLaunchArgument(
            'scan_source', default_value='depth',
            description='"depth" uses depthimage_to_laserscan, "lidar" assumes /scan published externally'),
        DeclareLaunchArgument(
            'use_rviz', default_value='true',
            description='Launch RViz with navigation config'),
        OpaqueFunction(function=launch_setup),
    ])
