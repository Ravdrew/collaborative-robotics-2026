"""
Nav2 navigation launch for TidyBot2.

This is an OVERLAY launch file — run it on top of the existing sim or real launch.
It starts depthimage_to_laserscan, SLAM Toolbox, and the full Nav2 stack.

Usage:
  # Simulation (on top of sim.launch.py)
  ros2 launch tidybot_navigation navigation.launch.py use_sim_time:=true

  # Real hardware with depth camera (on top of real.launch.py)
  ros2 launch tidybot_navigation navigation.launch.py

  # Real hardware with LiDAR (skip depthimage_to_laserscan)
  ros2 launch tidybot_navigation navigation.launch.py scan_source:=lidar

NOTE: The D435 is on a pan-tilt mount. Keep camera at pan=0, tilt=0 during
navigation, otherwise the laser scan plane rotates and corrupts the costmap.
This limitation goes away once a real LiDAR is available (scan_source:=lidar).
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, GroupAction
from launch.conditions import IfCondition, LaunchConfigurationEquals
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node


def generate_launch_description():
    pkg_dir = get_package_share_directory('tidybot_navigation')
    nav2_bringup_dir = get_package_share_directory('nav2_bringup')

    # Launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time')
    scan_source = LaunchConfiguration('scan_source')
    use_rviz = LaunchConfiguration('use_rviz')

    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time', default_value='false',
        description='Use simulation clock (requires /clock topic to be published)')

    declare_scan_source = DeclareLaunchArgument(
        'scan_source', default_value='depth',
        description='Scan source: "depth" uses depthimage_to_laserscan, "lidar" assumes /scan is published externally')

    declare_use_rviz = DeclareLaunchArgument(
        'use_rviz', default_value='true',
        description='Launch RViz with navigation config')

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
            'scan_height': 100,      # rows around image center to consider
            'output_frame': 'camera_link',
        }],
        remappings=[
            ('depth', '/camera/depth/image_nav'),
            ('depth_camera_info', '/camera/depth/camera_info_nav'),
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

    # RViz with navigation config
    rviz_node = Node(
        condition=IfCondition(use_rviz),
        package='rviz2',
        executable='rviz2',
        name='rviz2_nav',
        arguments=['-d', os.path.join(pkg_dir, 'rviz', 'navigation.rviz')],
        parameters=[{'use_sim_time': use_sim_time}],
    )

    return LaunchDescription([
        declare_use_sim_time,
        declare_scan_source,
        declare_use_rviz,
        depth_to_scan_node,
        slam_toolbox_node,
        nav2_bringup,
        rviz_node,
    ])
