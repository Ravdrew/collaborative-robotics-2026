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

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.conditions import IfCondition, LaunchConfigurationEquals
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def launch_setup(context, *args, **kwargs):
    pkg_dir = get_package_share_directory('tidybot_navigation')
    nav2_bringup_dir = get_package_share_directory('nav2_bringup')

    use_sim_time = LaunchConfiguration('use_sim_time')
    use_rviz = LaunchConfiguration('use_rviz')
    is_sim = LaunchConfiguration('sim').perform(context) == 'true'

    # In simulation, use un-flipped depth on nav topics (camera_link frame)
    # On real hardware, use standard RealSense topics
    # Both use camera_link as output frame — it's always in TF (from URDF)
    # and has the correct orientation for 2D SLAM (X-forward, Y-left, Z-up)
    if is_sim:
        depth_topic = '/camera/depth/image_nav'
        depth_info_topic = '/camera/depth/camera_info_nav'
    else:
        depth_topic = '/camera/depth/image_raw'
        depth_info_topic = '/camera/depth/camera_info'
    scan_output_frame = 'camera_link'

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

    # RViz with navigation config
    rviz_node = Node(
        condition=IfCondition(use_rviz),
        package='rviz2',
        executable='rviz2',
        name='rviz2_nav',
        arguments=['-d', os.path.join(pkg_dir, 'rviz', 'navigation.rviz')],
        parameters=[{'use_sim_time': use_sim_time}],
    )

    return [
        depth_to_scan_node,
        slam_toolbox_node,
        nav2_bringup,
        rviz_node,
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
