"""
Nav2 navigation launch for TidyBot2.

This is an OVERLAY launch file — run it on top of the existing sim or real launch.
It starts depthimage_to_laserscan, SLAM Toolbox, and the full Nav2 stack.

Usage:
  # Simulation (on top of sim.launch.py)
  ros2 launch tidybot_navigation navigation.launch.py sim:=true

  # Real hardware with depth camera (on top of real.launch.py)
  ros2 launch tidybot_navigation navigation.launch.py

  # Real hardware with LiDAR only (skip depthimage_to_laserscan)
  ros2 launch tidybot_navigation navigation.launch.py scan_source:=lidar

  # Real hardware with both LiDAR + depth camera
  ros2 launch tidybot_navigation navigation.launch.py scan_source:=both

scan_source modes:
  depth — depthimage_to_laserscan publishes /scan (SLAM + costmaps use depth)
  lidar — LiDAR driver publishes /scan externally (SLAM + costmaps use LiDAR)
  both  — LiDAR publishes /scan (used by SLAM), depth publishes /scan_depth
          (costmaps use both /scan and /scan_depth)

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
    use_ground_truth = LaunchConfiguration('use_ground_truth').perform(context) == 'true'

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

    # depthimage_to_laserscan — when scan_source:=depth (publishes /scan)
    depth_to_scan_node = Node(
        condition=LaunchConfigurationEquals('scan_source', 'depth'),
        package='depthimage_to_laserscan',
        executable='depthimage_to_laserscan_node',
        name='depthimage_to_laserscan',
        parameters=[{
            'use_sim_time': use_sim_time,
            'scan_time': 0.033,
            'range_min': 0.28,
            'range_max': 3.0,
            'scan_height': 100,
            'output_frame': scan_output_frame,
        }],
        remappings=[
            ('depth', depth_topic),
            ('depth_camera_info', depth_info_topic),
            ('scan', '/scan'),
        ],
    )

    # depthimage_to_laserscan — when scan_source:=both (publishes /scan_depth)
    # LiDAR owns /scan for SLAM; depth camera adds /scan_depth for costmaps only
    depth_to_scan_depth_node = Node(
        condition=LaunchConfigurationEquals('scan_source', 'both'),
        package='depthimage_to_laserscan',
        executable='depthimage_to_laserscan_node',
        name='depthimage_to_laserscan',
        parameters=[{
            'use_sim_time': use_sim_time,
            'scan_time': 0.033,
            'range_min': 0.28,
            'range_max': 3.0,
            'scan_height': 100,
            'output_frame': scan_output_frame,
        }],
        remappings=[
            ('depth', depth_topic),
            ('depth_camera_info', depth_info_topic),
            ('scan', '/scan_depth'),
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

    # Ground truth localization: static identity map→odom (bypasses SLAM)
    ground_truth_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='ground_truth_map_odom',
        arguments=['0', '0', '0', '0', '0', '0', 'map', 'odom'],
    )

    # Nav2 bringup (controller, planner, behavior, bt_navigator, etc.)
    nav2_bringup = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(nav2_bringup_dir, 'launch', 'navigation_launch.py')
        ),
        launch_arguments={
            'use_sim_time': use_sim_time,
            'params_file': os.path.join(pkg_dir, 'config', 'nav2_params.yaml'),
            'default_bt_xml_filename': os.path.join(pkg_dir, 'config', 'navigate_to_pose.xml'),
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

    # # this calls explore_lite as a frontier explorer that runs on startup
    # explore_node = Node(
    # package='explore_lite',
    # executable='explore',
    # name='explore',
    # parameters=[{
    #     'use_sim_time': use_sim_time,
    #     # # Tunable parameters
    #     # 'min_frontier_size': 0.5,
    #     # 'planner_frequency': 1.0,
    #     # 'clearing_rotation_allowed': True,
    # }],
    # )

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

    # Use ground truth (static map=odom) or SLAM for localization
    localization_node = ground_truth_tf if use_ground_truth else slam_toolbox_node

    return [
        depth_to_scan_node,
        depth_to_scan_depth_node,
        localization_node,
        nav2_bringup,
        rviz_node,
        # explore_node,
        # apriltag_node,
        # tag_localization_node,
        # ekf_node,
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
            'scan_source', default_value='both',
            description='"depth" uses depthimage_to_laserscan→/scan, "lidar" expects /scan from driver, "both" uses LiDAR→/scan + depth→/scan_depth'),
        DeclareLaunchArgument(
            'use_rviz', default_value='true',
            description='Launch RViz with navigation config'),
        DeclareLaunchArgument(
            'use_ground_truth', default_value='false',
            description='Use ground truth localization (static map=odom) instead of SLAM'),
        OpaqueFunction(function=launch_setup),
    ])
