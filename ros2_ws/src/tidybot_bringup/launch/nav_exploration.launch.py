"""
Navigation Exploration Launch File for TidyBot2.

Brings up the full autonomous frontier exploration stack in one command:
  1. MuJoCo simulation  (our_scene_frontier_explore.xml)
  2. Base state estimator  (/odom → /odom_est_pose)
  3. depthimage_to_laserscan  (depth camera → /scan for SLAM)
  4. slam_toolbox  (/scan → /map, built incrementally as robot explores)
  5. Frontier explorer  (/map + /odom_est_pose → /cmd_nav goals)
  6. A* navigator  (/cmd_nav + /map + /odom → /cmd_vel)

Usage:
    ros2 launch tidybot_bringup nav_exploration.launch.py
    ros2 launch tidybot_bringup nav_exploration.launch.py use_rviz:=false
    ros2 launch tidybot_bringup nav_exploration.launch.py show_mujoco_viewer:=false

Verify it is working (in a second terminal):
    ros2 topic list                        # /map, /scan, /cmd_nav, /cmd_vel should appear
    ros2 topic hz /map                     # map publishes every ~2 s as robot moves
    ros2 topic echo /cmd_nav               # frontier goals published here
    ros2 topic echo /nav_success           # True each time a goal is reached

Manually send a goal to test the navigator without frontier explorer:
    ros2 topic pub /cmd_nav geometry_msgs/msg/Pose2D \
        "{x: 2.0, y: 2.0, theta: 0.0}" --once

Check the depth camera TF frame name if /scan is not appearing:
    ros2 topic echo /camera/depth/camera_info --field header.frame_id
    ros2 run tf2_tools view_frames
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_bringup = get_package_share_directory('tidybot_bringup')

    # -------------------------------------------------------------------------
    # Arguments
    # -------------------------------------------------------------------------
    declare_use_rviz = DeclareLaunchArgument(
        'use_rviz', default_value='true',
        description='Launch RViz for visualization',
    )
    declare_show_viewer = DeclareLaunchArgument(
        'show_mujoco_viewer', default_value='true',
        description='Show the interactive MuJoCo viewer window',
    )

    # -------------------------------------------------------------------------
    # 1. MuJoCo simulation (robot, physics, camera, /odom)
    #    use_motion_planner:=false — arm IK not needed for base navigation
    # -------------------------------------------------------------------------
    sim_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_bringup, 'launch', 'sim.launch.py')
        ),
        launch_arguments={
            'scene':              'our_scene_frontier_explore.xml',
            'use_rviz':           LaunchConfiguration('use_rviz'),
            'show_mujoco_viewer': LaunchConfiguration('show_mujoco_viewer'),
            'use_sim_time':       'true',
            'use_motion_planner': 'false',
        }.items(),
    )

    # -------------------------------------------------------------------------
    # 2. Base state estimator
    #    Reads /odom, publishes /odom_est_pose (Pose2D) for frontier explorer.
    #    use_imu/vision disabled — neither is available in the basic sim scene.
    # -------------------------------------------------------------------------
    state_estimator = Node(
        package='tidybot_bringup',
        executable='our_base_state_estimator.py',
        name='base_state_estimator',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'use_imu':      False,
            'use_vision':   False,
        }],
    )

    # -------------------------------------------------------------------------
    # 3. depthimage_to_laserscan
    #    Converts the RealSense depth image into a virtual 2-D laser scan so
    #    slam_toolbox can build an occupancy map.
    #
    #    scan_height  – pixel rows averaged into the scan (5 is a good default).
    #    range_min/max – room is 10 m × 10 m, walls at ±5 m; cap at 4.5 m.
    #
    #    output_frame is intentionally left unset so the scan inherits the
    #    camera's own frame_id from camera_info, which is already in the TF
    #    tree published by robot_state_publisher.
    # -------------------------------------------------------------------------
    depthimage_to_laserscan = Node(
        package='depthimage_to_laserscan',
        executable='depthimage_to_laserscan_node',
        name='depthimage_to_laserscan',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'scan_height':  5,
            'range_min':    0.15,
            'range_max':    4.5,
        }],
        remappings=[
            ('image',       '/camera/depth/image_raw'),
            ('camera_info', '/camera/depth/camera_info'),
            ('scan',        '/scan'),
        ],
    )

    # -------------------------------------------------------------------------
    # 4. SLAM Toolbox (online async mapping)
    #    Subscribes to /scan + TF, builds /map incrementally.
    #    Also publishes the map→odom TF.
    # -------------------------------------------------------------------------
    slam_toolbox = Node(
        package='slam_toolbox',
        executable='async_slam_toolbox_node',
        name='slam_toolbox',
        output='screen',
        parameters=[{
            'use_sim_time':              True,
            'odom_frame':                'odom',
            'map_frame':                 'map',
            'base_frame':                'base_link',
            'scan_topic':                '/scan',
            'mode':                      'mapping',
            'resolution':                0.05,   # 5 cm/cell; room fits in 200×200 grid
            'max_laser_range':           4.5,
            'minimum_travel_distance':   0.1,    # update map every 10 cm of motion
            'minimum_travel_heading':    0.2,    # or every ~11° of turning
            'map_update_interval':       2.0,    # seconds between /map publishes
            'transform_publish_period':  0.02,
            'transform_timeout':         0.2,
            'tf_buffer_duration':        30.0,
            'stack_size_to_use':         40000000,
            'do_loop_closing':           True,
        }],
    )

    # -------------------------------------------------------------------------
    # 5. Frontier Explorer
    #    Subscribes: /odom_est_pose, /map, /nav_success
    #    Publishes:  /cmd_nav (next frontier goal as Pose2D)
    # -------------------------------------------------------------------------
    frontier_explorer = Node(
        package='tidybot_bringup',
        executable='our_frontier_exploration.py',
        name='frontier_explorer',
        output='screen',
    )

    # -------------------------------------------------------------------------
    # 6. A* Navigator
    #    Subscribes: /cmd_nav, /map, /odom
    #    Publishes:  /cmd_vel, /nav_success, /end_nav_pose
    # -------------------------------------------------------------------------
    astar_navigator = Node(
        package='tidybot_bringup',
        executable='our_astar_navigator.py',
        name='astar_navigator',
        output='screen',
    )

    return LaunchDescription([
        declare_use_rviz,
        declare_show_viewer,
        sim_launch,
        state_estimator,
        depthimage_to_laserscan,
        slam_toolbox,
        frontier_explorer,
        astar_navigator,
    ])
