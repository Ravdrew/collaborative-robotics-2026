"""
Frontier exploration mode for navigation.

This launch owns /cmd_nav via frontier_explorer.
Do not run together with nav_task_fsm.launch.py, which also commands /cmd_nav.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


# Build exploration-mode launch graph.
def generate_launch_description():
    pkg_bringup = get_package_share_directory("tidybot_bringup")

    # Define RViz launch argument.
    declare_use_rviz = DeclareLaunchArgument(
        "use_rviz",
        default_value="true",
        description="Launch RViz for visualization",
    )
    # Define MuJoCo viewer launch argument.
    declare_show_viewer = DeclareLaunchArgument(
        "show_mujoco_viewer",
        default_value="true",
        description="Show MuJoCo viewer window",
    )

    # Configure simulation launch arguments.
    sim_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_bringup, "launch", "sim.launch.py")
        ),
        launch_arguments={
            "scene": "our_scene_frontier_explore.xml",
            "use_rviz": LaunchConfiguration("use_rviz"),
            "show_mujoco_viewer": LaunchConfiguration("show_mujoco_viewer"),
            "use_sim_time": "true",
            "use_motion_planner": "false",
        }.items(),
    )

    # Configure state estimator parameters.
    state_estimator = Node(
        package="tidybot_bringup",
        executable="our_base_state_estimator.py",
        name="base_state_estimator",
        output="screen",
        parameters=[
            {
                "use_sim_time": True,
                "use_imu": False,
                "use_vision": False,
            }
        ],
    )

    # Configure depth to scan conversion parameters.
    depthimage_to_laserscan = Node(
        package="depthimage_to_laserscan",
        executable="depthimage_to_laserscan_node",
        name="depthimage_to_laserscan",
        output="screen",
        parameters=[
            {
                "use_sim_time": True,
                "scan_height": 5,
                "range_min": 0.15,
                "range_max": 4.5,
            }
        ],
        remappings=[
            ("image", "/camera/depth/image_raw"),
            ("camera_info", "/camera/depth/camera_info"),
            ("scan", "/scan"),
        ],
    )

    # Configure SLAM toolbox mapping parameters.
    slam_toolbox = Node(
        package="slam_toolbox",
        executable="async_slam_toolbox_node",
        name="slam_toolbox",
        output="screen",
        parameters=[
            {
                "use_sim_time": True,
                "odom_frame": "odom",
                "map_frame": "map",
                "base_frame": "base_link",
                "scan_topic": "/scan",
                "mode": "mapping",
                "resolution": 0.05,
                "max_laser_range": 4.5,
                "minimum_travel_distance": 0.1,
                "minimum_travel_heading": 0.2,
                "map_update_interval": 2.0,
                "transform_publish_period": 0.02,
                "transform_timeout": 0.2,
                "tf_buffer_duration": 30.0,
                "stack_size_to_use": 40000000,
                "do_loop_closing": True,
            }
        ],
    )

    # Start frontier goal generator node.
    frontier_explorer = Node(
        package="tidybot_bringup",
        executable="our_frontier_exploration.py",
        name="frontier_explorer",
        output="screen",
    )

    # Start A* navigation execution node.
    astar_navigator = Node(
        package="tidybot_bringup",
        executable="our_astar_navigator.py",
        name="astar_navigator",
        output="screen",
    )

    return LaunchDescription(
        [
            declare_use_rviz,
            declare_show_viewer,
            sim_launch,
            state_estimator,
            depthimage_to_laserscan,
            slam_toolbox,
            frontier_explorer,
            astar_navigator,
        ]
    )
