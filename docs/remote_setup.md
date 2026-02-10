# TidyBot2 Remote Control Setup

This guide explains how to control TidyBot2 remotely over the network using native ROS2.

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────────┐
│                       MINI PC (TidyBot2 Robot)                          │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │                      ROS2 Network (DDS)                            │  │
│  │  /cmd_vel  /joint_states  /odom  /camera/*  /right_arm/joint_cmd  │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│        ▲              ▲              ▲              ▲          ▲        │
│        │              │              │              │          │        │
│  ┌─────┴─────┐  ┌─────┴─────┐  ┌─────┴─────┐  ┌─────┴─────┐ ┌┴──────┐│
│  │ Phoenix6  │  │ Interbotix│  │ RealSense │  │ Pan-Tilt  │ │Micro- ││
│  │ Base Node │  │ xs_sdk x2 │  │ Camera    │  │ (on U2D2) │ │phone  ││
│  └───────────┘  └───────────┘  └───────────┘  └───────────┘ └────────┘│
│                       │                                                 │
│              ┌────────┴────────┐                                        │
│              │  arm_wrapper    │  (sim-compatible topic translation)     │
│              │  gripper_wrapper│                                        │
│              └─────────────────┘                                        │
└──────────────────────────────────────────────────────────────────────────┘
┌──────────────────────────────────────────────────────────────────────────┐
│                       MINI PC (TidyBot2 Robot)                          │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │                      ROS2 Network (DDS)                            │  │
│  │  /cmd_vel  /joint_states  /odom  /camera/*  /right_arm/joint_cmd  │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│        ▲              ▲              ▲              ▲          ▲        │
│        │              │              │              │          │        │
│  ┌─────┴─────┐  ┌─────┴─────┐  ┌─────┴─────┐  ┌─────┴─────┐ ┌┴──────┐│
│  │ Phoenix6  │  │ Interbotix│  │ RealSense │  │ Pan-Tilt  │ │Micro- ││
│  │ Base Node │  │ xs_sdk x2 │  │ Camera    │  │ (on U2D2) │ │phone  ││
│  └───────────┘  └───────────┘  └───────────┘  └───────────┘ └────────┘│
│                       │                                                 │
│              ┌────────┴────────┐                                        │
│              │  arm_wrapper    │  (sim-compatible topic translation)     │
│              │  gripper_wrapper│                                        │
│              └─────────────────┘                                        │
└──────────────────────────────────────────────────────────────────────────┘
                              │
                     WiFi / Ethernet / Tailscale VPN
                              │
┌──────────────────────────────────────────────────────────────────────────┐
│                    YOUR MACHINE (Remote Client)                          │
│                                                                          │
│  ROS2 Humble + Same Domain ID = Direct topic access                       │
│                                                                          │
│  ros2 topic pub /cmd_vel geometry_msgs/msg/Twist ...                     │
│  ros2 topic echo /joint_states                                           │
│  python3 my_robot_control.py  ← same code works in sim and real!         │
└──────────────────────────────────────────────────────────────────────────┘
┌──────────────────────────────────────────────────────────────────────────┐
│                    YOUR MACHINE (Remote Client)                          │
│                                                                          │
│  ROS2 Humble + Same Domain ID = Direct topic access                       │
│                                                                          │
│  ros2 topic pub /cmd_vel geometry_msgs/msg/Twist ...                     │
│  ros2 topic echo /joint_states                                           │
│  python3 my_robot_control.py  ← same code works in sim and real!         │
└──────────────────────────────────────────────────────────────────────────┘
```

## Key Principle: Sim-to-Real Topic Compatibility
## Key Principle: Sim-to-Real Topic Compatibility

You run **real ROS2 nodes** on your machine, not a custom client library. The robot uses **sim-compatible topic wrappers** (enabled by default) so that:
- The **same topics and code** work in both simulation (`sim.launch.py`) and real hardware (`real.launch.py`)
- Standard ROS2 tools work (`ros2 topic`, `ros2 run`, `rviz2`)
You run **real ROS2 nodes** on your machine, not a custom client library. The robot uses **sim-compatible topic wrappers** (enabled by default) so that:
- The **same topics and code** work in both simulation (`sim.launch.py`) and real hardware (`real.launch.py`)
- Standard ROS2 tools work (`ros2 topic`, `ros2 run`, `rviz2`)
- Skills transfer to any ROS2 robot

**Simulation path:**
```
Your code → /right_arm/joint_cmd → MuJoCo bridge → physics sim → /joint_states
```

**Real hardware path (use_sim_topics:=true, default):**
```
Your code → /right_arm/joint_cmd → arm_wrapper (velocity limiting) → xs_sdk → Dynamixel servos
```

**Simulation path:**
```
Your code → /right_arm/joint_cmd → MuJoCo bridge → physics sim → /joint_states
```

**Real hardware path (use_sim_topics:=true, default):**
```
Your code → /right_arm/joint_cmd → arm_wrapper (velocity limiting) → xs_sdk → Dynamixel servos
```

## Prerequisites

### On Your Machine (Client)

1. **ROS2 Humble installed** (required)
   ```bash
   # Ubuntu 22.04
   sudo apt install ros-humble-desktop
   ```

2. **tidybot_msgs package** (for custom message types)
   ```bash
   cd ~/ros2_ws/src
   git clone https://github.com/Stanford-ARM-Lab/collaborative-robotics-2026.git
   cd ~/ros2_ws
   colcon build --packages-select tidybot_msgs
   source install/setup.bash
   ```

3. **Cyclone DDS** (recommended for easier networking)
   ```bash
   sudo apt install ros-humble-rmw-cyclonedds-cpp
   ```

### Network Requirements

- Client and robot on the **same network** (same WiFi or Ethernet subnet)
- **OR** connected via Tailscale VPN (see Option D below for remote access from anywhere)
- If multicast is blocked, use FastDDS Discovery Server mode (see below)

## Quick Start

### Step 1: Start Robot (on TidyBot2 mini PC)

```bash
# SSH into robot
ssh locobot@192.168.0.207
# NUC IP address may change, check with ifconfig if not accessible
ssh locobot@192.168.0.207
# NUC IP address may change, check with ifconfig if not accessible
# ssh password: locobot

# Set up environment (sources ROS2, sets PYTHONPATH, builds workspace,
# sets ROS_DOMAIN_ID=42 and RMW_IMPLEMENTATION=rmw_cyclonedds_cpp)
# Set up environment (sources ROS2, sets PYTHONPATH, builds workspace,
# sets ROS_DOMAIN_ID=42 and RMW_IMPLEMENTATION=rmw_cyclonedds_cpp)
cd ~/collaborative-robotics-2026/ros2_ws
source setup_env.bash

# Launch robot (all hardware enabled by default)
ros2 launch tidybot_bringup real.launch.py

# To use IK controller
ros2 launch tidybot_bringup real.launch.py use_planner:=true

# Launch robot (all hardware enabled by default)
ros2 launch tidybot_bringup real.launch.py

# To use IK controller
ros2 launch tidybot_bringup real.launch.py use_planner:=true
```

### Step 2: Connect Client (on your machine)

```bash
# Set up environment (must match robot!)
export ROS_DOMAIN_ID=42
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

# Source your workspace
source ~/ros2_ws/install/setup.bash

# Test connection
ros2 topic list
```

You should see topics like:
```
/cmd_vel
/odom
/joint_states
/right_arm/joint_cmd
/left_arm/joint_cmd
/right_gripper/cmd
/left_gripper/cmd
/right_arm/joint_cmd
/left_arm/joint_cmd
/right_gripper/cmd
/left_gripper/cmd
/camera/color/image_raw
/camera/depth/image_raw
/camera/pan_tilt_cmd
/camera/depth/image_raw
/camera/pan_tilt_cmd
...
```

If you don't see ros2 topics, try disabling your local network firewall (as well as on the NUC):

```
sudo ufw disable
```

### Step 3: Control the Robot

```bash
# Move the base forward
ros2 topic pub /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.1}}" -r 10

# Read joint states
ros2 topic echo /joint_states

# Move right arm (6 joint positions as Float64MultiArray)
ros2 topic pub /right_arm/joint_cmd std_msgs/msg/Float64MultiArray "{data: [0.0, -0.5, 0.5, 0.0, 0.0, 0.0]}" --once

# Open right gripper (Float64MultiArray, 0.0 = open, 1.0 = closed)
ros2 topic pub /right_gripper/cmd std_msgs/msg/Float64MultiArray "{data: [0.0]}" --once
```

## Launch File Reference

### `real.launch.py` — Real Hardware

| Argument | Default | Description |
|----------|---------|-------------|
| `use_rviz` | `true` | Launch RViz for visualization |
| `use_base` | `true` | Phoenix 6 mobile base driver |
| `use_arms` | `true` | Interbotix arm drivers (dual U2D2) |
| `use_left_arm` | `true` | Left arm on U2D2 #2 |
| `use_pan_tilt` | `true` | Pan-tilt on U2D2 #1 (with right arm) |
| `use_camera` | `true` | RealSense D435 camera |
| `use_compression` | `false` | Image compression for remote clients |
| `use_microphone` | `true` | Microphone recording node |
| `use_planner` | `false` | IK motion planner (Pinocchio-based) |
| `use_sim_topics` | `true` | Sim-compatible topic wrappers |
| `load_configs` | `true` | Load motor configs from YAML files |

### `sim.launch.py` — MuJoCo Simulation

| Argument | Default | Description |
|----------|---------|-------------|
| `scene` | `scene_wx250s_bimanual.xml` | MuJoCo scene file |
| `use_rviz` | `true` | Launch RViz |
| `show_mujoco_viewer` | `true` | Show MuJoCo viewer window |
| `use_sim_time` | `true` | Use simulation time |
| `use_motion_planner` | `true` | Launch IK motion planner (Mink-based) |
# Move right arm (6 joint positions as Float64MultiArray)
ros2 topic pub /right_arm/joint_cmd std_msgs/msg/Float64MultiArray "{data: [0.0, -0.5, 0.5, 0.0, 0.0, 0.0]}" --once

# Open right gripper (Float64MultiArray, 0.0 = open, 1.0 = closed)
ros2 topic pub /right_gripper/cmd std_msgs/msg/Float64MultiArray "{data: [0.0]}" --once
```

## Launch File Reference

### `real.launch.py` — Real Hardware

| Argument | Default | Description |
|----------|---------|-------------|
| `use_rviz` | `true` | Launch RViz for visualization |
| `use_base` | `true` | Phoenix 6 mobile base driver |
| `use_arms` | `true` | Interbotix arm drivers (dual U2D2) |
| `use_left_arm` | `true` | Left arm on U2D2 #2 |
| `use_pan_tilt` | `true` | Pan-tilt on U2D2 #1 (with right arm) |
| `use_camera` | `true` | RealSense D435 camera |
| `use_compression` | `false` | Image compression for remote clients |
| `use_microphone` | `true` | Microphone recording node |
| `use_planner` | `false` | IK motion planner (Pinocchio-based) |
| `use_sim_topics` | `true` | Sim-compatible topic wrappers |
| `load_configs` | `true` | Load motor configs from YAML files |

### `sim.launch.py` — MuJoCo Simulation

| Argument | Default | Description |
|----------|---------|-------------|
| `scene` | `scene_wx250s_bimanual.xml` | MuJoCo scene file |
| `use_rviz` | `true` | Launch RViz |
| `show_mujoco_viewer` | `true` | Show MuJoCo viewer window |
| `use_sim_time` | `true` | Use simulation time |
| `use_motion_planner` | `true` | Launch IK motion planner (Mink-based) |

## Network Configuration Options

### Option A: Cyclone DDS with Multicast (Simplest)

Works when robot and client are on the same subnet and multicast is allowed.

**Robot:**
```bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_DOMAIN_ID=42
```

**Client:**
```bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_DOMAIN_ID=42
```

### Option B: FastDDS Discovery Server (Works Across Subnets)

Use when multicast is blocked or machines are on different subnets.

**Robot (start discovery server first):**
```bash
# Terminal 1: Start discovery server
fastdds discovery --server-id 0 --port 11811

# Terminal 2: Launch robot
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export FASTRTPS_DEFAULT_PROFILES_FILE=$(ros2 pkg prefix tidybot_bringup)/share/tidybot_bringup/config/fastdds_robot.xml
export ROS_DOMAIN_ID=42
ros2 launch tidybot_bringup real.launch.py
ros2 launch tidybot_bringup real.launch.py
```

**Client:**
```bash
# Edit config to set robot IP
# In fastdds_client.xml, replace 192.168.1.100 with actual robot IP

export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export FASTRTPS_DEFAULT_PROFILES_FILE=<path_to>/fastdds_client.xml
export ROS_DOMAIN_ID=42
```

### Option C: Zenoh Bridge (Enterprise-Grade)

For complex network topologies. See [Zenoh documentation](https://zenoh.io/docs/getting-started/first-app/).

### Option D: Tailscale VPN (Remote Access from Anywhere)

Use Tailscale to securely connect to the robot from anywhere, even across different networks or from outside your lab/building. This is the easiest way to get remote access without port forwarding or VPN configuration.

#### Why Use Tailscale?

- **Access from anywhere**: Control the robot from home, coffee shops, or different networks
- **No port forwarding needed**: Works through NAT and firewalls automatically
- **Encrypted connection**: All traffic is secured via WireGuard
- **Simple setup**: Just install and authenticate on both machines
- **Direct peer-to-peer**: Low latency compared to traditional VPNs
- **Works with ROS2**: DDS multicast works over Tailscale network

#### Installation

**On the Robot (TidyBot2 mini PC):**

```bash
# SSH into the robot
ssh locobot@192.168.0.207

# Install Tailscale
curl -fsSL https://tailscale.com/install.sh | sh

# Start Tailscale and authenticate
sudo tailscale up

# Follow the link in terminal to authenticate with your Tailscale account
# The robot will now appear in your Tailscale network
```

**On Your Client Machine:**

For **Linux** or**Windows WSL**:
```bash
# Install Tailscale
curl -fsSL https://tailscale.com/install.sh | sh

# Start and authenticate
sudo tailscale up
```

For **macOS**:
```bash
# Download from https://tailscale.com/download/mac
# Or install via Homebrew (CLI-only):
brew install tailscale
sudo brew services start tailscale
sudo tailscale up
```

For **Windows** (if using WSL2):
- Install Tailscale on Windows (not inside WSL)
- Download from https://tailscale.com/download/windows
- WSL2 will automatically have access to the Tailscale network

#### Usage with ROS2

Once both machines are connected to Tailscale:

1. **Find the robot's Tailscale IP:**
   ```bash
   # On the robot
   tailscale ip -4
   # Example output: 100.101.102.103
   ```

2. **Connect from your client using Tailscale IP:**
   ```bash
   # SSH via Tailscale (works from anywhere!)
   ssh locobot@100.101.102.103

   # Or use the Tailscale machine name
   ssh locobot@tidybot2-nuc
   ```

3. **Configure ROS2 environment:**
   ```bash
   # On both robot and client
   export ROS_DOMAIN_ID=42
   export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

   # Source your workspace
   source ~/ros2_ws/install/setup.bash
   ```

4. **ROS2 should now work over Tailscale:**
   ```bash
   # Test from your client machine
   ros2 topic list
   ros2 topic echo /joint_states
   ```

#### Cyclone DDS Configuration for Tailscale

If multicast doesn't work automatically over Tailscale, create a Cyclone DDS config file:

**cyclone_tailscale.xml:**
```xml
<?xml version="1.0" encoding="UTF-8" ?>
<CycloneDDS xmlns="https://cdds.io/config">
  <Domain id="any">
    <General>
      <NetworkInterfaceAddress>tailscale0</NetworkInterfaceAddress>
    </General>
  </Domain>
</CycloneDDS>
```

Then use it:
```bash
export CYCLONEDDS_URI=file:///path/to/cyclone_tailscale.xml
```

#### Tailscale Advantages for Robotics

- **No firewall configuration needed**: Tailscale handles NAT traversal
- **Mobile access**: Control robot from your phone or laptop anywhere
- **Team access**: Share access to the robot with teammates easily
- **SSH simplification**: No need to remember changing local IPs
- **Image compression recommended**: Enable `use_compression:=true` for camera topics over Tailscale

#### Tailscale Tips

- **Persistent connection**: Tailscale stays connected across reboots (if configured)
- **Exit nodes**: Can use your lab network as an exit node for full network access
- **Subnet routing**: Can expose the entire robot's local network if needed
- **MagicDNS**: Use machine names instead of IPs (e.g., `tidybot2-nuc`)

## ROS2 Topics Reference

### Mobile Base

| Topic | Type | Description |
|-------|------|-------------|
| `/cmd_vel` | `geometry_msgs/Twist` | Velocity commands (linear.x, linear.y, angular.z) |
| `/base/target_pose` | `geometry_msgs/Pose2D` | Go-to-goal position (x, y, theta) |
| `/odom` | `nav_msgs/Odometry` | Odometry feedback |
| `/base/goal_reached` | `std_msgs/Bool` | True when target pose reached |

### Arms (sim-compatible topics, `use_sim_topics:=true`)

These topics work identically in simulation and on real hardware:
### Arms (sim-compatible topics, `use_sim_topics:=true`)

These topics work identically in simulation and on real hardware:

| Topic | Type | Description |
|-------|------|-------------|
| `/right_arm/joint_cmd` | `std_msgs/Float64MultiArray` | Right arm 6-joint positions [waist, shoulder, elbow, forearm_roll, wrist_angle, wrist_rotate] |
| `/left_arm/joint_cmd` | `std_msgs/Float64MultiArray` | Left arm 6-joint positions |
| `/right_gripper/cmd` | `std_msgs/Float64MultiArray` | Right gripper (0.0=open, 1.0=closed) |
| `/left_gripper/cmd` | `std_msgs/Float64MultiArray` | Left gripper |
| `/right_arm/command` | `tidybot_msgs/ArmCommand` | Right arm with duration (trajectory interpolation) |
| `/left_arm/command` | `tidybot_msgs/ArmCommand` | Left arm with duration |
| `/joint_states` | `sensor_msgs/JointState` | All joint positions/velocities (aggregated) |
| `/right_arm/joint_cmd` | `std_msgs/Float64MultiArray` | Right arm 6-joint positions [waist, shoulder, elbow, forearm_roll, wrist_angle, wrist_rotate] |
| `/left_arm/joint_cmd` | `std_msgs/Float64MultiArray` | Left arm 6-joint positions |
| `/right_gripper/cmd` | `std_msgs/Float64MultiArray` | Right gripper (0.0=open, 1.0=closed) |
| `/left_gripper/cmd` | `std_msgs/Float64MultiArray` | Left gripper |
| `/right_arm/command` | `tidybot_msgs/ArmCommand` | Right arm with duration (trajectory interpolation) |
| `/left_arm/command` | `tidybot_msgs/ArmCommand` | Left arm with duration |
| `/joint_states` | `sensor_msgs/JointState` | All joint positions/velocities (aggregated) |

### Camera

| Topic | Type | Description |
|-------|------|-------------|
| `/camera/color/image_raw` | `sensor_msgs/Image` | RGB camera feed (640x480 @ 15fps) |
| `/camera/depth/image_raw` | `sensor_msgs/Image` | Depth camera feed (640x480 @ 15fps) |
| `/camera/color/camera_info` | `sensor_msgs/CameraInfo` | RGB camera intrinsics |
| `/camera/depth/camera_info` | `sensor_msgs/CameraInfo` | Depth camera intrinsics |
| `/camera/color/image_compressed` | `sensor_msgs/CompressedImage` | JPEG compressed RGB (requires `use_compression:=true`) |
| `/camera/depth/image_compressed` | `sensor_msgs/CompressedImage` | PNG compressed depth (requires `use_compression:=true`) |
| `/camera/color/image_raw` | `sensor_msgs/Image` | RGB camera feed (640x480 @ 15fps) |
| `/camera/depth/image_raw` | `sensor_msgs/Image` | Depth camera feed (640x480 @ 15fps) |
| `/camera/color/camera_info` | `sensor_msgs/CameraInfo` | RGB camera intrinsics |
| `/camera/depth/camera_info` | `sensor_msgs/CameraInfo` | Depth camera intrinsics |
| `/camera/color/image_compressed` | `sensor_msgs/CompressedImage` | JPEG compressed RGB (requires `use_compression:=true`) |
| `/camera/depth/image_compressed` | `sensor_msgs/CompressedImage` | PNG compressed depth (requires `use_compression:=true`) |
| `/camera/pan_tilt_cmd` | `std_msgs/Float64MultiArray` | Pan/tilt angles [pan, tilt] radians |
| `/camera/pan_tilt_state` | `sensor_msgs/JointState` | Current pan/tilt positions |

### Custom Messages (`tidybot_msgs`)

| Message | Fields | Description |
|---------|--------|-------------|
| `ArmCommand` | `header`, `float64[6] joint_positions`, `float64 duration` | Target arm pose with interpolation time |
| `ArmVelocityCommand` | `header`, `float64[6] joint_velocities`, `float64 duration` | Joint velocity command |
| `CartesianVelocityCommand` | `header`, `Twist twist`, `string frame_id`, `float64 duration` | End-effector velocity in task space |
| `GripperCommand` | `header`, `float64 position`, `float64 effort` | Gripper (0=open, 1=closed) with effort |
| `PanTilt` | `header`, `float64 pan`, `float64 tilt` | Pan-tilt angles in radians |

### Services

| Service | Type | Description |
|---------|------|-------------|
| `/plan_to_target` | `tidybot_msgs/PlanToTarget` | Plan and execute arm motion with collision/singularity checking |
| `/microphone/record` | `tidybot_msgs/AudioRecord` | Start/stop audio recording |
| `/camera/pan_tilt_state` | `sensor_msgs/JointState` | Current pan/tilt positions |

### Custom Messages (`tidybot_msgs`)

| Message | Fields | Description |
|---------|--------|-------------|
| `ArmCommand` | `header`, `float64[6] joint_positions`, `float64 duration` | Target arm pose with interpolation time |
| `ArmVelocityCommand` | `header`, `float64[6] joint_velocities`, `float64 duration` | Joint velocity command |
| `CartesianVelocityCommand` | `header`, `Twist twist`, `string frame_id`, `float64 duration` | End-effector velocity in task space |
| `GripperCommand` | `header`, `float64 position`, `float64 effort` | Gripper (0=open, 1=closed) with effort |
| `PanTilt` | `header`, `float64 pan`, `float64 tilt` | Pan-tilt angles in radians |

### Services

| Service | Type | Description |
|---------|------|-------------|
| `/plan_to_target` | `tidybot_msgs/PlanToTarget` | Plan and execute arm motion with collision/singularity checking |
| `/microphone/record` | `tidybot_msgs/AudioRecord` | Start/stop audio recording |

## Example: Python Control Script

Create a file `my_robot_control.py`:

```python
#!/usr/bin/env python3
"""
Example control script for TidyBot2.
Uses sim-compatible topics — works unchanged in both simulation and real hardware.
"""
"""
Example control script for TidyBot2.
Uses sim-compatible topics — works unchanged in both simulation and real hardware.
"""
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64MultiArray
from std_msgs.msg import Float64MultiArray

class MyRobotController(Node):
    def __init__(self):
        super().__init__('my_robot_controller')

        # Publishers (same topics work in sim and real!)
        self.base_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.right_arm_pub = self.create_publisher(Float64MultiArray, '/right_arm/joint_cmd', 10)
        self.right_gripper_pub = self.create_publisher(Float64MultiArray, '/right_gripper/cmd', 10)
        self.right_arm_pub = self.create_publisher(Float64MultiArray, '/right_arm/joint_cmd', 10)
        self.right_gripper_pub = self.create_publisher(Float64MultiArray, '/right_gripper/cmd', 10)

        # Timer for control loop
        self.create_timer(0.1, self.control_loop)

    def control_loop(self):
        # Move base forward
        vel = Twist()
        vel.linear.x = 0.1
        self.base_pub.publish(vel)

        # Move right arm (6 joints: waist, shoulder, elbow, forearm_roll, wrist_angle, wrist_rotate)
        arm_cmd = Float64MultiArray()
        arm_cmd.data = [0.0, -0.5, 0.5, 0.0, 0.0, 0.0]
        self.right_arm_pub.publish(arm_cmd)

        # Open right gripper
        gripper_cmd = Float64MultiArray()
        gripper_cmd.data = [0.0]  # 0.0 = open, 1.0 = closed
        self.right_gripper_pub.publish(gripper_cmd)
        # Move right arm (6 joints: waist, shoulder, elbow, forearm_roll, wrist_angle, wrist_rotate)
        arm_cmd = Float64MultiArray()
        arm_cmd.data = [0.0, -0.5, 0.5, 0.0, 0.0, 0.0]
        self.right_arm_pub.publish(arm_cmd)

        # Open right gripper
        gripper_cmd = Float64MultiArray()
        gripper_cmd.data = [0.0]  # 0.0 = open, 1.0 = closed
        self.right_gripper_pub.publish(gripper_cmd)

def main():
    rclpy.init()
    node = MyRobotController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

Run it:
```bash
# This SAME script works for simulation and real robot!
python3 my_robot_control.py
```

You can also use `tidybot_msgs` for trajectory-interpolated arm commands:

```python
from tidybot_msgs.msg import ArmCommand

# Publish to /right_arm/command for smooth interpolated motion
arm_cmd = ArmCommand()
arm_cmd.joint_positions = [0.0, -0.5, 0.5, 0.0, 0.0, 0.0]
arm_cmd.duration = 2.0  # seconds to reach target (cosine easing)
```

You can also use `tidybot_msgs` for trajectory-interpolated arm commands:

```python
from tidybot_msgs.msg import ArmCommand

# Publish to /right_arm/command for smooth interpolated motion
arm_cmd = ArmCommand()
arm_cmd.joint_positions = [0.0, -0.5, 0.5, 0.0, 0.0, 0.0]
arm_cmd.duration = 2.0  # seconds to reach target (cosine easing)
```

## Visualization with RViz

Launch RViz on your client machine to visualize the robot:

```bash
# Using the client launch file (includes local robot_state_publisher + RViz)
# Using the client launch file (includes local robot_state_publisher + RViz)
ros2 launch tidybot_client client_rviz.launch.py

# With a custom domain ID
ros2 launch tidybot_client client_rviz.launch.py domain_id:=42

# Or use the connect_robot launch (sets up env, optional TF)
ros2 launch tidybot_client connect_robot.launch.py robot_ip:=192.168.0.207 start_tf:=true

# With a custom domain ID
ros2 launch tidybot_client client_rviz.launch.py domain_id:=42

# Or use the connect_robot launch (sets up env, optional TF)
ros2 launch tidybot_client connect_robot.launch.py robot_ip:=192.168.0.207 start_tf:=true

# Or manually
rviz2 -d $(ros2 pkg prefix tidybot_bringup)/share/tidybot_bringup/rviz/tidybot.rviz
```

## Hardware Details

### Dual U2D2 Setup

| U2D2 Port | Components | Motor IDs |
|-----------|------------|-----------|
| `/dev/ttyUSB_RIGHT` (U2D2 #1) | Right arm + pan-tilt | Arm: 1-9, Pan: 21, Tilt: 22 |
| `/dev/ttyUSB_LEFT` (U2D2 #2) | Left arm | 11-19 |

### Arm Joint Order (6-DOF WX250s)

`[waist, shoulder, elbow, forearm_roll, wrist_angle, wrist_rotate]`

### Nodes Launched by `real.launch.py`

| Node | Package | Description |
|------|---------|-------------|
| `robot_state_publisher` | `robot_state_publisher` | URDF → TF transforms |
| `joint_state_aggregator` | `joint_state_publisher` | Aggregates arm + pan-tilt joint states → `/joint_states` |
| `phoenix6_base` | `tidybot_control` | Mobile base driver (CAN bus) |
| `xs_sdk` (right_arm) | `interbotix_xs_sdk` | Right arm + pan-tilt Dynamixel driver |
| `xs_sdk` (left_arm) | `interbotix_xs_sdk` | Left arm Dynamixel driver |
| `arm_wrapper` | `tidybot_control` | Sim→real topic translation with velocity limiting |
| `gripper_wrapper` | `tidybot_control` | Sim→real gripper translation (normalized → PWM) |
| `realsense` | `realsense2_camera` | RealSense D435 camera driver |
| `image_compression` | `tidybot_network_bridge` | JPEG/PNG compression (optional) |
| `motion_planner` | `tidybot_ik` | Pinocchio-based IK solver (optional) |
| `microphone` | `tidybot_control` | Audio recording service |

## Hardware Details

### Dual U2D2 Setup

| U2D2 Port | Components | Motor IDs |
|-----------|------------|-----------|
| `/dev/ttyUSB_RIGHT` (U2D2 #1) | Right arm + pan-tilt | Arm: 1-9, Pan: 21, Tilt: 22 |
| `/dev/ttyUSB_LEFT` (U2D2 #2) | Left arm | 11-19 |

### Arm Joint Order (6-DOF WX250s)

`[waist, shoulder, elbow, forearm_roll, wrist_angle, wrist_rotate]`

### Nodes Launched by `real.launch.py`

| Node | Package | Description |
|------|---------|-------------|
| `robot_state_publisher` | `robot_state_publisher` | URDF → TF transforms |
| `joint_state_aggregator` | `joint_state_publisher` | Aggregates arm + pan-tilt joint states → `/joint_states` |
| `phoenix6_base` | `tidybot_control` | Mobile base driver (CAN bus) |
| `xs_sdk` (right_arm) | `interbotix_xs_sdk` | Right arm + pan-tilt Dynamixel driver |
| `xs_sdk` (left_arm) | `interbotix_xs_sdk` | Left arm Dynamixel driver |
| `arm_wrapper` | `tidybot_control` | Sim→real topic translation with velocity limiting |
| `gripper_wrapper` | `tidybot_control` | Sim→real gripper translation (normalized → PWM) |
| `realsense` | `realsense2_camera` | RealSense D435 camera driver |
| `image_compression` | `tidybot_network_bridge` | JPEG/PNG compression (optional) |
| `motion_planner` | `tidybot_ik` | Pinocchio-based IK solver (optional) |
| `microphone` | `tidybot_control` | Audio recording service |

## Troubleshooting

### No topics visible (`ros2 topic list` empty)

1. **Check ROS_DOMAIN_ID matches:**
   ```bash
   echo $ROS_DOMAIN_ID  # Should be 42 on both robot and client
   echo $ROS_DOMAIN_ID  # Should be 42 on both robot and client
   ```

2. **Check RMW implementation matches:**
   ```bash
   echo $RMW_IMPLEMENTATION  # Should be rmw_cyclonedds_cpp on both
   echo $RMW_IMPLEMENTATION  # Should be rmw_cyclonedds_cpp on both
   ```

3. **Check network connectivity:**
   ```bash
   # For local network
   ping <ROBOT_IP>
   
   # For Tailscale
   tailscale ping <ROBOT_TAILSCALE_IP>
   tailscale status  # Check if both machines are connected
   ```

4. **Check firewall (temporarily disable for testing):**
   ```bash
   sudo ufw disable
   ```

5. **Try explicit peer list (Cyclone DDS):**
   Edit `cyclone_dds_client.xml` and add robot IP explicitly.

### Tailscale-specific issues

1. **Check Tailscale connection:**
   ```bash
   tailscale status
   # Both machines should show as "active; relay ..." or "active; direct ..."
   ```

2. **Verify Tailscale IPs:**
   ```bash
   # On robot
   tailscale ip -4
   
   # On client
   tailscale ip -4
   
   # Both should have IPs in 100.x.x.x range
   ```

3. **Test Tailscale connectivity:**
   ```bash
   # From client, ping the robot's Tailscale IP
   ping <ROBOT_TAILSCALE_IP>
   ```

4. **Check Tailscale subnet routes:**
   ```bash
   # If you need access to robot's local network
   tailscale status
   # Look for "relay" vs "direct" connection
   ```

5. **Enable image compression over Tailscale:**
   ```bash
   # On robot
   ros2 launch tidybot_bringup real.launch.py use_compression:=true
   
   # Reduces bandwidth from ~14 MB/s to ~1-2 MB/s
   ```

### Camera images not appearing / slow

1. **Enable compression on the robot:**
1. **Enable compression on the robot:**
   ```bash
   ros2 launch tidybot_bringup real.launch.py use_compression:=true
   ros2 launch tidybot_bringup real.launch.py use_compression:=true
   ```
   Then subscribe to `/camera/color/image_compressed` on the client.
   Then subscribe to `/camera/color/image_compressed` on the client.

2. **Check bandwidth:**
   - Raw images (640x480 @ 15fps): ~14 MB/s
   - Raw images (640x480 @ 15fps): ~14 MB/s
   - Compressed images: ~1-2 MB/s
   - **Important**: Always use compression when connecting over Tailscale

3. **Camera resolution** is set to 640x480 @ 15fps by default in the launch file.
3. **Camera resolution** is set to 640x480 @ 15fps by default in the launch file.

### High latency

1. **Check WiFi signal strength**
2. **Use Ethernet if possible** (for local network)
3. **Enable image compression** with `use_compression:=true`
4. **For Tailscale**: Check connection type with `tailscale status`
   - "direct" connection is best (peer-to-peer)
   - "relay" connection adds latency but works through any firewall

### "Failed to connect to arm"

1. **Check USB connections** on robot
2. **Check device permissions:**
   ```bash
   sudo chmod 666 /dev/ttyUSB0
   # Or add to dialout group permanently
   sudo usermod -a -G dialout $USER
   ```
3. **Check U2D2 symlinks** — the launch expects `/dev/ttyUSB_RIGHT` and `/dev/ttyUSB_LEFT`
3. **Check U2D2 symlinks** — the launch expects `/dev/ttyUSB_RIGHT` and `/dev/ttyUSB_LEFT`

## Environment Variables Summary

```bash
# Both robot and client (set automatically by setup_env.bash on robot)
# Both robot and client (set automatically by setup_env.bash on robot)

# Required
export ROS_DOMAIN_ID=42                           # Must match!
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp      # Or rmw_fastrtps_cpp

# Optional (for Cyclone DDS)
export CYCLONEDDS_URI=file:///path/to/config.xml

# Optional (for FastDDS)
export FASTRTPS_DEFAULT_PROFILES_FILE=/path/to/config.xml

# Robot only (set by setup_env.bash)
# Robot only (set by setup_env.bash)
export TIDYBOT2_PATH=/home/locobot/tidybot2
export TIDYBOT_REPO_ROOT=/home/locobot/collaborative-robotics-2026
export TIDYBOT_REPO_ROOT=/home/locobot/collaborative-robotics-2026
```

## Security Notes

- ROS2 DDS traffic is **not encrypted by default**
- **Tailscale encrypts all traffic** via WireGuard protocol
- For production use without Tailscale, enable DDS security features
- Don't expose ROS2 ports to the internet without VPN
- **Recommended**: Use Tailscale for all remote access outside local network

## Further Reading

- [ROS2 Networking Concepts](https://docs.ros.org/en/humble/Concepts/About-Different-Middleware-Vendors.html)
- [Cyclone DDS Configuration](https://cyclonedds.io/docs/)
- [FastDDS Discovery Server](https://fast-dds.docs.eprosima.com/en/latest/fastdds/discovery/discovery_server.html)
- [Tailscale Documentation](https://tailscale.com/kb/)
- [Tailscale ROS2 Best Practices](https://tailscale.com/kb/)
