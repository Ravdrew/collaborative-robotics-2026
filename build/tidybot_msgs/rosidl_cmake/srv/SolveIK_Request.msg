# Inverse kinematics service request
# Computes joint positions for a target end-effector pose

# Arm to solve IK for: "right" or "left"
string arm_name

# Target end-effector pose in base_link frame
geometry_msgs/Pose target_pose

# Optional: Initial joint positions as seed for solver (6 values)
# If empty, uses current joint positions
float64[] seed_joint_positions

# Whether to consider orientation in the solution
# If false, only position is matched
bool use_orientation

# Maximum time allowed for solving in seconds
float64 timeout

