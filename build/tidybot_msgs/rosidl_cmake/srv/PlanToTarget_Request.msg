# Plan and execute motion to target pose service
# Validates trajectory for collisions and singularities before execution

# Which arm to move: "right" or "left"
string arm_name

# Target end-effector pose in base_link frame
geometry_msgs/Pose target_pose

# Whether to match orientation or position only
bool use_orientation true

# Whether to execute the motion if planning succeeds
bool execute true

# Motion duration in seconds (for execution)
float64 duration 2.0

# Maximum allowed condition number for Jacobian (singularity threshold)
# Lower = more conservative. Default 100.0 is reasonable, but to prevent unintended failure, set 500.0 here as default
float64 max_condition_number 500.0

