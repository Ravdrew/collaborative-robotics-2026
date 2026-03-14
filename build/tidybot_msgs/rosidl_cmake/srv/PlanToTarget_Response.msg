

# Whether planning succeeded (collision-free, no singularity)
bool success

# Joint positions solution (6 values if success)
# Order: waist, shoulder, elbow, forearm_roll, wrist_angle, wrist_rotate
float64[] joint_positions

# Whether execution was attempted (only if success=true and execute=true)
bool executed

# Position error of IK solution (meters)
float64 position_error

# Orientation error of IK solution (radians)
float64 orientation_error

# Jacobian condition number at solution (lower = further from singularity)
float64 condition_number

# Detailed status message
string message
