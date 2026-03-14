

# Whether a valid IK solution was found
bool success

# Joint positions solution (6 values)
# Order: waist, shoulder, elbow, forearm_roll, wrist_angle, wrist_rotate
float64[] joint_positions

# Residual position error in meters
float64 position_error

# Residual orientation error in radians
float64 orientation_error

# Status message (e.g., "Success", "No solution found", "Timeout")
string message
