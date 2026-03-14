// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from interbotix_xs_msgs:srv/RobotInfo.idl
// generated code does not contain a copyright notice

#ifndef INTERBOTIX_XS_MSGS__SRV__DETAIL__ROBOT_INFO__TRAITS_HPP_
#define INTERBOTIX_XS_MSGS__SRV__DETAIL__ROBOT_INFO__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "interbotix_xs_msgs/srv/detail/robot_info__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

namespace interbotix_xs_msgs
{

namespace srv
{

inline void to_flow_style_yaml(
  const RobotInfo_Request & msg,
  std::ostream & out)
{
  out << "{";
  // member: cmd_type
  {
    out << "cmd_type: ";
    rosidl_generator_traits::value_to_yaml(msg.cmd_type, out);
    out << ", ";
  }

  // member: name
  {
    out << "name: ";
    rosidl_generator_traits::value_to_yaml(msg.name, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const RobotInfo_Request & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: cmd_type
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "cmd_type: ";
    rosidl_generator_traits::value_to_yaml(msg.cmd_type, out);
    out << "\n";
  }

  // member: name
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "name: ";
    rosidl_generator_traits::value_to_yaml(msg.name, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const RobotInfo_Request & msg, bool use_flow_style = false)
{
  std::ostringstream out;
  if (use_flow_style) {
    to_flow_style_yaml(msg, out);
  } else {
    to_block_style_yaml(msg, out);
  }
  return out.str();
}

}  // namespace srv

}  // namespace interbotix_xs_msgs

namespace rosidl_generator_traits
{

[[deprecated("use interbotix_xs_msgs::srv::to_block_style_yaml() instead")]]
inline void to_yaml(
  const interbotix_xs_msgs::srv::RobotInfo_Request & msg,
  std::ostream & out, size_t indentation = 0)
{
  interbotix_xs_msgs::srv::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use interbotix_xs_msgs::srv::to_yaml() instead")]]
inline std::string to_yaml(const interbotix_xs_msgs::srv::RobotInfo_Request & msg)
{
  return interbotix_xs_msgs::srv::to_yaml(msg);
}

template<>
inline const char * data_type<interbotix_xs_msgs::srv::RobotInfo_Request>()
{
  return "interbotix_xs_msgs::srv::RobotInfo_Request";
}

template<>
inline const char * name<interbotix_xs_msgs::srv::RobotInfo_Request>()
{
  return "interbotix_xs_msgs/srv/RobotInfo_Request";
}

template<>
struct has_fixed_size<interbotix_xs_msgs::srv::RobotInfo_Request>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<interbotix_xs_msgs::srv::RobotInfo_Request>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<interbotix_xs_msgs::srv::RobotInfo_Request>
  : std::true_type {};

}  // namespace rosidl_generator_traits

namespace interbotix_xs_msgs
{

namespace srv
{

inline void to_flow_style_yaml(
  const RobotInfo_Response & msg,
  std::ostream & out)
{
  out << "{";
  // member: mode
  {
    out << "mode: ";
    rosidl_generator_traits::value_to_yaml(msg.mode, out);
    out << ", ";
  }

  // member: profile_type
  {
    out << "profile_type: ";
    rosidl_generator_traits::value_to_yaml(msg.profile_type, out);
    out << ", ";
  }

  // member: joint_names
  {
    if (msg.joint_names.size() == 0) {
      out << "joint_names: []";
    } else {
      out << "joint_names: [";
      size_t pending_items = msg.joint_names.size();
      for (auto item : msg.joint_names) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: joint_ids
  {
    if (msg.joint_ids.size() == 0) {
      out << "joint_ids: []";
    } else {
      out << "joint_ids: [";
      size_t pending_items = msg.joint_ids.size();
      for (auto item : msg.joint_ids) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: joint_lower_limits
  {
    if (msg.joint_lower_limits.size() == 0) {
      out << "joint_lower_limits: []";
    } else {
      out << "joint_lower_limits: [";
      size_t pending_items = msg.joint_lower_limits.size();
      for (auto item : msg.joint_lower_limits) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: joint_upper_limits
  {
    if (msg.joint_upper_limits.size() == 0) {
      out << "joint_upper_limits: []";
    } else {
      out << "joint_upper_limits: [";
      size_t pending_items = msg.joint_upper_limits.size();
      for (auto item : msg.joint_upper_limits) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: joint_velocity_limits
  {
    if (msg.joint_velocity_limits.size() == 0) {
      out << "joint_velocity_limits: []";
    } else {
      out << "joint_velocity_limits: [";
      size_t pending_items = msg.joint_velocity_limits.size();
      for (auto item : msg.joint_velocity_limits) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: joint_sleep_positions
  {
    if (msg.joint_sleep_positions.size() == 0) {
      out << "joint_sleep_positions: []";
    } else {
      out << "joint_sleep_positions: [";
      size_t pending_items = msg.joint_sleep_positions.size();
      for (auto item : msg.joint_sleep_positions) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: joint_state_indices
  {
    if (msg.joint_state_indices.size() == 0) {
      out << "joint_state_indices: []";
    } else {
      out << "joint_state_indices: [";
      size_t pending_items = msg.joint_state_indices.size();
      for (auto item : msg.joint_state_indices) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: num_joints
  {
    out << "num_joints: ";
    rosidl_generator_traits::value_to_yaml(msg.num_joints, out);
    out << ", ";
  }

  // member: name
  {
    if (msg.name.size() == 0) {
      out << "name: []";
    } else {
      out << "name: [";
      size_t pending_items = msg.name.size();
      for (auto item : msg.name) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const RobotInfo_Response & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: mode
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "mode: ";
    rosidl_generator_traits::value_to_yaml(msg.mode, out);
    out << "\n";
  }

  // member: profile_type
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "profile_type: ";
    rosidl_generator_traits::value_to_yaml(msg.profile_type, out);
    out << "\n";
  }

  // member: joint_names
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.joint_names.size() == 0) {
      out << "joint_names: []\n";
    } else {
      out << "joint_names:\n";
      for (auto item : msg.joint_names) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: joint_ids
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.joint_ids.size() == 0) {
      out << "joint_ids: []\n";
    } else {
      out << "joint_ids:\n";
      for (auto item : msg.joint_ids) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: joint_lower_limits
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.joint_lower_limits.size() == 0) {
      out << "joint_lower_limits: []\n";
    } else {
      out << "joint_lower_limits:\n";
      for (auto item : msg.joint_lower_limits) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: joint_upper_limits
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.joint_upper_limits.size() == 0) {
      out << "joint_upper_limits: []\n";
    } else {
      out << "joint_upper_limits:\n";
      for (auto item : msg.joint_upper_limits) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: joint_velocity_limits
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.joint_velocity_limits.size() == 0) {
      out << "joint_velocity_limits: []\n";
    } else {
      out << "joint_velocity_limits:\n";
      for (auto item : msg.joint_velocity_limits) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: joint_sleep_positions
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.joint_sleep_positions.size() == 0) {
      out << "joint_sleep_positions: []\n";
    } else {
      out << "joint_sleep_positions:\n";
      for (auto item : msg.joint_sleep_positions) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: joint_state_indices
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.joint_state_indices.size() == 0) {
      out << "joint_state_indices: []\n";
    } else {
      out << "joint_state_indices:\n";
      for (auto item : msg.joint_state_indices) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: num_joints
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "num_joints: ";
    rosidl_generator_traits::value_to_yaml(msg.num_joints, out);
    out << "\n";
  }

  // member: name
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.name.size() == 0) {
      out << "name: []\n";
    } else {
      out << "name:\n";
      for (auto item : msg.name) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const RobotInfo_Response & msg, bool use_flow_style = false)
{
  std::ostringstream out;
  if (use_flow_style) {
    to_flow_style_yaml(msg, out);
  } else {
    to_block_style_yaml(msg, out);
  }
  return out.str();
}

}  // namespace srv

}  // namespace interbotix_xs_msgs

namespace rosidl_generator_traits
{

[[deprecated("use interbotix_xs_msgs::srv::to_block_style_yaml() instead")]]
inline void to_yaml(
  const interbotix_xs_msgs::srv::RobotInfo_Response & msg,
  std::ostream & out, size_t indentation = 0)
{
  interbotix_xs_msgs::srv::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use interbotix_xs_msgs::srv::to_yaml() instead")]]
inline std::string to_yaml(const interbotix_xs_msgs::srv::RobotInfo_Response & msg)
{
  return interbotix_xs_msgs::srv::to_yaml(msg);
}

template<>
inline const char * data_type<interbotix_xs_msgs::srv::RobotInfo_Response>()
{
  return "interbotix_xs_msgs::srv::RobotInfo_Response";
}

template<>
inline const char * name<interbotix_xs_msgs::srv::RobotInfo_Response>()
{
  return "interbotix_xs_msgs/srv/RobotInfo_Response";
}

template<>
struct has_fixed_size<interbotix_xs_msgs::srv::RobotInfo_Response>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<interbotix_xs_msgs::srv::RobotInfo_Response>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<interbotix_xs_msgs::srv::RobotInfo_Response>
  : std::true_type {};

}  // namespace rosidl_generator_traits

namespace rosidl_generator_traits
{

template<>
inline const char * data_type<interbotix_xs_msgs::srv::RobotInfo>()
{
  return "interbotix_xs_msgs::srv::RobotInfo";
}

template<>
inline const char * name<interbotix_xs_msgs::srv::RobotInfo>()
{
  return "interbotix_xs_msgs/srv/RobotInfo";
}

template<>
struct has_fixed_size<interbotix_xs_msgs::srv::RobotInfo>
  : std::integral_constant<
    bool,
    has_fixed_size<interbotix_xs_msgs::srv::RobotInfo_Request>::value &&
    has_fixed_size<interbotix_xs_msgs::srv::RobotInfo_Response>::value
  >
{
};

template<>
struct has_bounded_size<interbotix_xs_msgs::srv::RobotInfo>
  : std::integral_constant<
    bool,
    has_bounded_size<interbotix_xs_msgs::srv::RobotInfo_Request>::value &&
    has_bounded_size<interbotix_xs_msgs::srv::RobotInfo_Response>::value
  >
{
};

template<>
struct is_service<interbotix_xs_msgs::srv::RobotInfo>
  : std::true_type
{
};

template<>
struct is_service_request<interbotix_xs_msgs::srv::RobotInfo_Request>
  : std::true_type
{
};

template<>
struct is_service_response<interbotix_xs_msgs::srv::RobotInfo_Response>
  : std::true_type
{
};

}  // namespace rosidl_generator_traits

#endif  // INTERBOTIX_XS_MSGS__SRV__DETAIL__ROBOT_INFO__TRAITS_HPP_
