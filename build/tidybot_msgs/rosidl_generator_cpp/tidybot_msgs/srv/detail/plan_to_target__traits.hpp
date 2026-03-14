// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from tidybot_msgs:srv/PlanToTarget.idl
// generated code does not contain a copyright notice

#ifndef TIDYBOT_MSGS__SRV__DETAIL__PLAN_TO_TARGET__TRAITS_HPP_
#define TIDYBOT_MSGS__SRV__DETAIL__PLAN_TO_TARGET__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "tidybot_msgs/srv/detail/plan_to_target__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

// Include directives for member types
// Member 'target_pose'
#include "geometry_msgs/msg/detail/pose__traits.hpp"

namespace tidybot_msgs
{

namespace srv
{

inline void to_flow_style_yaml(
  const PlanToTarget_Request & msg,
  std::ostream & out)
{
  out << "{";
  // member: arm_name
  {
    out << "arm_name: ";
    rosidl_generator_traits::value_to_yaml(msg.arm_name, out);
    out << ", ";
  }

  // member: target_pose
  {
    out << "target_pose: ";
    to_flow_style_yaml(msg.target_pose, out);
    out << ", ";
  }

  // member: use_orientation
  {
    out << "use_orientation: ";
    rosidl_generator_traits::value_to_yaml(msg.use_orientation, out);
    out << ", ";
  }

  // member: execute
  {
    out << "execute: ";
    rosidl_generator_traits::value_to_yaml(msg.execute, out);
    out << ", ";
  }

  // member: duration
  {
    out << "duration: ";
    rosidl_generator_traits::value_to_yaml(msg.duration, out);
    out << ", ";
  }

  // member: max_condition_number
  {
    out << "max_condition_number: ";
    rosidl_generator_traits::value_to_yaml(msg.max_condition_number, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const PlanToTarget_Request & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: arm_name
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "arm_name: ";
    rosidl_generator_traits::value_to_yaml(msg.arm_name, out);
    out << "\n";
  }

  // member: target_pose
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "target_pose:\n";
    to_block_style_yaml(msg.target_pose, out, indentation + 2);
  }

  // member: use_orientation
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "use_orientation: ";
    rosidl_generator_traits::value_to_yaml(msg.use_orientation, out);
    out << "\n";
  }

  // member: execute
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "execute: ";
    rosidl_generator_traits::value_to_yaml(msg.execute, out);
    out << "\n";
  }

  // member: duration
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "duration: ";
    rosidl_generator_traits::value_to_yaml(msg.duration, out);
    out << "\n";
  }

  // member: max_condition_number
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "max_condition_number: ";
    rosidl_generator_traits::value_to_yaml(msg.max_condition_number, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const PlanToTarget_Request & msg, bool use_flow_style = false)
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

}  // namespace tidybot_msgs

namespace rosidl_generator_traits
{

[[deprecated("use tidybot_msgs::srv::to_block_style_yaml() instead")]]
inline void to_yaml(
  const tidybot_msgs::srv::PlanToTarget_Request & msg,
  std::ostream & out, size_t indentation = 0)
{
  tidybot_msgs::srv::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use tidybot_msgs::srv::to_yaml() instead")]]
inline std::string to_yaml(const tidybot_msgs::srv::PlanToTarget_Request & msg)
{
  return tidybot_msgs::srv::to_yaml(msg);
}

template<>
inline const char * data_type<tidybot_msgs::srv::PlanToTarget_Request>()
{
  return "tidybot_msgs::srv::PlanToTarget_Request";
}

template<>
inline const char * name<tidybot_msgs::srv::PlanToTarget_Request>()
{
  return "tidybot_msgs/srv/PlanToTarget_Request";
}

template<>
struct has_fixed_size<tidybot_msgs::srv::PlanToTarget_Request>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<tidybot_msgs::srv::PlanToTarget_Request>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<tidybot_msgs::srv::PlanToTarget_Request>
  : std::true_type {};

}  // namespace rosidl_generator_traits

namespace tidybot_msgs
{

namespace srv
{

inline void to_flow_style_yaml(
  const PlanToTarget_Response & msg,
  std::ostream & out)
{
  out << "{";
  // member: success
  {
    out << "success: ";
    rosidl_generator_traits::value_to_yaml(msg.success, out);
    out << ", ";
  }

  // member: joint_positions
  {
    if (msg.joint_positions.size() == 0) {
      out << "joint_positions: []";
    } else {
      out << "joint_positions: [";
      size_t pending_items = msg.joint_positions.size();
      for (auto item : msg.joint_positions) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: executed
  {
    out << "executed: ";
    rosidl_generator_traits::value_to_yaml(msg.executed, out);
    out << ", ";
  }

  // member: position_error
  {
    out << "position_error: ";
    rosidl_generator_traits::value_to_yaml(msg.position_error, out);
    out << ", ";
  }

  // member: orientation_error
  {
    out << "orientation_error: ";
    rosidl_generator_traits::value_to_yaml(msg.orientation_error, out);
    out << ", ";
  }

  // member: condition_number
  {
    out << "condition_number: ";
    rosidl_generator_traits::value_to_yaml(msg.condition_number, out);
    out << ", ";
  }

  // member: message
  {
    out << "message: ";
    rosidl_generator_traits::value_to_yaml(msg.message, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const PlanToTarget_Response & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: success
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "success: ";
    rosidl_generator_traits::value_to_yaml(msg.success, out);
    out << "\n";
  }

  // member: joint_positions
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.joint_positions.size() == 0) {
      out << "joint_positions: []\n";
    } else {
      out << "joint_positions:\n";
      for (auto item : msg.joint_positions) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: executed
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "executed: ";
    rosidl_generator_traits::value_to_yaml(msg.executed, out);
    out << "\n";
  }

  // member: position_error
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "position_error: ";
    rosidl_generator_traits::value_to_yaml(msg.position_error, out);
    out << "\n";
  }

  // member: orientation_error
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "orientation_error: ";
    rosidl_generator_traits::value_to_yaml(msg.orientation_error, out);
    out << "\n";
  }

  // member: condition_number
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "condition_number: ";
    rosidl_generator_traits::value_to_yaml(msg.condition_number, out);
    out << "\n";
  }

  // member: message
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "message: ";
    rosidl_generator_traits::value_to_yaml(msg.message, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const PlanToTarget_Response & msg, bool use_flow_style = false)
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

}  // namespace tidybot_msgs

namespace rosidl_generator_traits
{

[[deprecated("use tidybot_msgs::srv::to_block_style_yaml() instead")]]
inline void to_yaml(
  const tidybot_msgs::srv::PlanToTarget_Response & msg,
  std::ostream & out, size_t indentation = 0)
{
  tidybot_msgs::srv::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use tidybot_msgs::srv::to_yaml() instead")]]
inline std::string to_yaml(const tidybot_msgs::srv::PlanToTarget_Response & msg)
{
  return tidybot_msgs::srv::to_yaml(msg);
}

template<>
inline const char * data_type<tidybot_msgs::srv::PlanToTarget_Response>()
{
  return "tidybot_msgs::srv::PlanToTarget_Response";
}

template<>
inline const char * name<tidybot_msgs::srv::PlanToTarget_Response>()
{
  return "tidybot_msgs/srv/PlanToTarget_Response";
}

template<>
struct has_fixed_size<tidybot_msgs::srv::PlanToTarget_Response>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<tidybot_msgs::srv::PlanToTarget_Response>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<tidybot_msgs::srv::PlanToTarget_Response>
  : std::true_type {};

}  // namespace rosidl_generator_traits

namespace rosidl_generator_traits
{

template<>
inline const char * data_type<tidybot_msgs::srv::PlanToTarget>()
{
  return "tidybot_msgs::srv::PlanToTarget";
}

template<>
inline const char * name<tidybot_msgs::srv::PlanToTarget>()
{
  return "tidybot_msgs/srv/PlanToTarget";
}

template<>
struct has_fixed_size<tidybot_msgs::srv::PlanToTarget>
  : std::integral_constant<
    bool,
    has_fixed_size<tidybot_msgs::srv::PlanToTarget_Request>::value &&
    has_fixed_size<tidybot_msgs::srv::PlanToTarget_Response>::value
  >
{
};

template<>
struct has_bounded_size<tidybot_msgs::srv::PlanToTarget>
  : std::integral_constant<
    bool,
    has_bounded_size<tidybot_msgs::srv::PlanToTarget_Request>::value &&
    has_bounded_size<tidybot_msgs::srv::PlanToTarget_Response>::value
  >
{
};

template<>
struct is_service<tidybot_msgs::srv::PlanToTarget>
  : std::true_type
{
};

template<>
struct is_service_request<tidybot_msgs::srv::PlanToTarget_Request>
  : std::true_type
{
};

template<>
struct is_service_response<tidybot_msgs::srv::PlanToTarget_Response>
  : std::true_type
{
};

}  // namespace rosidl_generator_traits

#endif  // TIDYBOT_MSGS__SRV__DETAIL__PLAN_TO_TARGET__TRAITS_HPP_
