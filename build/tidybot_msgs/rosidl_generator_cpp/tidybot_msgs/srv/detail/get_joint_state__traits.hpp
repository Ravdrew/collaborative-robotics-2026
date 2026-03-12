// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from tidybot_msgs:srv/GetJointState.idl
// generated code does not contain a copyright notice

#ifndef TIDYBOT_MSGS__SRV__DETAIL__GET_JOINT_STATE__TRAITS_HPP_
#define TIDYBOT_MSGS__SRV__DETAIL__GET_JOINT_STATE__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "tidybot_msgs/srv/detail/get_joint_state__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

namespace tidybot_msgs
{

namespace srv
{

inline void to_flow_style_yaml(
  const GetJointState_Request & msg,
  std::ostream & out)
{
  out << "{";
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
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const GetJointState_Request & msg,
  std::ostream & out, size_t indentation = 0)
{
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
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const GetJointState_Request & msg, bool use_flow_style = false)
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
  const tidybot_msgs::srv::GetJointState_Request & msg,
  std::ostream & out, size_t indentation = 0)
{
  tidybot_msgs::srv::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use tidybot_msgs::srv::to_yaml() instead")]]
inline std::string to_yaml(const tidybot_msgs::srv::GetJointState_Request & msg)
{
  return tidybot_msgs::srv::to_yaml(msg);
}

template<>
inline const char * data_type<tidybot_msgs::srv::GetJointState_Request>()
{
  return "tidybot_msgs::srv::GetJointState_Request";
}

template<>
inline const char * name<tidybot_msgs::srv::GetJointState_Request>()
{
  return "tidybot_msgs/srv/GetJointState_Request";
}

template<>
struct has_fixed_size<tidybot_msgs::srv::GetJointState_Request>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<tidybot_msgs::srv::GetJointState_Request>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<tidybot_msgs::srv::GetJointState_Request>
  : std::true_type {};

}  // namespace rosidl_generator_traits

// Include directives for member types
// Member 'state'
#include "sensor_msgs/msg/detail/joint_state__traits.hpp"

namespace tidybot_msgs
{

namespace srv
{

inline void to_flow_style_yaml(
  const GetJointState_Response & msg,
  std::ostream & out)
{
  out << "{";
  // member: state
  {
    out << "state: ";
    to_flow_style_yaml(msg.state, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const GetJointState_Response & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: state
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "state:\n";
    to_block_style_yaml(msg.state, out, indentation + 2);
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const GetJointState_Response & msg, bool use_flow_style = false)
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
  const tidybot_msgs::srv::GetJointState_Response & msg,
  std::ostream & out, size_t indentation = 0)
{
  tidybot_msgs::srv::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use tidybot_msgs::srv::to_yaml() instead")]]
inline std::string to_yaml(const tidybot_msgs::srv::GetJointState_Response & msg)
{
  return tidybot_msgs::srv::to_yaml(msg);
}

template<>
inline const char * data_type<tidybot_msgs::srv::GetJointState_Response>()
{
  return "tidybot_msgs::srv::GetJointState_Response";
}

template<>
inline const char * name<tidybot_msgs::srv::GetJointState_Response>()
{
  return "tidybot_msgs/srv/GetJointState_Response";
}

template<>
struct has_fixed_size<tidybot_msgs::srv::GetJointState_Response>
  : std::integral_constant<bool, has_fixed_size<sensor_msgs::msg::JointState>::value> {};

template<>
struct has_bounded_size<tidybot_msgs::srv::GetJointState_Response>
  : std::integral_constant<bool, has_bounded_size<sensor_msgs::msg::JointState>::value> {};

template<>
struct is_message<tidybot_msgs::srv::GetJointState_Response>
  : std::true_type {};

}  // namespace rosidl_generator_traits

namespace rosidl_generator_traits
{

template<>
inline const char * data_type<tidybot_msgs::srv::GetJointState>()
{
  return "tidybot_msgs::srv::GetJointState";
}

template<>
inline const char * name<tidybot_msgs::srv::GetJointState>()
{
  return "tidybot_msgs/srv/GetJointState";
}

template<>
struct has_fixed_size<tidybot_msgs::srv::GetJointState>
  : std::integral_constant<
    bool,
    has_fixed_size<tidybot_msgs::srv::GetJointState_Request>::value &&
    has_fixed_size<tidybot_msgs::srv::GetJointState_Response>::value
  >
{
};

template<>
struct has_bounded_size<tidybot_msgs::srv::GetJointState>
  : std::integral_constant<
    bool,
    has_bounded_size<tidybot_msgs::srv::GetJointState_Request>::value &&
    has_bounded_size<tidybot_msgs::srv::GetJointState_Response>::value
  >
{
};

template<>
struct is_service<tidybot_msgs::srv::GetJointState>
  : std::true_type
{
};

template<>
struct is_service_request<tidybot_msgs::srv::GetJointState_Request>
  : std::true_type
{
};

template<>
struct is_service_response<tidybot_msgs::srv::GetJointState_Response>
  : std::true_type
{
};

}  // namespace rosidl_generator_traits

#endif  // TIDYBOT_MSGS__SRV__DETAIL__GET_JOINT_STATE__TRAITS_HPP_
