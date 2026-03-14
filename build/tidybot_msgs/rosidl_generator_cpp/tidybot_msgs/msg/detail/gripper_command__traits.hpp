// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from tidybot_msgs:msg/GripperCommand.idl
// generated code does not contain a copyright notice

#ifndef TIDYBOT_MSGS__MSG__DETAIL__GRIPPER_COMMAND__TRAITS_HPP_
#define TIDYBOT_MSGS__MSG__DETAIL__GRIPPER_COMMAND__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "tidybot_msgs/msg/detail/gripper_command__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

// Include directives for member types
// Member 'header'
#include "std_msgs/msg/detail/header__traits.hpp"

namespace tidybot_msgs
{

namespace msg
{

inline void to_flow_style_yaml(
  const GripperCommand & msg,
  std::ostream & out)
{
  out << "{";
  // member: header
  {
    out << "header: ";
    to_flow_style_yaml(msg.header, out);
    out << ", ";
  }

  // member: position
  {
    out << "position: ";
    rosidl_generator_traits::value_to_yaml(msg.position, out);
    out << ", ";
  }

  // member: effort
  {
    out << "effort: ";
    rosidl_generator_traits::value_to_yaml(msg.effort, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const GripperCommand & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: header
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "header:\n";
    to_block_style_yaml(msg.header, out, indentation + 2);
  }

  // member: position
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "position: ";
    rosidl_generator_traits::value_to_yaml(msg.position, out);
    out << "\n";
  }

  // member: effort
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "effort: ";
    rosidl_generator_traits::value_to_yaml(msg.effort, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const GripperCommand & msg, bool use_flow_style = false)
{
  std::ostringstream out;
  if (use_flow_style) {
    to_flow_style_yaml(msg, out);
  } else {
    to_block_style_yaml(msg, out);
  }
  return out.str();
}

}  // namespace msg

}  // namespace tidybot_msgs

namespace rosidl_generator_traits
{

[[deprecated("use tidybot_msgs::msg::to_block_style_yaml() instead")]]
inline void to_yaml(
  const tidybot_msgs::msg::GripperCommand & msg,
  std::ostream & out, size_t indentation = 0)
{
  tidybot_msgs::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use tidybot_msgs::msg::to_yaml() instead")]]
inline std::string to_yaml(const tidybot_msgs::msg::GripperCommand & msg)
{
  return tidybot_msgs::msg::to_yaml(msg);
}

template<>
inline const char * data_type<tidybot_msgs::msg::GripperCommand>()
{
  return "tidybot_msgs::msg::GripperCommand";
}

template<>
inline const char * name<tidybot_msgs::msg::GripperCommand>()
{
  return "tidybot_msgs/msg/GripperCommand";
}

template<>
struct has_fixed_size<tidybot_msgs::msg::GripperCommand>
  : std::integral_constant<bool, has_fixed_size<std_msgs::msg::Header>::value> {};

template<>
struct has_bounded_size<tidybot_msgs::msg::GripperCommand>
  : std::integral_constant<bool, has_bounded_size<std_msgs::msg::Header>::value> {};

template<>
struct is_message<tidybot_msgs::msg::GripperCommand>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // TIDYBOT_MSGS__MSG__DETAIL__GRIPPER_COMMAND__TRAITS_HPP_
