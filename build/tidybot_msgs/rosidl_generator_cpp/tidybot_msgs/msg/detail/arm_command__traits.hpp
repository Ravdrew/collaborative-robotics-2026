// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from tidybot_msgs:msg/ArmCommand.idl
// generated code does not contain a copyright notice

#ifndef TIDYBOT_MSGS__MSG__DETAIL__ARM_COMMAND__TRAITS_HPP_
#define TIDYBOT_MSGS__MSG__DETAIL__ARM_COMMAND__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "tidybot_msgs/msg/detail/arm_command__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

// Include directives for member types
// Member 'header'
#include "std_msgs/msg/detail/header__traits.hpp"

namespace tidybot_msgs
{

namespace msg
{

inline void to_flow_style_yaml(
  const ArmCommand & msg,
  std::ostream & out)
{
  out << "{";
  // member: header
  {
    out << "header: ";
    to_flow_style_yaml(msg.header, out);
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

  // member: duration
  {
    out << "duration: ";
    rosidl_generator_traits::value_to_yaml(msg.duration, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const ArmCommand & msg,
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

  // member: duration
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "duration: ";
    rosidl_generator_traits::value_to_yaml(msg.duration, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const ArmCommand & msg, bool use_flow_style = false)
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
  const tidybot_msgs::msg::ArmCommand & msg,
  std::ostream & out, size_t indentation = 0)
{
  tidybot_msgs::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use tidybot_msgs::msg::to_yaml() instead")]]
inline std::string to_yaml(const tidybot_msgs::msg::ArmCommand & msg)
{
  return tidybot_msgs::msg::to_yaml(msg);
}

template<>
inline const char * data_type<tidybot_msgs::msg::ArmCommand>()
{
  return "tidybot_msgs::msg::ArmCommand";
}

template<>
inline const char * name<tidybot_msgs::msg::ArmCommand>()
{
  return "tidybot_msgs/msg/ArmCommand";
}

template<>
struct has_fixed_size<tidybot_msgs::msg::ArmCommand>
  : std::integral_constant<bool, has_fixed_size<std_msgs::msg::Header>::value> {};

template<>
struct has_bounded_size<tidybot_msgs::msg::ArmCommand>
  : std::integral_constant<bool, has_bounded_size<std_msgs::msg::Header>::value> {};

template<>
struct is_message<tidybot_msgs::msg::ArmCommand>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // TIDYBOT_MSGS__MSG__DETAIL__ARM_COMMAND__TRAITS_HPP_
