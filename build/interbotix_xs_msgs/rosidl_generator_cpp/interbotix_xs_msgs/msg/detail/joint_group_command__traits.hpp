// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from interbotix_xs_msgs:msg/JointGroupCommand.idl
// generated code does not contain a copyright notice

#ifndef INTERBOTIX_XS_MSGS__MSG__DETAIL__JOINT_GROUP_COMMAND__TRAITS_HPP_
#define INTERBOTIX_XS_MSGS__MSG__DETAIL__JOINT_GROUP_COMMAND__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "interbotix_xs_msgs/msg/detail/joint_group_command__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

namespace interbotix_xs_msgs
{

namespace msg
{

inline void to_flow_style_yaml(
  const JointGroupCommand & msg,
  std::ostream & out)
{
  out << "{";
  // member: name
  {
    out << "name: ";
    rosidl_generator_traits::value_to_yaml(msg.name, out);
    out << ", ";
  }

  // member: cmd
  {
    if (msg.cmd.size() == 0) {
      out << "cmd: []";
    } else {
      out << "cmd: [";
      size_t pending_items = msg.cmd.size();
      for (auto item : msg.cmd) {
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
  const JointGroupCommand & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: name
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "name: ";
    rosidl_generator_traits::value_to_yaml(msg.name, out);
    out << "\n";
  }

  // member: cmd
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.cmd.size() == 0) {
      out << "cmd: []\n";
    } else {
      out << "cmd:\n";
      for (auto item : msg.cmd) {
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

inline std::string to_yaml(const JointGroupCommand & msg, bool use_flow_style = false)
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

}  // namespace interbotix_xs_msgs

namespace rosidl_generator_traits
{

[[deprecated("use interbotix_xs_msgs::msg::to_block_style_yaml() instead")]]
inline void to_yaml(
  const interbotix_xs_msgs::msg::JointGroupCommand & msg,
  std::ostream & out, size_t indentation = 0)
{
  interbotix_xs_msgs::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use interbotix_xs_msgs::msg::to_yaml() instead")]]
inline std::string to_yaml(const interbotix_xs_msgs::msg::JointGroupCommand & msg)
{
  return interbotix_xs_msgs::msg::to_yaml(msg);
}

template<>
inline const char * data_type<interbotix_xs_msgs::msg::JointGroupCommand>()
{
  return "interbotix_xs_msgs::msg::JointGroupCommand";
}

template<>
inline const char * name<interbotix_xs_msgs::msg::JointGroupCommand>()
{
  return "interbotix_xs_msgs/msg/JointGroupCommand";
}

template<>
struct has_fixed_size<interbotix_xs_msgs::msg::JointGroupCommand>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<interbotix_xs_msgs::msg::JointGroupCommand>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<interbotix_xs_msgs::msg::JointGroupCommand>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // INTERBOTIX_XS_MSGS__MSG__DETAIL__JOINT_GROUP_COMMAND__TRAITS_HPP_
