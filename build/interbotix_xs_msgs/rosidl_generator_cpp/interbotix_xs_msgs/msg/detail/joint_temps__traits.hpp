// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from interbotix_xs_msgs:msg/JointTemps.idl
// generated code does not contain a copyright notice

#ifndef INTERBOTIX_XS_MSGS__MSG__DETAIL__JOINT_TEMPS__TRAITS_HPP_
#define INTERBOTIX_XS_MSGS__MSG__DETAIL__JOINT_TEMPS__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "interbotix_xs_msgs/msg/detail/joint_temps__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

namespace interbotix_xs_msgs
{

namespace msg
{

inline void to_flow_style_yaml(
  const JointTemps & msg,
  std::ostream & out)
{
  out << "{";
  // member: names
  {
    if (msg.names.size() == 0) {
      out << "names: []";
    } else {
      out << "names: [";
      size_t pending_items = msg.names.size();
      for (auto item : msg.names) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: temps
  {
    if (msg.temps.size() == 0) {
      out << "temps: []";
    } else {
      out << "temps: [";
      size_t pending_items = msg.temps.size();
      for (auto item : msg.temps) {
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
  const JointTemps & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: names
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.names.size() == 0) {
      out << "names: []\n";
    } else {
      out << "names:\n";
      for (auto item : msg.names) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: temps
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.temps.size() == 0) {
      out << "temps: []\n";
    } else {
      out << "temps:\n";
      for (auto item : msg.temps) {
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

inline std::string to_yaml(const JointTemps & msg, bool use_flow_style = false)
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
  const interbotix_xs_msgs::msg::JointTemps & msg,
  std::ostream & out, size_t indentation = 0)
{
  interbotix_xs_msgs::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use interbotix_xs_msgs::msg::to_yaml() instead")]]
inline std::string to_yaml(const interbotix_xs_msgs::msg::JointTemps & msg)
{
  return interbotix_xs_msgs::msg::to_yaml(msg);
}

template<>
inline const char * data_type<interbotix_xs_msgs::msg::JointTemps>()
{
  return "interbotix_xs_msgs::msg::JointTemps";
}

template<>
inline const char * name<interbotix_xs_msgs::msg::JointTemps>()
{
  return "interbotix_xs_msgs/msg/JointTemps";
}

template<>
struct has_fixed_size<interbotix_xs_msgs::msg::JointTemps>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<interbotix_xs_msgs::msg::JointTemps>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<interbotix_xs_msgs::msg::JointTemps>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // INTERBOTIX_XS_MSGS__MSG__DETAIL__JOINT_TEMPS__TRAITS_HPP_
