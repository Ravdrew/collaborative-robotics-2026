// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from interbotix_xs_msgs:msg/TurretJoy.idl
// generated code does not contain a copyright notice

#ifndef INTERBOTIX_XS_MSGS__MSG__DETAIL__TURRET_JOY__TRAITS_HPP_
#define INTERBOTIX_XS_MSGS__MSG__DETAIL__TURRET_JOY__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "interbotix_xs_msgs/msg/detail/turret_joy__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

namespace interbotix_xs_msgs
{

namespace msg
{

inline void to_flow_style_yaml(
  const TurretJoy & msg,
  std::ostream & out)
{
  out << "{";
  // member: pan_cmd
  {
    out << "pan_cmd: ";
    rosidl_generator_traits::value_to_yaml(msg.pan_cmd, out);
    out << ", ";
  }

  // member: tilt_cmd
  {
    out << "tilt_cmd: ";
    rosidl_generator_traits::value_to_yaml(msg.tilt_cmd, out);
    out << ", ";
  }

  // member: speed_cmd
  {
    out << "speed_cmd: ";
    rosidl_generator_traits::value_to_yaml(msg.speed_cmd, out);
    out << ", ";
  }

  // member: speed_toggle_cmd
  {
    out << "speed_toggle_cmd: ";
    rosidl_generator_traits::value_to_yaml(msg.speed_toggle_cmd, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const TurretJoy & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: pan_cmd
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "pan_cmd: ";
    rosidl_generator_traits::value_to_yaml(msg.pan_cmd, out);
    out << "\n";
  }

  // member: tilt_cmd
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "tilt_cmd: ";
    rosidl_generator_traits::value_to_yaml(msg.tilt_cmd, out);
    out << "\n";
  }

  // member: speed_cmd
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "speed_cmd: ";
    rosidl_generator_traits::value_to_yaml(msg.speed_cmd, out);
    out << "\n";
  }

  // member: speed_toggle_cmd
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "speed_toggle_cmd: ";
    rosidl_generator_traits::value_to_yaml(msg.speed_toggle_cmd, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const TurretJoy & msg, bool use_flow_style = false)
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
  const interbotix_xs_msgs::msg::TurretJoy & msg,
  std::ostream & out, size_t indentation = 0)
{
  interbotix_xs_msgs::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use interbotix_xs_msgs::msg::to_yaml() instead")]]
inline std::string to_yaml(const interbotix_xs_msgs::msg::TurretJoy & msg)
{
  return interbotix_xs_msgs::msg::to_yaml(msg);
}

template<>
inline const char * data_type<interbotix_xs_msgs::msg::TurretJoy>()
{
  return "interbotix_xs_msgs::msg::TurretJoy";
}

template<>
inline const char * name<interbotix_xs_msgs::msg::TurretJoy>()
{
  return "interbotix_xs_msgs/msg/TurretJoy";
}

template<>
struct has_fixed_size<interbotix_xs_msgs::msg::TurretJoy>
  : std::integral_constant<bool, true> {};

template<>
struct has_bounded_size<interbotix_xs_msgs::msg::TurretJoy>
  : std::integral_constant<bool, true> {};

template<>
struct is_message<interbotix_xs_msgs::msg::TurretJoy>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // INTERBOTIX_XS_MSGS__MSG__DETAIL__TURRET_JOY__TRAITS_HPP_
