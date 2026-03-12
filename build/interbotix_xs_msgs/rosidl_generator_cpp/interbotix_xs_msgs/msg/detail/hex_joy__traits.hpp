// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from interbotix_xs_msgs:msg/HexJoy.idl
// generated code does not contain a copyright notice

#ifndef INTERBOTIX_XS_MSGS__MSG__DETAIL__HEX_JOY__TRAITS_HPP_
#define INTERBOTIX_XS_MSGS__MSG__DETAIL__HEX_JOY__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "interbotix_xs_msgs/msg/detail/hex_joy__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

namespace interbotix_xs_msgs
{

namespace msg
{

inline void to_flow_style_yaml(
  const HexJoy & msg,
  std::ostream & out)
{
  out << "{";
  // member: world_x_cmd
  {
    out << "world_x_cmd: ";
    rosidl_generator_traits::value_to_yaml(msg.world_x_cmd, out);
    out << ", ";
  }

  // member: world_y_cmd
  {
    out << "world_y_cmd: ";
    rosidl_generator_traits::value_to_yaml(msg.world_y_cmd, out);
    out << ", ";
  }

  // member: world_yaw_cmd
  {
    out << "world_yaw_cmd: ";
    rosidl_generator_traits::value_to_yaml(msg.world_yaw_cmd, out);
    out << ", ";
  }

  // member: place_x_cmd
  {
    out << "place_x_cmd: ";
    rosidl_generator_traits::value_to_yaml(msg.place_x_cmd, out);
    out << ", ";
  }

  // member: place_y_cmd
  {
    out << "place_y_cmd: ";
    rosidl_generator_traits::value_to_yaml(msg.place_y_cmd, out);
    out << ", ";
  }

  // member: place_z_cmd
  {
    out << "place_z_cmd: ";
    rosidl_generator_traits::value_to_yaml(msg.place_z_cmd, out);
    out << ", ";
  }

  // member: place_roll_cmd
  {
    out << "place_roll_cmd: ";
    rosidl_generator_traits::value_to_yaml(msg.place_roll_cmd, out);
    out << ", ";
  }

  // member: place_pitch_cmd
  {
    out << "place_pitch_cmd: ";
    rosidl_generator_traits::value_to_yaml(msg.place_pitch_cmd, out);
    out << ", ";
  }

  // member: pose_cmd
  {
    out << "pose_cmd: ";
    rosidl_generator_traits::value_to_yaml(msg.pose_cmd, out);
    out << ", ";
  }

  // member: move_type_cmd
  {
    out << "move_type_cmd: ";
    rosidl_generator_traits::value_to_yaml(msg.move_type_cmd, out);
    out << ", ";
  }

  // member: gait_toggle_cmd
  {
    out << "gait_toggle_cmd: ";
    rosidl_generator_traits::value_to_yaml(msg.gait_toggle_cmd, out);
    out << ", ";
  }

  // member: leg_toggle_cmd
  {
    out << "leg_toggle_cmd: ";
    rosidl_generator_traits::value_to_yaml(msg.leg_toggle_cmd, out);
    out << ", ";
  }

  // member: stance_cmd
  {
    out << "stance_cmd: ";
    rosidl_generator_traits::value_to_yaml(msg.stance_cmd, out);
    out << ", ";
  }

  // member: reboot_cmd
  {
    out << "reboot_cmd: ";
    rosidl_generator_traits::value_to_yaml(msg.reboot_cmd, out);
    out << ", ";
  }

  // member: set_home_pose_cmd
  {
    out << "set_home_pose_cmd: ";
    rosidl_generator_traits::value_to_yaml(msg.set_home_pose_cmd, out);
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
  const HexJoy & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: world_x_cmd
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "world_x_cmd: ";
    rosidl_generator_traits::value_to_yaml(msg.world_x_cmd, out);
    out << "\n";
  }

  // member: world_y_cmd
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "world_y_cmd: ";
    rosidl_generator_traits::value_to_yaml(msg.world_y_cmd, out);
    out << "\n";
  }

  // member: world_yaw_cmd
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "world_yaw_cmd: ";
    rosidl_generator_traits::value_to_yaml(msg.world_yaw_cmd, out);
    out << "\n";
  }

  // member: place_x_cmd
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "place_x_cmd: ";
    rosidl_generator_traits::value_to_yaml(msg.place_x_cmd, out);
    out << "\n";
  }

  // member: place_y_cmd
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "place_y_cmd: ";
    rosidl_generator_traits::value_to_yaml(msg.place_y_cmd, out);
    out << "\n";
  }

  // member: place_z_cmd
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "place_z_cmd: ";
    rosidl_generator_traits::value_to_yaml(msg.place_z_cmd, out);
    out << "\n";
  }

  // member: place_roll_cmd
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "place_roll_cmd: ";
    rosidl_generator_traits::value_to_yaml(msg.place_roll_cmd, out);
    out << "\n";
  }

  // member: place_pitch_cmd
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "place_pitch_cmd: ";
    rosidl_generator_traits::value_to_yaml(msg.place_pitch_cmd, out);
    out << "\n";
  }

  // member: pose_cmd
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "pose_cmd: ";
    rosidl_generator_traits::value_to_yaml(msg.pose_cmd, out);
    out << "\n";
  }

  // member: move_type_cmd
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "move_type_cmd: ";
    rosidl_generator_traits::value_to_yaml(msg.move_type_cmd, out);
    out << "\n";
  }

  // member: gait_toggle_cmd
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "gait_toggle_cmd: ";
    rosidl_generator_traits::value_to_yaml(msg.gait_toggle_cmd, out);
    out << "\n";
  }

  // member: leg_toggle_cmd
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "leg_toggle_cmd: ";
    rosidl_generator_traits::value_to_yaml(msg.leg_toggle_cmd, out);
    out << "\n";
  }

  // member: stance_cmd
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "stance_cmd: ";
    rosidl_generator_traits::value_to_yaml(msg.stance_cmd, out);
    out << "\n";
  }

  // member: reboot_cmd
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "reboot_cmd: ";
    rosidl_generator_traits::value_to_yaml(msg.reboot_cmd, out);
    out << "\n";
  }

  // member: set_home_pose_cmd
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "set_home_pose_cmd: ";
    rosidl_generator_traits::value_to_yaml(msg.set_home_pose_cmd, out);
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

inline std::string to_yaml(const HexJoy & msg, bool use_flow_style = false)
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
  const interbotix_xs_msgs::msg::HexJoy & msg,
  std::ostream & out, size_t indentation = 0)
{
  interbotix_xs_msgs::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use interbotix_xs_msgs::msg::to_yaml() instead")]]
inline std::string to_yaml(const interbotix_xs_msgs::msg::HexJoy & msg)
{
  return interbotix_xs_msgs::msg::to_yaml(msg);
}

template<>
inline const char * data_type<interbotix_xs_msgs::msg::HexJoy>()
{
  return "interbotix_xs_msgs::msg::HexJoy";
}

template<>
inline const char * name<interbotix_xs_msgs::msg::HexJoy>()
{
  return "interbotix_xs_msgs/msg/HexJoy";
}

template<>
struct has_fixed_size<interbotix_xs_msgs::msg::HexJoy>
  : std::integral_constant<bool, true> {};

template<>
struct has_bounded_size<interbotix_xs_msgs::msg::HexJoy>
  : std::integral_constant<bool, true> {};

template<>
struct is_message<interbotix_xs_msgs::msg::HexJoy>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // INTERBOTIX_XS_MSGS__MSG__DETAIL__HEX_JOY__TRAITS_HPP_
