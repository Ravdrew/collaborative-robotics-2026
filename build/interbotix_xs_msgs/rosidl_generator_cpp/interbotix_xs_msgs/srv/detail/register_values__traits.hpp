// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from interbotix_xs_msgs:srv/RegisterValues.idl
// generated code does not contain a copyright notice

#ifndef INTERBOTIX_XS_MSGS__SRV__DETAIL__REGISTER_VALUES__TRAITS_HPP_
#define INTERBOTIX_XS_MSGS__SRV__DETAIL__REGISTER_VALUES__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "interbotix_xs_msgs/srv/detail/register_values__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

namespace interbotix_xs_msgs
{

namespace srv
{

inline void to_flow_style_yaml(
  const RegisterValues_Request & msg,
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
    out << ", ";
  }

  // member: reg
  {
    out << "reg: ";
    rosidl_generator_traits::value_to_yaml(msg.reg, out);
    out << ", ";
  }

  // member: value
  {
    out << "value: ";
    rosidl_generator_traits::value_to_yaml(msg.value, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const RegisterValues_Request & msg,
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

  // member: reg
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "reg: ";
    rosidl_generator_traits::value_to_yaml(msg.reg, out);
    out << "\n";
  }

  // member: value
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "value: ";
    rosidl_generator_traits::value_to_yaml(msg.value, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const RegisterValues_Request & msg, bool use_flow_style = false)
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
  const interbotix_xs_msgs::srv::RegisterValues_Request & msg,
  std::ostream & out, size_t indentation = 0)
{
  interbotix_xs_msgs::srv::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use interbotix_xs_msgs::srv::to_yaml() instead")]]
inline std::string to_yaml(const interbotix_xs_msgs::srv::RegisterValues_Request & msg)
{
  return interbotix_xs_msgs::srv::to_yaml(msg);
}

template<>
inline const char * data_type<interbotix_xs_msgs::srv::RegisterValues_Request>()
{
  return "interbotix_xs_msgs::srv::RegisterValues_Request";
}

template<>
inline const char * name<interbotix_xs_msgs::srv::RegisterValues_Request>()
{
  return "interbotix_xs_msgs/srv/RegisterValues_Request";
}

template<>
struct has_fixed_size<interbotix_xs_msgs::srv::RegisterValues_Request>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<interbotix_xs_msgs::srv::RegisterValues_Request>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<interbotix_xs_msgs::srv::RegisterValues_Request>
  : std::true_type {};

}  // namespace rosidl_generator_traits

namespace interbotix_xs_msgs
{

namespace srv
{

inline void to_flow_style_yaml(
  const RegisterValues_Response & msg,
  std::ostream & out)
{
  out << "{";
  // member: values
  {
    if (msg.values.size() == 0) {
      out << "values: []";
    } else {
      out << "values: [";
      size_t pending_items = msg.values.size();
      for (auto item : msg.values) {
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
  const RegisterValues_Response & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: values
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.values.size() == 0) {
      out << "values: []\n";
    } else {
      out << "values:\n";
      for (auto item : msg.values) {
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

inline std::string to_yaml(const RegisterValues_Response & msg, bool use_flow_style = false)
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
  const interbotix_xs_msgs::srv::RegisterValues_Response & msg,
  std::ostream & out, size_t indentation = 0)
{
  interbotix_xs_msgs::srv::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use interbotix_xs_msgs::srv::to_yaml() instead")]]
inline std::string to_yaml(const interbotix_xs_msgs::srv::RegisterValues_Response & msg)
{
  return interbotix_xs_msgs::srv::to_yaml(msg);
}

template<>
inline const char * data_type<interbotix_xs_msgs::srv::RegisterValues_Response>()
{
  return "interbotix_xs_msgs::srv::RegisterValues_Response";
}

template<>
inline const char * name<interbotix_xs_msgs::srv::RegisterValues_Response>()
{
  return "interbotix_xs_msgs/srv/RegisterValues_Response";
}

template<>
struct has_fixed_size<interbotix_xs_msgs::srv::RegisterValues_Response>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<interbotix_xs_msgs::srv::RegisterValues_Response>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<interbotix_xs_msgs::srv::RegisterValues_Response>
  : std::true_type {};

}  // namespace rosidl_generator_traits

namespace rosidl_generator_traits
{

template<>
inline const char * data_type<interbotix_xs_msgs::srv::RegisterValues>()
{
  return "interbotix_xs_msgs::srv::RegisterValues";
}

template<>
inline const char * name<interbotix_xs_msgs::srv::RegisterValues>()
{
  return "interbotix_xs_msgs/srv/RegisterValues";
}

template<>
struct has_fixed_size<interbotix_xs_msgs::srv::RegisterValues>
  : std::integral_constant<
    bool,
    has_fixed_size<interbotix_xs_msgs::srv::RegisterValues_Request>::value &&
    has_fixed_size<interbotix_xs_msgs::srv::RegisterValues_Response>::value
  >
{
};

template<>
struct has_bounded_size<interbotix_xs_msgs::srv::RegisterValues>
  : std::integral_constant<
    bool,
    has_bounded_size<interbotix_xs_msgs::srv::RegisterValues_Request>::value &&
    has_bounded_size<interbotix_xs_msgs::srv::RegisterValues_Response>::value
  >
{
};

template<>
struct is_service<interbotix_xs_msgs::srv::RegisterValues>
  : std::true_type
{
};

template<>
struct is_service_request<interbotix_xs_msgs::srv::RegisterValues_Request>
  : std::true_type
{
};

template<>
struct is_service_response<interbotix_xs_msgs::srv::RegisterValues_Response>
  : std::true_type
{
};

}  // namespace rosidl_generator_traits

#endif  // INTERBOTIX_XS_MSGS__SRV__DETAIL__REGISTER_VALUES__TRAITS_HPP_
