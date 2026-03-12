// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from tidybot_msgs:srv/AudioRecord.idl
// generated code does not contain a copyright notice

#ifndef TIDYBOT_MSGS__SRV__DETAIL__AUDIO_RECORD__TRAITS_HPP_
#define TIDYBOT_MSGS__SRV__DETAIL__AUDIO_RECORD__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "tidybot_msgs/srv/detail/audio_record__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

namespace tidybot_msgs
{

namespace srv
{

inline void to_flow_style_yaml(
  const AudioRecord_Request & msg,
  std::ostream & out)
{
  out << "{";
  // member: start
  {
    out << "start: ";
    rosidl_generator_traits::value_to_yaml(msg.start, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const AudioRecord_Request & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: start
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "start: ";
    rosidl_generator_traits::value_to_yaml(msg.start, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const AudioRecord_Request & msg, bool use_flow_style = false)
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
  const tidybot_msgs::srv::AudioRecord_Request & msg,
  std::ostream & out, size_t indentation = 0)
{
  tidybot_msgs::srv::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use tidybot_msgs::srv::to_yaml() instead")]]
inline std::string to_yaml(const tidybot_msgs::srv::AudioRecord_Request & msg)
{
  return tidybot_msgs::srv::to_yaml(msg);
}

template<>
inline const char * data_type<tidybot_msgs::srv::AudioRecord_Request>()
{
  return "tidybot_msgs::srv::AudioRecord_Request";
}

template<>
inline const char * name<tidybot_msgs::srv::AudioRecord_Request>()
{
  return "tidybot_msgs/srv/AudioRecord_Request";
}

template<>
struct has_fixed_size<tidybot_msgs::srv::AudioRecord_Request>
  : std::integral_constant<bool, true> {};

template<>
struct has_bounded_size<tidybot_msgs::srv::AudioRecord_Request>
  : std::integral_constant<bool, true> {};

template<>
struct is_message<tidybot_msgs::srv::AudioRecord_Request>
  : std::true_type {};

}  // namespace rosidl_generator_traits

namespace tidybot_msgs
{

namespace srv
{

inline void to_flow_style_yaml(
  const AudioRecord_Response & msg,
  std::ostream & out)
{
  out << "{";
  // member: success
  {
    out << "success: ";
    rosidl_generator_traits::value_to_yaml(msg.success, out);
    out << ", ";
  }

  // member: message
  {
    out << "message: ";
    rosidl_generator_traits::value_to_yaml(msg.message, out);
    out << ", ";
  }

  // member: audio_data
  {
    if (msg.audio_data.size() == 0) {
      out << "audio_data: []";
    } else {
      out << "audio_data: [";
      size_t pending_items = msg.audio_data.size();
      for (auto item : msg.audio_data) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: sample_rate
  {
    out << "sample_rate: ";
    rosidl_generator_traits::value_to_yaml(msg.sample_rate, out);
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
  const AudioRecord_Response & msg,
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

  // member: message
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "message: ";
    rosidl_generator_traits::value_to_yaml(msg.message, out);
    out << "\n";
  }

  // member: audio_data
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.audio_data.size() == 0) {
      out << "audio_data: []\n";
    } else {
      out << "audio_data:\n";
      for (auto item : msg.audio_data) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: sample_rate
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "sample_rate: ";
    rosidl_generator_traits::value_to_yaml(msg.sample_rate, out);
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
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const AudioRecord_Response & msg, bool use_flow_style = false)
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
  const tidybot_msgs::srv::AudioRecord_Response & msg,
  std::ostream & out, size_t indentation = 0)
{
  tidybot_msgs::srv::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use tidybot_msgs::srv::to_yaml() instead")]]
inline std::string to_yaml(const tidybot_msgs::srv::AudioRecord_Response & msg)
{
  return tidybot_msgs::srv::to_yaml(msg);
}

template<>
inline const char * data_type<tidybot_msgs::srv::AudioRecord_Response>()
{
  return "tidybot_msgs::srv::AudioRecord_Response";
}

template<>
inline const char * name<tidybot_msgs::srv::AudioRecord_Response>()
{
  return "tidybot_msgs/srv/AudioRecord_Response";
}

template<>
struct has_fixed_size<tidybot_msgs::srv::AudioRecord_Response>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<tidybot_msgs::srv::AudioRecord_Response>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<tidybot_msgs::srv::AudioRecord_Response>
  : std::true_type {};

}  // namespace rosidl_generator_traits

namespace rosidl_generator_traits
{

template<>
inline const char * data_type<tidybot_msgs::srv::AudioRecord>()
{
  return "tidybot_msgs::srv::AudioRecord";
}

template<>
inline const char * name<tidybot_msgs::srv::AudioRecord>()
{
  return "tidybot_msgs/srv/AudioRecord";
}

template<>
struct has_fixed_size<tidybot_msgs::srv::AudioRecord>
  : std::integral_constant<
    bool,
    has_fixed_size<tidybot_msgs::srv::AudioRecord_Request>::value &&
    has_fixed_size<tidybot_msgs::srv::AudioRecord_Response>::value
  >
{
};

template<>
struct has_bounded_size<tidybot_msgs::srv::AudioRecord>
  : std::integral_constant<
    bool,
    has_bounded_size<tidybot_msgs::srv::AudioRecord_Request>::value &&
    has_bounded_size<tidybot_msgs::srv::AudioRecord_Response>::value
  >
{
};

template<>
struct is_service<tidybot_msgs::srv::AudioRecord>
  : std::true_type
{
};

template<>
struct is_service_request<tidybot_msgs::srv::AudioRecord_Request>
  : std::true_type
{
};

template<>
struct is_service_response<tidybot_msgs::srv::AudioRecord_Response>
  : std::true_type
{
};

}  // namespace rosidl_generator_traits

#endif  // TIDYBOT_MSGS__SRV__DETAIL__AUDIO_RECORD__TRAITS_HPP_
