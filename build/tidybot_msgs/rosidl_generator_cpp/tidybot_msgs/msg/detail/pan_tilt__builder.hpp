// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from tidybot_msgs:msg/PanTilt.idl
// generated code does not contain a copyright notice

#ifndef TIDYBOT_MSGS__MSG__DETAIL__PAN_TILT__BUILDER_HPP_
#define TIDYBOT_MSGS__MSG__DETAIL__PAN_TILT__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "tidybot_msgs/msg/detail/pan_tilt__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace tidybot_msgs
{

namespace msg
{

namespace builder
{

class Init_PanTilt_tilt
{
public:
  explicit Init_PanTilt_tilt(::tidybot_msgs::msg::PanTilt & msg)
  : msg_(msg)
  {}
  ::tidybot_msgs::msg::PanTilt tilt(::tidybot_msgs::msg::PanTilt::_tilt_type arg)
  {
    msg_.tilt = std::move(arg);
    return std::move(msg_);
  }

private:
  ::tidybot_msgs::msg::PanTilt msg_;
};

class Init_PanTilt_pan
{
public:
  explicit Init_PanTilt_pan(::tidybot_msgs::msg::PanTilt & msg)
  : msg_(msg)
  {}
  Init_PanTilt_tilt pan(::tidybot_msgs::msg::PanTilt::_pan_type arg)
  {
    msg_.pan = std::move(arg);
    return Init_PanTilt_tilt(msg_);
  }

private:
  ::tidybot_msgs::msg::PanTilt msg_;
};

class Init_PanTilt_header
{
public:
  Init_PanTilt_header()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_PanTilt_pan header(::tidybot_msgs::msg::PanTilt::_header_type arg)
  {
    msg_.header = std::move(arg);
    return Init_PanTilt_pan(msg_);
  }

private:
  ::tidybot_msgs::msg::PanTilt msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::tidybot_msgs::msg::PanTilt>()
{
  return tidybot_msgs::msg::builder::Init_PanTilt_header();
}

}  // namespace tidybot_msgs

#endif  // TIDYBOT_MSGS__MSG__DETAIL__PAN_TILT__BUILDER_HPP_
