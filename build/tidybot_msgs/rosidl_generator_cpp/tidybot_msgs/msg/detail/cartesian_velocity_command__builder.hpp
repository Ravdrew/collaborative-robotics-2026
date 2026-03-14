// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from tidybot_msgs:msg/CartesianVelocityCommand.idl
// generated code does not contain a copyright notice

#ifndef TIDYBOT_MSGS__MSG__DETAIL__CARTESIAN_VELOCITY_COMMAND__BUILDER_HPP_
#define TIDYBOT_MSGS__MSG__DETAIL__CARTESIAN_VELOCITY_COMMAND__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "tidybot_msgs/msg/detail/cartesian_velocity_command__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace tidybot_msgs
{

namespace msg
{

namespace builder
{

class Init_CartesianVelocityCommand_duration
{
public:
  explicit Init_CartesianVelocityCommand_duration(::tidybot_msgs::msg::CartesianVelocityCommand & msg)
  : msg_(msg)
  {}
  ::tidybot_msgs::msg::CartesianVelocityCommand duration(::tidybot_msgs::msg::CartesianVelocityCommand::_duration_type arg)
  {
    msg_.duration = std::move(arg);
    return std::move(msg_);
  }

private:
  ::tidybot_msgs::msg::CartesianVelocityCommand msg_;
};

class Init_CartesianVelocityCommand_frame_id
{
public:
  explicit Init_CartesianVelocityCommand_frame_id(::tidybot_msgs::msg::CartesianVelocityCommand & msg)
  : msg_(msg)
  {}
  Init_CartesianVelocityCommand_duration frame_id(::tidybot_msgs::msg::CartesianVelocityCommand::_frame_id_type arg)
  {
    msg_.frame_id = std::move(arg);
    return Init_CartesianVelocityCommand_duration(msg_);
  }

private:
  ::tidybot_msgs::msg::CartesianVelocityCommand msg_;
};

class Init_CartesianVelocityCommand_twist
{
public:
  explicit Init_CartesianVelocityCommand_twist(::tidybot_msgs::msg::CartesianVelocityCommand & msg)
  : msg_(msg)
  {}
  Init_CartesianVelocityCommand_frame_id twist(::tidybot_msgs::msg::CartesianVelocityCommand::_twist_type arg)
  {
    msg_.twist = std::move(arg);
    return Init_CartesianVelocityCommand_frame_id(msg_);
  }

private:
  ::tidybot_msgs::msg::CartesianVelocityCommand msg_;
};

class Init_CartesianVelocityCommand_header
{
public:
  Init_CartesianVelocityCommand_header()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_CartesianVelocityCommand_twist header(::tidybot_msgs::msg::CartesianVelocityCommand::_header_type arg)
  {
    msg_.header = std::move(arg);
    return Init_CartesianVelocityCommand_twist(msg_);
  }

private:
  ::tidybot_msgs::msg::CartesianVelocityCommand msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::tidybot_msgs::msg::CartesianVelocityCommand>()
{
  return tidybot_msgs::msg::builder::Init_CartesianVelocityCommand_header();
}

}  // namespace tidybot_msgs

#endif  // TIDYBOT_MSGS__MSG__DETAIL__CARTESIAN_VELOCITY_COMMAND__BUILDER_HPP_
