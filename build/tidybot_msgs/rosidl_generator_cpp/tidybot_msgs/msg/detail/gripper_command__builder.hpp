// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from tidybot_msgs:msg/GripperCommand.idl
// generated code does not contain a copyright notice

#ifndef TIDYBOT_MSGS__MSG__DETAIL__GRIPPER_COMMAND__BUILDER_HPP_
#define TIDYBOT_MSGS__MSG__DETAIL__GRIPPER_COMMAND__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "tidybot_msgs/msg/detail/gripper_command__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace tidybot_msgs
{

namespace msg
{

namespace builder
{

class Init_GripperCommand_effort
{
public:
  explicit Init_GripperCommand_effort(::tidybot_msgs::msg::GripperCommand & msg)
  : msg_(msg)
  {}
  ::tidybot_msgs::msg::GripperCommand effort(::tidybot_msgs::msg::GripperCommand::_effort_type arg)
  {
    msg_.effort = std::move(arg);
    return std::move(msg_);
  }

private:
  ::tidybot_msgs::msg::GripperCommand msg_;
};

class Init_GripperCommand_position
{
public:
  explicit Init_GripperCommand_position(::tidybot_msgs::msg::GripperCommand & msg)
  : msg_(msg)
  {}
  Init_GripperCommand_effort position(::tidybot_msgs::msg::GripperCommand::_position_type arg)
  {
    msg_.position = std::move(arg);
    return Init_GripperCommand_effort(msg_);
  }

private:
  ::tidybot_msgs::msg::GripperCommand msg_;
};

class Init_GripperCommand_header
{
public:
  Init_GripperCommand_header()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_GripperCommand_position header(::tidybot_msgs::msg::GripperCommand::_header_type arg)
  {
    msg_.header = std::move(arg);
    return Init_GripperCommand_position(msg_);
  }

private:
  ::tidybot_msgs::msg::GripperCommand msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::tidybot_msgs::msg::GripperCommand>()
{
  return tidybot_msgs::msg::builder::Init_GripperCommand_header();
}

}  // namespace tidybot_msgs

#endif  // TIDYBOT_MSGS__MSG__DETAIL__GRIPPER_COMMAND__BUILDER_HPP_
