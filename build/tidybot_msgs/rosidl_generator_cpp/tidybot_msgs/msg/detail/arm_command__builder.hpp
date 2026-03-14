// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from tidybot_msgs:msg/ArmCommand.idl
// generated code does not contain a copyright notice

#ifndef TIDYBOT_MSGS__MSG__DETAIL__ARM_COMMAND__BUILDER_HPP_
#define TIDYBOT_MSGS__MSG__DETAIL__ARM_COMMAND__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "tidybot_msgs/msg/detail/arm_command__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace tidybot_msgs
{

namespace msg
{

namespace builder
{

class Init_ArmCommand_duration
{
public:
  explicit Init_ArmCommand_duration(::tidybot_msgs::msg::ArmCommand & msg)
  : msg_(msg)
  {}
  ::tidybot_msgs::msg::ArmCommand duration(::tidybot_msgs::msg::ArmCommand::_duration_type arg)
  {
    msg_.duration = std::move(arg);
    return std::move(msg_);
  }

private:
  ::tidybot_msgs::msg::ArmCommand msg_;
};

class Init_ArmCommand_joint_positions
{
public:
  explicit Init_ArmCommand_joint_positions(::tidybot_msgs::msg::ArmCommand & msg)
  : msg_(msg)
  {}
  Init_ArmCommand_duration joint_positions(::tidybot_msgs::msg::ArmCommand::_joint_positions_type arg)
  {
    msg_.joint_positions = std::move(arg);
    return Init_ArmCommand_duration(msg_);
  }

private:
  ::tidybot_msgs::msg::ArmCommand msg_;
};

class Init_ArmCommand_header
{
public:
  Init_ArmCommand_header()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_ArmCommand_joint_positions header(::tidybot_msgs::msg::ArmCommand::_header_type arg)
  {
    msg_.header = std::move(arg);
    return Init_ArmCommand_joint_positions(msg_);
  }

private:
  ::tidybot_msgs::msg::ArmCommand msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::tidybot_msgs::msg::ArmCommand>()
{
  return tidybot_msgs::msg::builder::Init_ArmCommand_header();
}

}  // namespace tidybot_msgs

#endif  // TIDYBOT_MSGS__MSG__DETAIL__ARM_COMMAND__BUILDER_HPP_
