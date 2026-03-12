// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from tidybot_msgs:msg/ArmVelocityCommand.idl
// generated code does not contain a copyright notice

#ifndef TIDYBOT_MSGS__MSG__DETAIL__ARM_VELOCITY_COMMAND__BUILDER_HPP_
#define TIDYBOT_MSGS__MSG__DETAIL__ARM_VELOCITY_COMMAND__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "tidybot_msgs/msg/detail/arm_velocity_command__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace tidybot_msgs
{

namespace msg
{

namespace builder
{

class Init_ArmVelocityCommand_duration
{
public:
  explicit Init_ArmVelocityCommand_duration(::tidybot_msgs::msg::ArmVelocityCommand & msg)
  : msg_(msg)
  {}
  ::tidybot_msgs::msg::ArmVelocityCommand duration(::tidybot_msgs::msg::ArmVelocityCommand::_duration_type arg)
  {
    msg_.duration = std::move(arg);
    return std::move(msg_);
  }

private:
  ::tidybot_msgs::msg::ArmVelocityCommand msg_;
};

class Init_ArmVelocityCommand_joint_velocities
{
public:
  explicit Init_ArmVelocityCommand_joint_velocities(::tidybot_msgs::msg::ArmVelocityCommand & msg)
  : msg_(msg)
  {}
  Init_ArmVelocityCommand_duration joint_velocities(::tidybot_msgs::msg::ArmVelocityCommand::_joint_velocities_type arg)
  {
    msg_.joint_velocities = std::move(arg);
    return Init_ArmVelocityCommand_duration(msg_);
  }

private:
  ::tidybot_msgs::msg::ArmVelocityCommand msg_;
};

class Init_ArmVelocityCommand_header
{
public:
  Init_ArmVelocityCommand_header()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_ArmVelocityCommand_joint_velocities header(::tidybot_msgs::msg::ArmVelocityCommand::_header_type arg)
  {
    msg_.header = std::move(arg);
    return Init_ArmVelocityCommand_joint_velocities(msg_);
  }

private:
  ::tidybot_msgs::msg::ArmVelocityCommand msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::tidybot_msgs::msg::ArmVelocityCommand>()
{
  return tidybot_msgs::msg::builder::Init_ArmVelocityCommand_header();
}

}  // namespace tidybot_msgs

#endif  // TIDYBOT_MSGS__MSG__DETAIL__ARM_VELOCITY_COMMAND__BUILDER_HPP_
