// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from tidybot_msgs:srv/PlanToTarget.idl
// generated code does not contain a copyright notice

#ifndef TIDYBOT_MSGS__SRV__DETAIL__PLAN_TO_TARGET__BUILDER_HPP_
#define TIDYBOT_MSGS__SRV__DETAIL__PLAN_TO_TARGET__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "tidybot_msgs/srv/detail/plan_to_target__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace tidybot_msgs
{

namespace srv
{

namespace builder
{

class Init_PlanToTarget_Request_max_condition_number
{
public:
  explicit Init_PlanToTarget_Request_max_condition_number(::tidybot_msgs::srv::PlanToTarget_Request & msg)
  : msg_(msg)
  {}
  ::tidybot_msgs::srv::PlanToTarget_Request max_condition_number(::tidybot_msgs::srv::PlanToTarget_Request::_max_condition_number_type arg)
  {
    msg_.max_condition_number = std::move(arg);
    return std::move(msg_);
  }

private:
  ::tidybot_msgs::srv::PlanToTarget_Request msg_;
};

class Init_PlanToTarget_Request_duration
{
public:
  explicit Init_PlanToTarget_Request_duration(::tidybot_msgs::srv::PlanToTarget_Request & msg)
  : msg_(msg)
  {}
  Init_PlanToTarget_Request_max_condition_number duration(::tidybot_msgs::srv::PlanToTarget_Request::_duration_type arg)
  {
    msg_.duration = std::move(arg);
    return Init_PlanToTarget_Request_max_condition_number(msg_);
  }

private:
  ::tidybot_msgs::srv::PlanToTarget_Request msg_;
};

class Init_PlanToTarget_Request_execute
{
public:
  explicit Init_PlanToTarget_Request_execute(::tidybot_msgs::srv::PlanToTarget_Request & msg)
  : msg_(msg)
  {}
  Init_PlanToTarget_Request_duration execute(::tidybot_msgs::srv::PlanToTarget_Request::_execute_type arg)
  {
    msg_.execute = std::move(arg);
    return Init_PlanToTarget_Request_duration(msg_);
  }

private:
  ::tidybot_msgs::srv::PlanToTarget_Request msg_;
};

class Init_PlanToTarget_Request_use_orientation
{
public:
  explicit Init_PlanToTarget_Request_use_orientation(::tidybot_msgs::srv::PlanToTarget_Request & msg)
  : msg_(msg)
  {}
  Init_PlanToTarget_Request_execute use_orientation(::tidybot_msgs::srv::PlanToTarget_Request::_use_orientation_type arg)
  {
    msg_.use_orientation = std::move(arg);
    return Init_PlanToTarget_Request_execute(msg_);
  }

private:
  ::tidybot_msgs::srv::PlanToTarget_Request msg_;
};

class Init_PlanToTarget_Request_target_pose
{
public:
  explicit Init_PlanToTarget_Request_target_pose(::tidybot_msgs::srv::PlanToTarget_Request & msg)
  : msg_(msg)
  {}
  Init_PlanToTarget_Request_use_orientation target_pose(::tidybot_msgs::srv::PlanToTarget_Request::_target_pose_type arg)
  {
    msg_.target_pose = std::move(arg);
    return Init_PlanToTarget_Request_use_orientation(msg_);
  }

private:
  ::tidybot_msgs::srv::PlanToTarget_Request msg_;
};

class Init_PlanToTarget_Request_arm_name
{
public:
  Init_PlanToTarget_Request_arm_name()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_PlanToTarget_Request_target_pose arm_name(::tidybot_msgs::srv::PlanToTarget_Request::_arm_name_type arg)
  {
    msg_.arm_name = std::move(arg);
    return Init_PlanToTarget_Request_target_pose(msg_);
  }

private:
  ::tidybot_msgs::srv::PlanToTarget_Request msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::tidybot_msgs::srv::PlanToTarget_Request>()
{
  return tidybot_msgs::srv::builder::Init_PlanToTarget_Request_arm_name();
}

}  // namespace tidybot_msgs


namespace tidybot_msgs
{

namespace srv
{

namespace builder
{

class Init_PlanToTarget_Response_message
{
public:
  explicit Init_PlanToTarget_Response_message(::tidybot_msgs::srv::PlanToTarget_Response & msg)
  : msg_(msg)
  {}
  ::tidybot_msgs::srv::PlanToTarget_Response message(::tidybot_msgs::srv::PlanToTarget_Response::_message_type arg)
  {
    msg_.message = std::move(arg);
    return std::move(msg_);
  }

private:
  ::tidybot_msgs::srv::PlanToTarget_Response msg_;
};

class Init_PlanToTarget_Response_condition_number
{
public:
  explicit Init_PlanToTarget_Response_condition_number(::tidybot_msgs::srv::PlanToTarget_Response & msg)
  : msg_(msg)
  {}
  Init_PlanToTarget_Response_message condition_number(::tidybot_msgs::srv::PlanToTarget_Response::_condition_number_type arg)
  {
    msg_.condition_number = std::move(arg);
    return Init_PlanToTarget_Response_message(msg_);
  }

private:
  ::tidybot_msgs::srv::PlanToTarget_Response msg_;
};

class Init_PlanToTarget_Response_orientation_error
{
public:
  explicit Init_PlanToTarget_Response_orientation_error(::tidybot_msgs::srv::PlanToTarget_Response & msg)
  : msg_(msg)
  {}
  Init_PlanToTarget_Response_condition_number orientation_error(::tidybot_msgs::srv::PlanToTarget_Response::_orientation_error_type arg)
  {
    msg_.orientation_error = std::move(arg);
    return Init_PlanToTarget_Response_condition_number(msg_);
  }

private:
  ::tidybot_msgs::srv::PlanToTarget_Response msg_;
};

class Init_PlanToTarget_Response_position_error
{
public:
  explicit Init_PlanToTarget_Response_position_error(::tidybot_msgs::srv::PlanToTarget_Response & msg)
  : msg_(msg)
  {}
  Init_PlanToTarget_Response_orientation_error position_error(::tidybot_msgs::srv::PlanToTarget_Response::_position_error_type arg)
  {
    msg_.position_error = std::move(arg);
    return Init_PlanToTarget_Response_orientation_error(msg_);
  }

private:
  ::tidybot_msgs::srv::PlanToTarget_Response msg_;
};

class Init_PlanToTarget_Response_executed
{
public:
  explicit Init_PlanToTarget_Response_executed(::tidybot_msgs::srv::PlanToTarget_Response & msg)
  : msg_(msg)
  {}
  Init_PlanToTarget_Response_position_error executed(::tidybot_msgs::srv::PlanToTarget_Response::_executed_type arg)
  {
    msg_.executed = std::move(arg);
    return Init_PlanToTarget_Response_position_error(msg_);
  }

private:
  ::tidybot_msgs::srv::PlanToTarget_Response msg_;
};

class Init_PlanToTarget_Response_joint_positions
{
public:
  explicit Init_PlanToTarget_Response_joint_positions(::tidybot_msgs::srv::PlanToTarget_Response & msg)
  : msg_(msg)
  {}
  Init_PlanToTarget_Response_executed joint_positions(::tidybot_msgs::srv::PlanToTarget_Response::_joint_positions_type arg)
  {
    msg_.joint_positions = std::move(arg);
    return Init_PlanToTarget_Response_executed(msg_);
  }

private:
  ::tidybot_msgs::srv::PlanToTarget_Response msg_;
};

class Init_PlanToTarget_Response_success
{
public:
  Init_PlanToTarget_Response_success()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_PlanToTarget_Response_joint_positions success(::tidybot_msgs::srv::PlanToTarget_Response::_success_type arg)
  {
    msg_.success = std::move(arg);
    return Init_PlanToTarget_Response_joint_positions(msg_);
  }

private:
  ::tidybot_msgs::srv::PlanToTarget_Response msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::tidybot_msgs::srv::PlanToTarget_Response>()
{
  return tidybot_msgs::srv::builder::Init_PlanToTarget_Response_success();
}

}  // namespace tidybot_msgs

#endif  // TIDYBOT_MSGS__SRV__DETAIL__PLAN_TO_TARGET__BUILDER_HPP_
