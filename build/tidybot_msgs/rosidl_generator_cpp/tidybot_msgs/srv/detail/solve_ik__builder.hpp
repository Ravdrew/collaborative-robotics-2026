// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from tidybot_msgs:srv/SolveIK.idl
// generated code does not contain a copyright notice

#ifndef TIDYBOT_MSGS__SRV__DETAIL__SOLVE_IK__BUILDER_HPP_
#define TIDYBOT_MSGS__SRV__DETAIL__SOLVE_IK__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "tidybot_msgs/srv/detail/solve_ik__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace tidybot_msgs
{

namespace srv
{

namespace builder
{

class Init_SolveIK_Request_timeout
{
public:
  explicit Init_SolveIK_Request_timeout(::tidybot_msgs::srv::SolveIK_Request & msg)
  : msg_(msg)
  {}
  ::tidybot_msgs::srv::SolveIK_Request timeout(::tidybot_msgs::srv::SolveIK_Request::_timeout_type arg)
  {
    msg_.timeout = std::move(arg);
    return std::move(msg_);
  }

private:
  ::tidybot_msgs::srv::SolveIK_Request msg_;
};

class Init_SolveIK_Request_use_orientation
{
public:
  explicit Init_SolveIK_Request_use_orientation(::tidybot_msgs::srv::SolveIK_Request & msg)
  : msg_(msg)
  {}
  Init_SolveIK_Request_timeout use_orientation(::tidybot_msgs::srv::SolveIK_Request::_use_orientation_type arg)
  {
    msg_.use_orientation = std::move(arg);
    return Init_SolveIK_Request_timeout(msg_);
  }

private:
  ::tidybot_msgs::srv::SolveIK_Request msg_;
};

class Init_SolveIK_Request_seed_joint_positions
{
public:
  explicit Init_SolveIK_Request_seed_joint_positions(::tidybot_msgs::srv::SolveIK_Request & msg)
  : msg_(msg)
  {}
  Init_SolveIK_Request_use_orientation seed_joint_positions(::tidybot_msgs::srv::SolveIK_Request::_seed_joint_positions_type arg)
  {
    msg_.seed_joint_positions = std::move(arg);
    return Init_SolveIK_Request_use_orientation(msg_);
  }

private:
  ::tidybot_msgs::srv::SolveIK_Request msg_;
};

class Init_SolveIK_Request_target_pose
{
public:
  explicit Init_SolveIK_Request_target_pose(::tidybot_msgs::srv::SolveIK_Request & msg)
  : msg_(msg)
  {}
  Init_SolveIK_Request_seed_joint_positions target_pose(::tidybot_msgs::srv::SolveIK_Request::_target_pose_type arg)
  {
    msg_.target_pose = std::move(arg);
    return Init_SolveIK_Request_seed_joint_positions(msg_);
  }

private:
  ::tidybot_msgs::srv::SolveIK_Request msg_;
};

class Init_SolveIK_Request_arm_name
{
public:
  Init_SolveIK_Request_arm_name()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_SolveIK_Request_target_pose arm_name(::tidybot_msgs::srv::SolveIK_Request::_arm_name_type arg)
  {
    msg_.arm_name = std::move(arg);
    return Init_SolveIK_Request_target_pose(msg_);
  }

private:
  ::tidybot_msgs::srv::SolveIK_Request msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::tidybot_msgs::srv::SolveIK_Request>()
{
  return tidybot_msgs::srv::builder::Init_SolveIK_Request_arm_name();
}

}  // namespace tidybot_msgs


namespace tidybot_msgs
{

namespace srv
{

namespace builder
{

class Init_SolveIK_Response_message
{
public:
  explicit Init_SolveIK_Response_message(::tidybot_msgs::srv::SolveIK_Response & msg)
  : msg_(msg)
  {}
  ::tidybot_msgs::srv::SolveIK_Response message(::tidybot_msgs::srv::SolveIK_Response::_message_type arg)
  {
    msg_.message = std::move(arg);
    return std::move(msg_);
  }

private:
  ::tidybot_msgs::srv::SolveIK_Response msg_;
};

class Init_SolveIK_Response_orientation_error
{
public:
  explicit Init_SolveIK_Response_orientation_error(::tidybot_msgs::srv::SolveIK_Response & msg)
  : msg_(msg)
  {}
  Init_SolveIK_Response_message orientation_error(::tidybot_msgs::srv::SolveIK_Response::_orientation_error_type arg)
  {
    msg_.orientation_error = std::move(arg);
    return Init_SolveIK_Response_message(msg_);
  }

private:
  ::tidybot_msgs::srv::SolveIK_Response msg_;
};

class Init_SolveIK_Response_position_error
{
public:
  explicit Init_SolveIK_Response_position_error(::tidybot_msgs::srv::SolveIK_Response & msg)
  : msg_(msg)
  {}
  Init_SolveIK_Response_orientation_error position_error(::tidybot_msgs::srv::SolveIK_Response::_position_error_type arg)
  {
    msg_.position_error = std::move(arg);
    return Init_SolveIK_Response_orientation_error(msg_);
  }

private:
  ::tidybot_msgs::srv::SolveIK_Response msg_;
};

class Init_SolveIK_Response_joint_positions
{
public:
  explicit Init_SolveIK_Response_joint_positions(::tidybot_msgs::srv::SolveIK_Response & msg)
  : msg_(msg)
  {}
  Init_SolveIK_Response_position_error joint_positions(::tidybot_msgs::srv::SolveIK_Response::_joint_positions_type arg)
  {
    msg_.joint_positions = std::move(arg);
    return Init_SolveIK_Response_position_error(msg_);
  }

private:
  ::tidybot_msgs::srv::SolveIK_Response msg_;
};

class Init_SolveIK_Response_success
{
public:
  Init_SolveIK_Response_success()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_SolveIK_Response_joint_positions success(::tidybot_msgs::srv::SolveIK_Response::_success_type arg)
  {
    msg_.success = std::move(arg);
    return Init_SolveIK_Response_joint_positions(msg_);
  }

private:
  ::tidybot_msgs::srv::SolveIK_Response msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::tidybot_msgs::srv::SolveIK_Response>()
{
  return tidybot_msgs::srv::builder::Init_SolveIK_Response_success();
}

}  // namespace tidybot_msgs

#endif  // TIDYBOT_MSGS__SRV__DETAIL__SOLVE_IK__BUILDER_HPP_
