// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from tidybot_msgs:srv/GetJointState.idl
// generated code does not contain a copyright notice

#ifndef TIDYBOT_MSGS__SRV__DETAIL__GET_JOINT_STATE__BUILDER_HPP_
#define TIDYBOT_MSGS__SRV__DETAIL__GET_JOINT_STATE__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "tidybot_msgs/srv/detail/get_joint_state__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace tidybot_msgs
{

namespace srv
{

namespace builder
{

class Init_GetJointState_Request_joint_names
{
public:
  Init_GetJointState_Request_joint_names()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::tidybot_msgs::srv::GetJointState_Request joint_names(::tidybot_msgs::srv::GetJointState_Request::_joint_names_type arg)
  {
    msg_.joint_names = std::move(arg);
    return std::move(msg_);
  }

private:
  ::tidybot_msgs::srv::GetJointState_Request msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::tidybot_msgs::srv::GetJointState_Request>()
{
  return tidybot_msgs::srv::builder::Init_GetJointState_Request_joint_names();
}

}  // namespace tidybot_msgs


namespace tidybot_msgs
{

namespace srv
{

namespace builder
{

class Init_GetJointState_Response_state
{
public:
  Init_GetJointState_Response_state()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::tidybot_msgs::srv::GetJointState_Response state(::tidybot_msgs::srv::GetJointState_Response::_state_type arg)
  {
    msg_.state = std::move(arg);
    return std::move(msg_);
  }

private:
  ::tidybot_msgs::srv::GetJointState_Response msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::tidybot_msgs::srv::GetJointState_Response>()
{
  return tidybot_msgs::srv::builder::Init_GetJointState_Response_state();
}

}  // namespace tidybot_msgs

#endif  // TIDYBOT_MSGS__SRV__DETAIL__GET_JOINT_STATE__BUILDER_HPP_
