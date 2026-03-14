// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from tidybot_msgs:srv/AudioRecord.idl
// generated code does not contain a copyright notice

#ifndef TIDYBOT_MSGS__SRV__DETAIL__AUDIO_RECORD__BUILDER_HPP_
#define TIDYBOT_MSGS__SRV__DETAIL__AUDIO_RECORD__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "tidybot_msgs/srv/detail/audio_record__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace tidybot_msgs
{

namespace srv
{

namespace builder
{

class Init_AudioRecord_Request_start
{
public:
  Init_AudioRecord_Request_start()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::tidybot_msgs::srv::AudioRecord_Request start(::tidybot_msgs::srv::AudioRecord_Request::_start_type arg)
  {
    msg_.start = std::move(arg);
    return std::move(msg_);
  }

private:
  ::tidybot_msgs::srv::AudioRecord_Request msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::tidybot_msgs::srv::AudioRecord_Request>()
{
  return tidybot_msgs::srv::builder::Init_AudioRecord_Request_start();
}

}  // namespace tidybot_msgs


namespace tidybot_msgs
{

namespace srv
{

namespace builder
{

class Init_AudioRecord_Response_duration
{
public:
  explicit Init_AudioRecord_Response_duration(::tidybot_msgs::srv::AudioRecord_Response & msg)
  : msg_(msg)
  {}
  ::tidybot_msgs::srv::AudioRecord_Response duration(::tidybot_msgs::srv::AudioRecord_Response::_duration_type arg)
  {
    msg_.duration = std::move(arg);
    return std::move(msg_);
  }

private:
  ::tidybot_msgs::srv::AudioRecord_Response msg_;
};

class Init_AudioRecord_Response_sample_rate
{
public:
  explicit Init_AudioRecord_Response_sample_rate(::tidybot_msgs::srv::AudioRecord_Response & msg)
  : msg_(msg)
  {}
  Init_AudioRecord_Response_duration sample_rate(::tidybot_msgs::srv::AudioRecord_Response::_sample_rate_type arg)
  {
    msg_.sample_rate = std::move(arg);
    return Init_AudioRecord_Response_duration(msg_);
  }

private:
  ::tidybot_msgs::srv::AudioRecord_Response msg_;
};

class Init_AudioRecord_Response_audio_data
{
public:
  explicit Init_AudioRecord_Response_audio_data(::tidybot_msgs::srv::AudioRecord_Response & msg)
  : msg_(msg)
  {}
  Init_AudioRecord_Response_sample_rate audio_data(::tidybot_msgs::srv::AudioRecord_Response::_audio_data_type arg)
  {
    msg_.audio_data = std::move(arg);
    return Init_AudioRecord_Response_sample_rate(msg_);
  }

private:
  ::tidybot_msgs::srv::AudioRecord_Response msg_;
};

class Init_AudioRecord_Response_message
{
public:
  explicit Init_AudioRecord_Response_message(::tidybot_msgs::srv::AudioRecord_Response & msg)
  : msg_(msg)
  {}
  Init_AudioRecord_Response_audio_data message(::tidybot_msgs::srv::AudioRecord_Response::_message_type arg)
  {
    msg_.message = std::move(arg);
    return Init_AudioRecord_Response_audio_data(msg_);
  }

private:
  ::tidybot_msgs::srv::AudioRecord_Response msg_;
};

class Init_AudioRecord_Response_success
{
public:
  Init_AudioRecord_Response_success()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_AudioRecord_Response_message success(::tidybot_msgs::srv::AudioRecord_Response::_success_type arg)
  {
    msg_.success = std::move(arg);
    return Init_AudioRecord_Response_message(msg_);
  }

private:
  ::tidybot_msgs::srv::AudioRecord_Response msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::tidybot_msgs::srv::AudioRecord_Response>()
{
  return tidybot_msgs::srv::builder::Init_AudioRecord_Response_success();
}

}  // namespace tidybot_msgs

#endif  // TIDYBOT_MSGS__SRV__DETAIL__AUDIO_RECORD__BUILDER_HPP_
