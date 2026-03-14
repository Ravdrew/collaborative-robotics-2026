// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from tidybot_msgs:srv/AudioRecord.idl
// generated code does not contain a copyright notice

#ifndef TIDYBOT_MSGS__SRV__DETAIL__AUDIO_RECORD__STRUCT_HPP_
#define TIDYBOT_MSGS__SRV__DETAIL__AUDIO_RECORD__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


#ifndef _WIN32
# define DEPRECATED__tidybot_msgs__srv__AudioRecord_Request __attribute__((deprecated))
#else
# define DEPRECATED__tidybot_msgs__srv__AudioRecord_Request __declspec(deprecated)
#endif

namespace tidybot_msgs
{

namespace srv
{

// message struct
template<class ContainerAllocator>
struct AudioRecord_Request_
{
  using Type = AudioRecord_Request_<ContainerAllocator>;

  explicit AudioRecord_Request_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->start = false;
    }
  }

  explicit AudioRecord_Request_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    (void)_alloc;
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->start = false;
    }
  }

  // field types and members
  using _start_type =
    bool;
  _start_type start;

  // setters for named parameter idiom
  Type & set__start(
    const bool & _arg)
  {
    this->start = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    tidybot_msgs::srv::AudioRecord_Request_<ContainerAllocator> *;
  using ConstRawPtr =
    const tidybot_msgs::srv::AudioRecord_Request_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<tidybot_msgs::srv::AudioRecord_Request_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<tidybot_msgs::srv::AudioRecord_Request_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      tidybot_msgs::srv::AudioRecord_Request_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<tidybot_msgs::srv::AudioRecord_Request_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      tidybot_msgs::srv::AudioRecord_Request_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<tidybot_msgs::srv::AudioRecord_Request_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<tidybot_msgs::srv::AudioRecord_Request_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<tidybot_msgs::srv::AudioRecord_Request_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__tidybot_msgs__srv__AudioRecord_Request
    std::shared_ptr<tidybot_msgs::srv::AudioRecord_Request_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__tidybot_msgs__srv__AudioRecord_Request
    std::shared_ptr<tidybot_msgs::srv::AudioRecord_Request_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const AudioRecord_Request_ & other) const
  {
    if (this->start != other.start) {
      return false;
    }
    return true;
  }
  bool operator!=(const AudioRecord_Request_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct AudioRecord_Request_

// alias to use template instance with default allocator
using AudioRecord_Request =
  tidybot_msgs::srv::AudioRecord_Request_<std::allocator<void>>;

// constant definitions

}  // namespace srv

}  // namespace tidybot_msgs


#ifndef _WIN32
# define DEPRECATED__tidybot_msgs__srv__AudioRecord_Response __attribute__((deprecated))
#else
# define DEPRECATED__tidybot_msgs__srv__AudioRecord_Response __declspec(deprecated)
#endif

namespace tidybot_msgs
{

namespace srv
{

// message struct
template<class ContainerAllocator>
struct AudioRecord_Response_
{
  using Type = AudioRecord_Response_<ContainerAllocator>;

  explicit AudioRecord_Response_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->success = false;
      this->message = "";
      this->sample_rate = 0l;
      this->duration = 0.0;
    }
  }

  explicit AudioRecord_Response_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : message(_alloc)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->success = false;
      this->message = "";
      this->sample_rate = 0l;
      this->duration = 0.0;
    }
  }

  // field types and members
  using _success_type =
    bool;
  _success_type success;
  using _message_type =
    std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>>;
  _message_type message;
  using _audio_data_type =
    std::vector<float, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<float>>;
  _audio_data_type audio_data;
  using _sample_rate_type =
    int32_t;
  _sample_rate_type sample_rate;
  using _duration_type =
    double;
  _duration_type duration;

  // setters for named parameter idiom
  Type & set__success(
    const bool & _arg)
  {
    this->success = _arg;
    return *this;
  }
  Type & set__message(
    const std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>> & _arg)
  {
    this->message = _arg;
    return *this;
  }
  Type & set__audio_data(
    const std::vector<float, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<float>> & _arg)
  {
    this->audio_data = _arg;
    return *this;
  }
  Type & set__sample_rate(
    const int32_t & _arg)
  {
    this->sample_rate = _arg;
    return *this;
  }
  Type & set__duration(
    const double & _arg)
  {
    this->duration = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    tidybot_msgs::srv::AudioRecord_Response_<ContainerAllocator> *;
  using ConstRawPtr =
    const tidybot_msgs::srv::AudioRecord_Response_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<tidybot_msgs::srv::AudioRecord_Response_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<tidybot_msgs::srv::AudioRecord_Response_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      tidybot_msgs::srv::AudioRecord_Response_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<tidybot_msgs::srv::AudioRecord_Response_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      tidybot_msgs::srv::AudioRecord_Response_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<tidybot_msgs::srv::AudioRecord_Response_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<tidybot_msgs::srv::AudioRecord_Response_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<tidybot_msgs::srv::AudioRecord_Response_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__tidybot_msgs__srv__AudioRecord_Response
    std::shared_ptr<tidybot_msgs::srv::AudioRecord_Response_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__tidybot_msgs__srv__AudioRecord_Response
    std::shared_ptr<tidybot_msgs::srv::AudioRecord_Response_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const AudioRecord_Response_ & other) const
  {
    if (this->success != other.success) {
      return false;
    }
    if (this->message != other.message) {
      return false;
    }
    if (this->audio_data != other.audio_data) {
      return false;
    }
    if (this->sample_rate != other.sample_rate) {
      return false;
    }
    if (this->duration != other.duration) {
      return false;
    }
    return true;
  }
  bool operator!=(const AudioRecord_Response_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct AudioRecord_Response_

// alias to use template instance with default allocator
using AudioRecord_Response =
  tidybot_msgs::srv::AudioRecord_Response_<std::allocator<void>>;

// constant definitions

}  // namespace srv

}  // namespace tidybot_msgs

namespace tidybot_msgs
{

namespace srv
{

struct AudioRecord
{
  using Request = tidybot_msgs::srv::AudioRecord_Request;
  using Response = tidybot_msgs::srv::AudioRecord_Response;
};

}  // namespace srv

}  // namespace tidybot_msgs

#endif  // TIDYBOT_MSGS__SRV__DETAIL__AUDIO_RECORD__STRUCT_HPP_
