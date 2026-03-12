// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from tidybot_msgs:msg/PanTilt.idl
// generated code does not contain a copyright notice

#ifndef TIDYBOT_MSGS__MSG__DETAIL__PAN_TILT__STRUCT_HPP_
#define TIDYBOT_MSGS__MSG__DETAIL__PAN_TILT__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


// Include directives for member types
// Member 'header'
#include "std_msgs/msg/detail/header__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__tidybot_msgs__msg__PanTilt __attribute__((deprecated))
#else
# define DEPRECATED__tidybot_msgs__msg__PanTilt __declspec(deprecated)
#endif

namespace tidybot_msgs
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct PanTilt_
{
  using Type = PanTilt_<ContainerAllocator>;

  explicit PanTilt_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : header(_init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->pan = 0.0;
      this->tilt = 0.0;
    }
  }

  explicit PanTilt_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : header(_alloc, _init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->pan = 0.0;
      this->tilt = 0.0;
    }
  }

  // field types and members
  using _header_type =
    std_msgs::msg::Header_<ContainerAllocator>;
  _header_type header;
  using _pan_type =
    double;
  _pan_type pan;
  using _tilt_type =
    double;
  _tilt_type tilt;

  // setters for named parameter idiom
  Type & set__header(
    const std_msgs::msg::Header_<ContainerAllocator> & _arg)
  {
    this->header = _arg;
    return *this;
  }
  Type & set__pan(
    const double & _arg)
  {
    this->pan = _arg;
    return *this;
  }
  Type & set__tilt(
    const double & _arg)
  {
    this->tilt = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    tidybot_msgs::msg::PanTilt_<ContainerAllocator> *;
  using ConstRawPtr =
    const tidybot_msgs::msg::PanTilt_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<tidybot_msgs::msg::PanTilt_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<tidybot_msgs::msg::PanTilt_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      tidybot_msgs::msg::PanTilt_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<tidybot_msgs::msg::PanTilt_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      tidybot_msgs::msg::PanTilt_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<tidybot_msgs::msg::PanTilt_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<tidybot_msgs::msg::PanTilt_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<tidybot_msgs::msg::PanTilt_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__tidybot_msgs__msg__PanTilt
    std::shared_ptr<tidybot_msgs::msg::PanTilt_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__tidybot_msgs__msg__PanTilt
    std::shared_ptr<tidybot_msgs::msg::PanTilt_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const PanTilt_ & other) const
  {
    if (this->header != other.header) {
      return false;
    }
    if (this->pan != other.pan) {
      return false;
    }
    if (this->tilt != other.tilt) {
      return false;
    }
    return true;
  }
  bool operator!=(const PanTilt_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct PanTilt_

// alias to use template instance with default allocator
using PanTilt =
  tidybot_msgs::msg::PanTilt_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace tidybot_msgs

#endif  // TIDYBOT_MSGS__MSG__DETAIL__PAN_TILT__STRUCT_HPP_
