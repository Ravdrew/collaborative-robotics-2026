// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from tidybot_msgs:msg/GripperCommand.idl
// generated code does not contain a copyright notice

#ifndef TIDYBOT_MSGS__MSG__DETAIL__GRIPPER_COMMAND__STRUCT_HPP_
#define TIDYBOT_MSGS__MSG__DETAIL__GRIPPER_COMMAND__STRUCT_HPP_

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
# define DEPRECATED__tidybot_msgs__msg__GripperCommand __attribute__((deprecated))
#else
# define DEPRECATED__tidybot_msgs__msg__GripperCommand __declspec(deprecated)
#endif

namespace tidybot_msgs
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct GripperCommand_
{
  using Type = GripperCommand_<ContainerAllocator>;

  explicit GripperCommand_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : header(_init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->position = 0.0;
      this->effort = 0.0;
    }
  }

  explicit GripperCommand_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : header(_alloc, _init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->position = 0.0;
      this->effort = 0.0;
    }
  }

  // field types and members
  using _header_type =
    std_msgs::msg::Header_<ContainerAllocator>;
  _header_type header;
  using _position_type =
    double;
  _position_type position;
  using _effort_type =
    double;
  _effort_type effort;

  // setters for named parameter idiom
  Type & set__header(
    const std_msgs::msg::Header_<ContainerAllocator> & _arg)
  {
    this->header = _arg;
    return *this;
  }
  Type & set__position(
    const double & _arg)
  {
    this->position = _arg;
    return *this;
  }
  Type & set__effort(
    const double & _arg)
  {
    this->effort = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    tidybot_msgs::msg::GripperCommand_<ContainerAllocator> *;
  using ConstRawPtr =
    const tidybot_msgs::msg::GripperCommand_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<tidybot_msgs::msg::GripperCommand_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<tidybot_msgs::msg::GripperCommand_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      tidybot_msgs::msg::GripperCommand_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<tidybot_msgs::msg::GripperCommand_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      tidybot_msgs::msg::GripperCommand_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<tidybot_msgs::msg::GripperCommand_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<tidybot_msgs::msg::GripperCommand_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<tidybot_msgs::msg::GripperCommand_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__tidybot_msgs__msg__GripperCommand
    std::shared_ptr<tidybot_msgs::msg::GripperCommand_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__tidybot_msgs__msg__GripperCommand
    std::shared_ptr<tidybot_msgs::msg::GripperCommand_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const GripperCommand_ & other) const
  {
    if (this->header != other.header) {
      return false;
    }
    if (this->position != other.position) {
      return false;
    }
    if (this->effort != other.effort) {
      return false;
    }
    return true;
  }
  bool operator!=(const GripperCommand_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct GripperCommand_

// alias to use template instance with default allocator
using GripperCommand =
  tidybot_msgs::msg::GripperCommand_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace tidybot_msgs

#endif  // TIDYBOT_MSGS__MSG__DETAIL__GRIPPER_COMMAND__STRUCT_HPP_
