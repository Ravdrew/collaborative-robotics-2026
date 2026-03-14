// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from tidybot_msgs:msg/ArmVelocityCommand.idl
// generated code does not contain a copyright notice

#ifndef TIDYBOT_MSGS__MSG__DETAIL__ARM_VELOCITY_COMMAND__STRUCT_HPP_
#define TIDYBOT_MSGS__MSG__DETAIL__ARM_VELOCITY_COMMAND__STRUCT_HPP_

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
# define DEPRECATED__tidybot_msgs__msg__ArmVelocityCommand __attribute__((deprecated))
#else
# define DEPRECATED__tidybot_msgs__msg__ArmVelocityCommand __declspec(deprecated)
#endif

namespace tidybot_msgs
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct ArmVelocityCommand_
{
  using Type = ArmVelocityCommand_<ContainerAllocator>;

  explicit ArmVelocityCommand_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : header(_init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      std::fill<typename std::array<double, 6>::iterator, double>(this->joint_velocities.begin(), this->joint_velocities.end(), 0.0);
      this->duration = 0.0;
    }
  }

  explicit ArmVelocityCommand_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : header(_alloc, _init),
    joint_velocities(_alloc)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      std::fill<typename std::array<double, 6>::iterator, double>(this->joint_velocities.begin(), this->joint_velocities.end(), 0.0);
      this->duration = 0.0;
    }
  }

  // field types and members
  using _header_type =
    std_msgs::msg::Header_<ContainerAllocator>;
  _header_type header;
  using _joint_velocities_type =
    std::array<double, 6>;
  _joint_velocities_type joint_velocities;
  using _duration_type =
    double;
  _duration_type duration;

  // setters for named parameter idiom
  Type & set__header(
    const std_msgs::msg::Header_<ContainerAllocator> & _arg)
  {
    this->header = _arg;
    return *this;
  }
  Type & set__joint_velocities(
    const std::array<double, 6> & _arg)
  {
    this->joint_velocities = _arg;
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
    tidybot_msgs::msg::ArmVelocityCommand_<ContainerAllocator> *;
  using ConstRawPtr =
    const tidybot_msgs::msg::ArmVelocityCommand_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<tidybot_msgs::msg::ArmVelocityCommand_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<tidybot_msgs::msg::ArmVelocityCommand_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      tidybot_msgs::msg::ArmVelocityCommand_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<tidybot_msgs::msg::ArmVelocityCommand_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      tidybot_msgs::msg::ArmVelocityCommand_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<tidybot_msgs::msg::ArmVelocityCommand_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<tidybot_msgs::msg::ArmVelocityCommand_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<tidybot_msgs::msg::ArmVelocityCommand_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__tidybot_msgs__msg__ArmVelocityCommand
    std::shared_ptr<tidybot_msgs::msg::ArmVelocityCommand_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__tidybot_msgs__msg__ArmVelocityCommand
    std::shared_ptr<tidybot_msgs::msg::ArmVelocityCommand_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const ArmVelocityCommand_ & other) const
  {
    if (this->header != other.header) {
      return false;
    }
    if (this->joint_velocities != other.joint_velocities) {
      return false;
    }
    if (this->duration != other.duration) {
      return false;
    }
    return true;
  }
  bool operator!=(const ArmVelocityCommand_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct ArmVelocityCommand_

// alias to use template instance with default allocator
using ArmVelocityCommand =
  tidybot_msgs::msg::ArmVelocityCommand_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace tidybot_msgs

#endif  // TIDYBOT_MSGS__MSG__DETAIL__ARM_VELOCITY_COMMAND__STRUCT_HPP_
