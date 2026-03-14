// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from tidybot_msgs:msg/CartesianVelocityCommand.idl
// generated code does not contain a copyright notice

#ifndef TIDYBOT_MSGS__MSG__DETAIL__CARTESIAN_VELOCITY_COMMAND__STRUCT_HPP_
#define TIDYBOT_MSGS__MSG__DETAIL__CARTESIAN_VELOCITY_COMMAND__STRUCT_HPP_

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
// Member 'twist'
#include "geometry_msgs/msg/detail/twist__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__tidybot_msgs__msg__CartesianVelocityCommand __attribute__((deprecated))
#else
# define DEPRECATED__tidybot_msgs__msg__CartesianVelocityCommand __declspec(deprecated)
#endif

namespace tidybot_msgs
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct CartesianVelocityCommand_
{
  using Type = CartesianVelocityCommand_<ContainerAllocator>;

  explicit CartesianVelocityCommand_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : header(_init),
    twist(_init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->frame_id = "";
      this->duration = 0.0;
    }
  }

  explicit CartesianVelocityCommand_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : header(_alloc, _init),
    twist(_alloc, _init),
    frame_id(_alloc)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->frame_id = "";
      this->duration = 0.0;
    }
  }

  // field types and members
  using _header_type =
    std_msgs::msg::Header_<ContainerAllocator>;
  _header_type header;
  using _twist_type =
    geometry_msgs::msg::Twist_<ContainerAllocator>;
  _twist_type twist;
  using _frame_id_type =
    std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>>;
  _frame_id_type frame_id;
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
  Type & set__twist(
    const geometry_msgs::msg::Twist_<ContainerAllocator> & _arg)
  {
    this->twist = _arg;
    return *this;
  }
  Type & set__frame_id(
    const std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>> & _arg)
  {
    this->frame_id = _arg;
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
    tidybot_msgs::msg::CartesianVelocityCommand_<ContainerAllocator> *;
  using ConstRawPtr =
    const tidybot_msgs::msg::CartesianVelocityCommand_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<tidybot_msgs::msg::CartesianVelocityCommand_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<tidybot_msgs::msg::CartesianVelocityCommand_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      tidybot_msgs::msg::CartesianVelocityCommand_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<tidybot_msgs::msg::CartesianVelocityCommand_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      tidybot_msgs::msg::CartesianVelocityCommand_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<tidybot_msgs::msg::CartesianVelocityCommand_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<tidybot_msgs::msg::CartesianVelocityCommand_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<tidybot_msgs::msg::CartesianVelocityCommand_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__tidybot_msgs__msg__CartesianVelocityCommand
    std::shared_ptr<tidybot_msgs::msg::CartesianVelocityCommand_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__tidybot_msgs__msg__CartesianVelocityCommand
    std::shared_ptr<tidybot_msgs::msg::CartesianVelocityCommand_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const CartesianVelocityCommand_ & other) const
  {
    if (this->header != other.header) {
      return false;
    }
    if (this->twist != other.twist) {
      return false;
    }
    if (this->frame_id != other.frame_id) {
      return false;
    }
    if (this->duration != other.duration) {
      return false;
    }
    return true;
  }
  bool operator!=(const CartesianVelocityCommand_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct CartesianVelocityCommand_

// alias to use template instance with default allocator
using CartesianVelocityCommand =
  tidybot_msgs::msg::CartesianVelocityCommand_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace tidybot_msgs

#endif  // TIDYBOT_MSGS__MSG__DETAIL__CARTESIAN_VELOCITY_COMMAND__STRUCT_HPP_
