// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from tidybot_msgs:srv/PlanToTarget.idl
// generated code does not contain a copyright notice

#ifndef TIDYBOT_MSGS__SRV__DETAIL__PLAN_TO_TARGET__STRUCT_HPP_
#define TIDYBOT_MSGS__SRV__DETAIL__PLAN_TO_TARGET__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


// Include directives for member types
// Member 'target_pose'
#include "geometry_msgs/msg/detail/pose__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__tidybot_msgs__srv__PlanToTarget_Request __attribute__((deprecated))
#else
# define DEPRECATED__tidybot_msgs__srv__PlanToTarget_Request __declspec(deprecated)
#endif

namespace tidybot_msgs
{

namespace srv
{

// message struct
template<class ContainerAllocator>
struct PlanToTarget_Request_
{
  using Type = PlanToTarget_Request_<ContainerAllocator>;

  explicit PlanToTarget_Request_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : target_pose(_init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::DEFAULTS_ONLY == _init)
    {
      this->use_orientation = true;
      this->execute = true;
      this->duration = 2.0;
      this->max_condition_number = 500.0;
    } else if (rosidl_runtime_cpp::MessageInitialization::ZERO == _init) {
      this->arm_name = "";
      this->use_orientation = false;
      this->execute = false;
      this->duration = 0.0;
      this->max_condition_number = 0.0;
    }
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->arm_name = "";
    }
  }

  explicit PlanToTarget_Request_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : arm_name(_alloc),
    target_pose(_alloc, _init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::DEFAULTS_ONLY == _init)
    {
      this->use_orientation = true;
      this->execute = true;
      this->duration = 2.0;
      this->max_condition_number = 500.0;
    } else if (rosidl_runtime_cpp::MessageInitialization::ZERO == _init) {
      this->arm_name = "";
      this->use_orientation = false;
      this->execute = false;
      this->duration = 0.0;
      this->max_condition_number = 0.0;
    }
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->arm_name = "";
    }
  }

  // field types and members
  using _arm_name_type =
    std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>>;
  _arm_name_type arm_name;
  using _target_pose_type =
    geometry_msgs::msg::Pose_<ContainerAllocator>;
  _target_pose_type target_pose;
  using _use_orientation_type =
    bool;
  _use_orientation_type use_orientation;
  using _execute_type =
    bool;
  _execute_type execute;
  using _duration_type =
    double;
  _duration_type duration;
  using _max_condition_number_type =
    double;
  _max_condition_number_type max_condition_number;

  // setters for named parameter idiom
  Type & set__arm_name(
    const std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>> & _arg)
  {
    this->arm_name = _arg;
    return *this;
  }
  Type & set__target_pose(
    const geometry_msgs::msg::Pose_<ContainerAllocator> & _arg)
  {
    this->target_pose = _arg;
    return *this;
  }
  Type & set__use_orientation(
    const bool & _arg)
  {
    this->use_orientation = _arg;
    return *this;
  }
  Type & set__execute(
    const bool & _arg)
  {
    this->execute = _arg;
    return *this;
  }
  Type & set__duration(
    const double & _arg)
  {
    this->duration = _arg;
    return *this;
  }
  Type & set__max_condition_number(
    const double & _arg)
  {
    this->max_condition_number = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    tidybot_msgs::srv::PlanToTarget_Request_<ContainerAllocator> *;
  using ConstRawPtr =
    const tidybot_msgs::srv::PlanToTarget_Request_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<tidybot_msgs::srv::PlanToTarget_Request_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<tidybot_msgs::srv::PlanToTarget_Request_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      tidybot_msgs::srv::PlanToTarget_Request_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<tidybot_msgs::srv::PlanToTarget_Request_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      tidybot_msgs::srv::PlanToTarget_Request_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<tidybot_msgs::srv::PlanToTarget_Request_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<tidybot_msgs::srv::PlanToTarget_Request_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<tidybot_msgs::srv::PlanToTarget_Request_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__tidybot_msgs__srv__PlanToTarget_Request
    std::shared_ptr<tidybot_msgs::srv::PlanToTarget_Request_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__tidybot_msgs__srv__PlanToTarget_Request
    std::shared_ptr<tidybot_msgs::srv::PlanToTarget_Request_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const PlanToTarget_Request_ & other) const
  {
    if (this->arm_name != other.arm_name) {
      return false;
    }
    if (this->target_pose != other.target_pose) {
      return false;
    }
    if (this->use_orientation != other.use_orientation) {
      return false;
    }
    if (this->execute != other.execute) {
      return false;
    }
    if (this->duration != other.duration) {
      return false;
    }
    if (this->max_condition_number != other.max_condition_number) {
      return false;
    }
    return true;
  }
  bool operator!=(const PlanToTarget_Request_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct PlanToTarget_Request_

// alias to use template instance with default allocator
using PlanToTarget_Request =
  tidybot_msgs::srv::PlanToTarget_Request_<std::allocator<void>>;

// constant definitions

}  // namespace srv

}  // namespace tidybot_msgs


#ifndef _WIN32
# define DEPRECATED__tidybot_msgs__srv__PlanToTarget_Response __attribute__((deprecated))
#else
# define DEPRECATED__tidybot_msgs__srv__PlanToTarget_Response __declspec(deprecated)
#endif

namespace tidybot_msgs
{

namespace srv
{

// message struct
template<class ContainerAllocator>
struct PlanToTarget_Response_
{
  using Type = PlanToTarget_Response_<ContainerAllocator>;

  explicit PlanToTarget_Response_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->success = false;
      this->executed = false;
      this->position_error = 0.0;
      this->orientation_error = 0.0;
      this->condition_number = 0.0;
      this->message = "";
    }
  }

  explicit PlanToTarget_Response_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : message(_alloc)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->success = false;
      this->executed = false;
      this->position_error = 0.0;
      this->orientation_error = 0.0;
      this->condition_number = 0.0;
      this->message = "";
    }
  }

  // field types and members
  using _success_type =
    bool;
  _success_type success;
  using _joint_positions_type =
    std::vector<double, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<double>>;
  _joint_positions_type joint_positions;
  using _executed_type =
    bool;
  _executed_type executed;
  using _position_error_type =
    double;
  _position_error_type position_error;
  using _orientation_error_type =
    double;
  _orientation_error_type orientation_error;
  using _condition_number_type =
    double;
  _condition_number_type condition_number;
  using _message_type =
    std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>>;
  _message_type message;

  // setters for named parameter idiom
  Type & set__success(
    const bool & _arg)
  {
    this->success = _arg;
    return *this;
  }
  Type & set__joint_positions(
    const std::vector<double, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<double>> & _arg)
  {
    this->joint_positions = _arg;
    return *this;
  }
  Type & set__executed(
    const bool & _arg)
  {
    this->executed = _arg;
    return *this;
  }
  Type & set__position_error(
    const double & _arg)
  {
    this->position_error = _arg;
    return *this;
  }
  Type & set__orientation_error(
    const double & _arg)
  {
    this->orientation_error = _arg;
    return *this;
  }
  Type & set__condition_number(
    const double & _arg)
  {
    this->condition_number = _arg;
    return *this;
  }
  Type & set__message(
    const std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>> & _arg)
  {
    this->message = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    tidybot_msgs::srv::PlanToTarget_Response_<ContainerAllocator> *;
  using ConstRawPtr =
    const tidybot_msgs::srv::PlanToTarget_Response_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<tidybot_msgs::srv::PlanToTarget_Response_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<tidybot_msgs::srv::PlanToTarget_Response_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      tidybot_msgs::srv::PlanToTarget_Response_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<tidybot_msgs::srv::PlanToTarget_Response_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      tidybot_msgs::srv::PlanToTarget_Response_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<tidybot_msgs::srv::PlanToTarget_Response_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<tidybot_msgs::srv::PlanToTarget_Response_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<tidybot_msgs::srv::PlanToTarget_Response_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__tidybot_msgs__srv__PlanToTarget_Response
    std::shared_ptr<tidybot_msgs::srv::PlanToTarget_Response_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__tidybot_msgs__srv__PlanToTarget_Response
    std::shared_ptr<tidybot_msgs::srv::PlanToTarget_Response_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const PlanToTarget_Response_ & other) const
  {
    if (this->success != other.success) {
      return false;
    }
    if (this->joint_positions != other.joint_positions) {
      return false;
    }
    if (this->executed != other.executed) {
      return false;
    }
    if (this->position_error != other.position_error) {
      return false;
    }
    if (this->orientation_error != other.orientation_error) {
      return false;
    }
    if (this->condition_number != other.condition_number) {
      return false;
    }
    if (this->message != other.message) {
      return false;
    }
    return true;
  }
  bool operator!=(const PlanToTarget_Response_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct PlanToTarget_Response_

// alias to use template instance with default allocator
using PlanToTarget_Response =
  tidybot_msgs::srv::PlanToTarget_Response_<std::allocator<void>>;

// constant definitions

}  // namespace srv

}  // namespace tidybot_msgs

namespace tidybot_msgs
{

namespace srv
{

struct PlanToTarget
{
  using Request = tidybot_msgs::srv::PlanToTarget_Request;
  using Response = tidybot_msgs::srv::PlanToTarget_Response;
};

}  // namespace srv

}  // namespace tidybot_msgs

#endif  // TIDYBOT_MSGS__SRV__DETAIL__PLAN_TO_TARGET__STRUCT_HPP_
