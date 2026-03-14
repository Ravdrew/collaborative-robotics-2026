// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from tidybot_msgs:srv/SolveIK.idl
// generated code does not contain a copyright notice

#ifndef TIDYBOT_MSGS__SRV__DETAIL__SOLVE_IK__STRUCT_HPP_
#define TIDYBOT_MSGS__SRV__DETAIL__SOLVE_IK__STRUCT_HPP_

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
# define DEPRECATED__tidybot_msgs__srv__SolveIK_Request __attribute__((deprecated))
#else
# define DEPRECATED__tidybot_msgs__srv__SolveIK_Request __declspec(deprecated)
#endif

namespace tidybot_msgs
{

namespace srv
{

// message struct
template<class ContainerAllocator>
struct SolveIK_Request_
{
  using Type = SolveIK_Request_<ContainerAllocator>;

  explicit SolveIK_Request_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : target_pose(_init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->arm_name = "";
      this->use_orientation = false;
      this->timeout = 0.0;
    }
  }

  explicit SolveIK_Request_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : arm_name(_alloc),
    target_pose(_alloc, _init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->arm_name = "";
      this->use_orientation = false;
      this->timeout = 0.0;
    }
  }

  // field types and members
  using _arm_name_type =
    std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>>;
  _arm_name_type arm_name;
  using _target_pose_type =
    geometry_msgs::msg::Pose_<ContainerAllocator>;
  _target_pose_type target_pose;
  using _seed_joint_positions_type =
    std::vector<double, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<double>>;
  _seed_joint_positions_type seed_joint_positions;
  using _use_orientation_type =
    bool;
  _use_orientation_type use_orientation;
  using _timeout_type =
    double;
  _timeout_type timeout;

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
  Type & set__seed_joint_positions(
    const std::vector<double, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<double>> & _arg)
  {
    this->seed_joint_positions = _arg;
    return *this;
  }
  Type & set__use_orientation(
    const bool & _arg)
  {
    this->use_orientation = _arg;
    return *this;
  }
  Type & set__timeout(
    const double & _arg)
  {
    this->timeout = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    tidybot_msgs::srv::SolveIK_Request_<ContainerAllocator> *;
  using ConstRawPtr =
    const tidybot_msgs::srv::SolveIK_Request_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<tidybot_msgs::srv::SolveIK_Request_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<tidybot_msgs::srv::SolveIK_Request_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      tidybot_msgs::srv::SolveIK_Request_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<tidybot_msgs::srv::SolveIK_Request_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      tidybot_msgs::srv::SolveIK_Request_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<tidybot_msgs::srv::SolveIK_Request_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<tidybot_msgs::srv::SolveIK_Request_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<tidybot_msgs::srv::SolveIK_Request_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__tidybot_msgs__srv__SolveIK_Request
    std::shared_ptr<tidybot_msgs::srv::SolveIK_Request_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__tidybot_msgs__srv__SolveIK_Request
    std::shared_ptr<tidybot_msgs::srv::SolveIK_Request_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const SolveIK_Request_ & other) const
  {
    if (this->arm_name != other.arm_name) {
      return false;
    }
    if (this->target_pose != other.target_pose) {
      return false;
    }
    if (this->seed_joint_positions != other.seed_joint_positions) {
      return false;
    }
    if (this->use_orientation != other.use_orientation) {
      return false;
    }
    if (this->timeout != other.timeout) {
      return false;
    }
    return true;
  }
  bool operator!=(const SolveIK_Request_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct SolveIK_Request_

// alias to use template instance with default allocator
using SolveIK_Request =
  tidybot_msgs::srv::SolveIK_Request_<std::allocator<void>>;

// constant definitions

}  // namespace srv

}  // namespace tidybot_msgs


#ifndef _WIN32
# define DEPRECATED__tidybot_msgs__srv__SolveIK_Response __attribute__((deprecated))
#else
# define DEPRECATED__tidybot_msgs__srv__SolveIK_Response __declspec(deprecated)
#endif

namespace tidybot_msgs
{

namespace srv
{

// message struct
template<class ContainerAllocator>
struct SolveIK_Response_
{
  using Type = SolveIK_Response_<ContainerAllocator>;

  explicit SolveIK_Response_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->success = false;
      this->position_error = 0.0;
      this->orientation_error = 0.0;
      this->message = "";
    }
  }

  explicit SolveIK_Response_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : message(_alloc)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->success = false;
      this->position_error = 0.0;
      this->orientation_error = 0.0;
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
  using _position_error_type =
    double;
  _position_error_type position_error;
  using _orientation_error_type =
    double;
  _orientation_error_type orientation_error;
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
  Type & set__message(
    const std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>> & _arg)
  {
    this->message = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    tidybot_msgs::srv::SolveIK_Response_<ContainerAllocator> *;
  using ConstRawPtr =
    const tidybot_msgs::srv::SolveIK_Response_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<tidybot_msgs::srv::SolveIK_Response_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<tidybot_msgs::srv::SolveIK_Response_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      tidybot_msgs::srv::SolveIK_Response_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<tidybot_msgs::srv::SolveIK_Response_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      tidybot_msgs::srv::SolveIK_Response_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<tidybot_msgs::srv::SolveIK_Response_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<tidybot_msgs::srv::SolveIK_Response_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<tidybot_msgs::srv::SolveIK_Response_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__tidybot_msgs__srv__SolveIK_Response
    std::shared_ptr<tidybot_msgs::srv::SolveIK_Response_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__tidybot_msgs__srv__SolveIK_Response
    std::shared_ptr<tidybot_msgs::srv::SolveIK_Response_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const SolveIK_Response_ & other) const
  {
    if (this->success != other.success) {
      return false;
    }
    if (this->joint_positions != other.joint_positions) {
      return false;
    }
    if (this->position_error != other.position_error) {
      return false;
    }
    if (this->orientation_error != other.orientation_error) {
      return false;
    }
    if (this->message != other.message) {
      return false;
    }
    return true;
  }
  bool operator!=(const SolveIK_Response_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct SolveIK_Response_

// alias to use template instance with default allocator
using SolveIK_Response =
  tidybot_msgs::srv::SolveIK_Response_<std::allocator<void>>;

// constant definitions

}  // namespace srv

}  // namespace tidybot_msgs

namespace tidybot_msgs
{

namespace srv
{

struct SolveIK
{
  using Request = tidybot_msgs::srv::SolveIK_Request;
  using Response = tidybot_msgs::srv::SolveIK_Response;
};

}  // namespace srv

}  // namespace tidybot_msgs

#endif  // TIDYBOT_MSGS__SRV__DETAIL__SOLVE_IK__STRUCT_HPP_
