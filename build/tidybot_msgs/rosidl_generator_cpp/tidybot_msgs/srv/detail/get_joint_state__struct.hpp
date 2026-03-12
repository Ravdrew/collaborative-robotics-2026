// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from tidybot_msgs:srv/GetJointState.idl
// generated code does not contain a copyright notice

#ifndef TIDYBOT_MSGS__SRV__DETAIL__GET_JOINT_STATE__STRUCT_HPP_
#define TIDYBOT_MSGS__SRV__DETAIL__GET_JOINT_STATE__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


#ifndef _WIN32
# define DEPRECATED__tidybot_msgs__srv__GetJointState_Request __attribute__((deprecated))
#else
# define DEPRECATED__tidybot_msgs__srv__GetJointState_Request __declspec(deprecated)
#endif

namespace tidybot_msgs
{

namespace srv
{

// message struct
template<class ContainerAllocator>
struct GetJointState_Request_
{
  using Type = GetJointState_Request_<ContainerAllocator>;

  explicit GetJointState_Request_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    (void)_init;
  }

  explicit GetJointState_Request_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    (void)_init;
    (void)_alloc;
  }

  // field types and members
  using _joint_names_type =
    std::vector<std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>>>>;
  _joint_names_type joint_names;

  // setters for named parameter idiom
  Type & set__joint_names(
    const std::vector<std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>>>> & _arg)
  {
    this->joint_names = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    tidybot_msgs::srv::GetJointState_Request_<ContainerAllocator> *;
  using ConstRawPtr =
    const tidybot_msgs::srv::GetJointState_Request_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<tidybot_msgs::srv::GetJointState_Request_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<tidybot_msgs::srv::GetJointState_Request_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      tidybot_msgs::srv::GetJointState_Request_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<tidybot_msgs::srv::GetJointState_Request_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      tidybot_msgs::srv::GetJointState_Request_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<tidybot_msgs::srv::GetJointState_Request_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<tidybot_msgs::srv::GetJointState_Request_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<tidybot_msgs::srv::GetJointState_Request_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__tidybot_msgs__srv__GetJointState_Request
    std::shared_ptr<tidybot_msgs::srv::GetJointState_Request_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__tidybot_msgs__srv__GetJointState_Request
    std::shared_ptr<tidybot_msgs::srv::GetJointState_Request_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const GetJointState_Request_ & other) const
  {
    if (this->joint_names != other.joint_names) {
      return false;
    }
    return true;
  }
  bool operator!=(const GetJointState_Request_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct GetJointState_Request_

// alias to use template instance with default allocator
using GetJointState_Request =
  tidybot_msgs::srv::GetJointState_Request_<std::allocator<void>>;

// constant definitions

}  // namespace srv

}  // namespace tidybot_msgs


// Include directives for member types
// Member 'state'
#include "sensor_msgs/msg/detail/joint_state__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__tidybot_msgs__srv__GetJointState_Response __attribute__((deprecated))
#else
# define DEPRECATED__tidybot_msgs__srv__GetJointState_Response __declspec(deprecated)
#endif

namespace tidybot_msgs
{

namespace srv
{

// message struct
template<class ContainerAllocator>
struct GetJointState_Response_
{
  using Type = GetJointState_Response_<ContainerAllocator>;

  explicit GetJointState_Response_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : state(_init)
  {
    (void)_init;
  }

  explicit GetJointState_Response_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : state(_alloc, _init)
  {
    (void)_init;
  }

  // field types and members
  using _state_type =
    sensor_msgs::msg::JointState_<ContainerAllocator>;
  _state_type state;

  // setters for named parameter idiom
  Type & set__state(
    const sensor_msgs::msg::JointState_<ContainerAllocator> & _arg)
  {
    this->state = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    tidybot_msgs::srv::GetJointState_Response_<ContainerAllocator> *;
  using ConstRawPtr =
    const tidybot_msgs::srv::GetJointState_Response_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<tidybot_msgs::srv::GetJointState_Response_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<tidybot_msgs::srv::GetJointState_Response_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      tidybot_msgs::srv::GetJointState_Response_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<tidybot_msgs::srv::GetJointState_Response_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      tidybot_msgs::srv::GetJointState_Response_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<tidybot_msgs::srv::GetJointState_Response_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<tidybot_msgs::srv::GetJointState_Response_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<tidybot_msgs::srv::GetJointState_Response_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__tidybot_msgs__srv__GetJointState_Response
    std::shared_ptr<tidybot_msgs::srv::GetJointState_Response_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__tidybot_msgs__srv__GetJointState_Response
    std::shared_ptr<tidybot_msgs::srv::GetJointState_Response_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const GetJointState_Response_ & other) const
  {
    if (this->state != other.state) {
      return false;
    }
    return true;
  }
  bool operator!=(const GetJointState_Response_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct GetJointState_Response_

// alias to use template instance with default allocator
using GetJointState_Response =
  tidybot_msgs::srv::GetJointState_Response_<std::allocator<void>>;

// constant definitions

}  // namespace srv

}  // namespace tidybot_msgs

namespace tidybot_msgs
{

namespace srv
{

struct GetJointState
{
  using Request = tidybot_msgs::srv::GetJointState_Request;
  using Response = tidybot_msgs::srv::GetJointState_Response;
};

}  // namespace srv

}  // namespace tidybot_msgs

#endif  // TIDYBOT_MSGS__SRV__DETAIL__GET_JOINT_STATE__STRUCT_HPP_
