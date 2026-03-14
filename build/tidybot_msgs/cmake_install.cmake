# Install script for directory: /home/locobot/Desktop/team1_guinea_pigs/collaborative-robotics-2026/ros2_ws/src/tidybot_msgs

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/home/locobot/Desktop/team1_guinea_pigs/collaborative-robotics-2026/install/tidybot_msgs")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/ament_index/resource_index/rosidl_interfaces" TYPE FILE FILES "/home/locobot/Desktop/team1_guinea_pigs/collaborative-robotics-2026/build/tidybot_msgs/ament_cmake_index/share/ament_index/resource_index/rosidl_interfaces/tidybot_msgs")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/tidybot_msgs/tidybot_msgs" TYPE DIRECTORY FILES "/home/locobot/Desktop/team1_guinea_pigs/collaborative-robotics-2026/build/tidybot_msgs/rosidl_generator_c/tidybot_msgs/" REGEX "/[^/]*\\.h$")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/tidybot_msgs/environment" TYPE FILE FILES "/opt/ros/humble/lib/python3.10/site-packages/ament_package/template/environment_hook/library_path.sh")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/tidybot_msgs/environment" TYPE FILE FILES "/home/locobot/Desktop/team1_guinea_pigs/collaborative-robotics-2026/build/tidybot_msgs/ament_cmake_environment_hooks/library_path.dsv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libtidybot_msgs__rosidl_generator_c.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libtidybot_msgs__rosidl_generator_c.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libtidybot_msgs__rosidl_generator_c.so"
         RPATH "")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES "/home/locobot/Desktop/team1_guinea_pigs/collaborative-robotics-2026/build/tidybot_msgs/libtidybot_msgs__rosidl_generator_c.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libtidybot_msgs__rosidl_generator_c.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libtidybot_msgs__rosidl_generator_c.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libtidybot_msgs__rosidl_generator_c.so"
         OLD_RPATH "/opt/ros/humble/lib:"
         NEW_RPATH "")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libtidybot_msgs__rosidl_generator_c.so")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/tidybot_msgs/tidybot_msgs" TYPE DIRECTORY FILES "/home/locobot/Desktop/team1_guinea_pigs/collaborative-robotics-2026/build/tidybot_msgs/rosidl_typesupport_fastrtps_c/tidybot_msgs/" REGEX "/[^/]*\\.cpp$" EXCLUDE)
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libtidybot_msgs__rosidl_typesupport_fastrtps_c.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libtidybot_msgs__rosidl_typesupport_fastrtps_c.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libtidybot_msgs__rosidl_typesupport_fastrtps_c.so"
         RPATH "")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES "/home/locobot/Desktop/team1_guinea_pigs/collaborative-robotics-2026/build/tidybot_msgs/libtidybot_msgs__rosidl_typesupport_fastrtps_c.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libtidybot_msgs__rosidl_typesupport_fastrtps_c.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libtidybot_msgs__rosidl_typesupport_fastrtps_c.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libtidybot_msgs__rosidl_typesupport_fastrtps_c.so"
         OLD_RPATH "/home/locobot/Desktop/team1_guinea_pigs/collaborative-robotics-2026/build/tidybot_msgs:/opt/ros/humble/lib:"
         NEW_RPATH "")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libtidybot_msgs__rosidl_typesupport_fastrtps_c.so")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/tidybot_msgs/tidybot_msgs" TYPE DIRECTORY FILES "/home/locobot/Desktop/team1_guinea_pigs/collaborative-robotics-2026/build/tidybot_msgs/rosidl_typesupport_introspection_c/tidybot_msgs/" REGEX "/[^/]*\\.h$")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libtidybot_msgs__rosidl_typesupport_introspection_c.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libtidybot_msgs__rosidl_typesupport_introspection_c.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libtidybot_msgs__rosidl_typesupport_introspection_c.so"
         RPATH "")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES "/home/locobot/Desktop/team1_guinea_pigs/collaborative-robotics-2026/build/tidybot_msgs/libtidybot_msgs__rosidl_typesupport_introspection_c.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libtidybot_msgs__rosidl_typesupport_introspection_c.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libtidybot_msgs__rosidl_typesupport_introspection_c.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libtidybot_msgs__rosidl_typesupport_introspection_c.so"
         OLD_RPATH "/home/locobot/Desktop/team1_guinea_pigs/collaborative-robotics-2026/build/tidybot_msgs:/opt/ros/humble/lib:"
         NEW_RPATH "")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libtidybot_msgs__rosidl_typesupport_introspection_c.so")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libtidybot_msgs__rosidl_typesupport_c.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libtidybot_msgs__rosidl_typesupport_c.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libtidybot_msgs__rosidl_typesupport_c.so"
         RPATH "")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES "/home/locobot/Desktop/team1_guinea_pigs/collaborative-robotics-2026/build/tidybot_msgs/libtidybot_msgs__rosidl_typesupport_c.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libtidybot_msgs__rosidl_typesupport_c.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libtidybot_msgs__rosidl_typesupport_c.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libtidybot_msgs__rosidl_typesupport_c.so"
         OLD_RPATH "/home/locobot/Desktop/team1_guinea_pigs/collaborative-robotics-2026/build/tidybot_msgs:/opt/ros/humble/lib:"
         NEW_RPATH "")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libtidybot_msgs__rosidl_typesupport_c.so")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/tidybot_msgs/tidybot_msgs" TYPE DIRECTORY FILES "/home/locobot/Desktop/team1_guinea_pigs/collaborative-robotics-2026/build/tidybot_msgs/rosidl_generator_cpp/tidybot_msgs/" REGEX "/[^/]*\\.hpp$")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/tidybot_msgs/tidybot_msgs" TYPE DIRECTORY FILES "/home/locobot/Desktop/team1_guinea_pigs/collaborative-robotics-2026/build/tidybot_msgs/rosidl_typesupport_fastrtps_cpp/tidybot_msgs/" REGEX "/[^/]*\\.cpp$" EXCLUDE)
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libtidybot_msgs__rosidl_typesupport_fastrtps_cpp.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libtidybot_msgs__rosidl_typesupport_fastrtps_cpp.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libtidybot_msgs__rosidl_typesupport_fastrtps_cpp.so"
         RPATH "")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES "/home/locobot/Desktop/team1_guinea_pigs/collaborative-robotics-2026/build/tidybot_msgs/libtidybot_msgs__rosidl_typesupport_fastrtps_cpp.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libtidybot_msgs__rosidl_typesupport_fastrtps_cpp.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libtidybot_msgs__rosidl_typesupport_fastrtps_cpp.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libtidybot_msgs__rosidl_typesupport_fastrtps_cpp.so"
         OLD_RPATH "/opt/ros/humble/lib:"
         NEW_RPATH "")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libtidybot_msgs__rosidl_typesupport_fastrtps_cpp.so")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/tidybot_msgs/tidybot_msgs" TYPE DIRECTORY FILES "/home/locobot/Desktop/team1_guinea_pigs/collaborative-robotics-2026/build/tidybot_msgs/rosidl_typesupport_introspection_cpp/tidybot_msgs/" REGEX "/[^/]*\\.hpp$")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libtidybot_msgs__rosidl_typesupport_introspection_cpp.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libtidybot_msgs__rosidl_typesupport_introspection_cpp.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libtidybot_msgs__rosidl_typesupport_introspection_cpp.so"
         RPATH "")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES "/home/locobot/Desktop/team1_guinea_pigs/collaborative-robotics-2026/build/tidybot_msgs/libtidybot_msgs__rosidl_typesupport_introspection_cpp.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libtidybot_msgs__rosidl_typesupport_introspection_cpp.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libtidybot_msgs__rosidl_typesupport_introspection_cpp.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libtidybot_msgs__rosidl_typesupport_introspection_cpp.so"
         OLD_RPATH "/opt/ros/humble/lib:"
         NEW_RPATH "")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libtidybot_msgs__rosidl_typesupport_introspection_cpp.so")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libtidybot_msgs__rosidl_typesupport_cpp.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libtidybot_msgs__rosidl_typesupport_cpp.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libtidybot_msgs__rosidl_typesupport_cpp.so"
         RPATH "")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES "/home/locobot/Desktop/team1_guinea_pigs/collaborative-robotics-2026/build/tidybot_msgs/libtidybot_msgs__rosidl_typesupport_cpp.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libtidybot_msgs__rosidl_typesupport_cpp.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libtidybot_msgs__rosidl_typesupport_cpp.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libtidybot_msgs__rosidl_typesupport_cpp.so"
         OLD_RPATH "/opt/ros/humble/lib:"
         NEW_RPATH "")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libtidybot_msgs__rosidl_typesupport_cpp.so")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/tidybot_msgs/environment" TYPE FILE FILES "/home/locobot/Desktop/team1_guinea_pigs/collaborative-robotics-2026/build/tidybot_msgs/ament_cmake_environment_hooks/pythonpath.sh")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/tidybot_msgs/environment" TYPE FILE FILES "/home/locobot/Desktop/team1_guinea_pigs/collaborative-robotics-2026/build/tidybot_msgs/ament_cmake_environment_hooks/pythonpath.dsv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/local/lib/python3.10/dist-packages/tidybot_msgs-0.1.0-py3.10.egg-info" TYPE DIRECTORY FILES "/home/locobot/Desktop/team1_guinea_pigs/collaborative-robotics-2026/build/tidybot_msgs/ament_cmake_python/tidybot_msgs/tidybot_msgs.egg-info/")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/local/lib/python3.10/dist-packages/tidybot_msgs" TYPE DIRECTORY FILES "/home/locobot/Desktop/team1_guinea_pigs/collaborative-robotics-2026/build/tidybot_msgs/rosidl_generator_py/tidybot_msgs/" REGEX "/[^/]*\\.pyc$" EXCLUDE REGEX "/\\_\\_pycache\\_\\_$" EXCLUDE)
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  execute_process(
        COMMAND
        "/usr/bin/python3" "-m" "compileall"
        "/home/locobot/Desktop/team1_guinea_pigs/collaborative-robotics-2026/install/tidybot_msgs/local/lib/python3.10/dist-packages/tidybot_msgs"
      )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/local/lib/python3.10/dist-packages/tidybot_msgs/tidybot_msgs_s__rosidl_typesupport_fastrtps_c.cpython-310-x86_64-linux-gnu.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/local/lib/python3.10/dist-packages/tidybot_msgs/tidybot_msgs_s__rosidl_typesupport_fastrtps_c.cpython-310-x86_64-linux-gnu.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/local/lib/python3.10/dist-packages/tidybot_msgs/tidybot_msgs_s__rosidl_typesupport_fastrtps_c.cpython-310-x86_64-linux-gnu.so"
         RPATH "")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/local/lib/python3.10/dist-packages/tidybot_msgs" TYPE SHARED_LIBRARY FILES "/home/locobot/Desktop/team1_guinea_pigs/collaborative-robotics-2026/build/tidybot_msgs/rosidl_generator_py/tidybot_msgs/tidybot_msgs_s__rosidl_typesupport_fastrtps_c.cpython-310-x86_64-linux-gnu.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/local/lib/python3.10/dist-packages/tidybot_msgs/tidybot_msgs_s__rosidl_typesupport_fastrtps_c.cpython-310-x86_64-linux-gnu.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/local/lib/python3.10/dist-packages/tidybot_msgs/tidybot_msgs_s__rosidl_typesupport_fastrtps_c.cpython-310-x86_64-linux-gnu.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/local/lib/python3.10/dist-packages/tidybot_msgs/tidybot_msgs_s__rosidl_typesupport_fastrtps_c.cpython-310-x86_64-linux-gnu.so"
         OLD_RPATH "/home/locobot/Desktop/team1_guinea_pigs/collaborative-robotics-2026/build/tidybot_msgs/rosidl_generator_py/tidybot_msgs:/home/locobot/Desktop/team1_guinea_pigs/collaborative-robotics-2026/build/tidybot_msgs:/opt/ros/humble/lib:"
         NEW_RPATH "")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/local/lib/python3.10/dist-packages/tidybot_msgs/tidybot_msgs_s__rosidl_typesupport_fastrtps_c.cpython-310-x86_64-linux-gnu.so")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/local/lib/python3.10/dist-packages/tidybot_msgs/tidybot_msgs_s__rosidl_typesupport_introspection_c.cpython-310-x86_64-linux-gnu.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/local/lib/python3.10/dist-packages/tidybot_msgs/tidybot_msgs_s__rosidl_typesupport_introspection_c.cpython-310-x86_64-linux-gnu.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/local/lib/python3.10/dist-packages/tidybot_msgs/tidybot_msgs_s__rosidl_typesupport_introspection_c.cpython-310-x86_64-linux-gnu.so"
         RPATH "")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/local/lib/python3.10/dist-packages/tidybot_msgs" TYPE SHARED_LIBRARY FILES "/home/locobot/Desktop/team1_guinea_pigs/collaborative-robotics-2026/build/tidybot_msgs/rosidl_generator_py/tidybot_msgs/tidybot_msgs_s__rosidl_typesupport_introspection_c.cpython-310-x86_64-linux-gnu.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/local/lib/python3.10/dist-packages/tidybot_msgs/tidybot_msgs_s__rosidl_typesupport_introspection_c.cpython-310-x86_64-linux-gnu.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/local/lib/python3.10/dist-packages/tidybot_msgs/tidybot_msgs_s__rosidl_typesupport_introspection_c.cpython-310-x86_64-linux-gnu.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/local/lib/python3.10/dist-packages/tidybot_msgs/tidybot_msgs_s__rosidl_typesupport_introspection_c.cpython-310-x86_64-linux-gnu.so"
         OLD_RPATH "/home/locobot/Desktop/team1_guinea_pigs/collaborative-robotics-2026/build/tidybot_msgs/rosidl_generator_py/tidybot_msgs:/home/locobot/Desktop/team1_guinea_pigs/collaborative-robotics-2026/build/tidybot_msgs:/opt/ros/humble/lib:"
         NEW_RPATH "")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/local/lib/python3.10/dist-packages/tidybot_msgs/tidybot_msgs_s__rosidl_typesupport_introspection_c.cpython-310-x86_64-linux-gnu.so")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/local/lib/python3.10/dist-packages/tidybot_msgs/tidybot_msgs_s__rosidl_typesupport_c.cpython-310-x86_64-linux-gnu.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/local/lib/python3.10/dist-packages/tidybot_msgs/tidybot_msgs_s__rosidl_typesupport_c.cpython-310-x86_64-linux-gnu.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/local/lib/python3.10/dist-packages/tidybot_msgs/tidybot_msgs_s__rosidl_typesupport_c.cpython-310-x86_64-linux-gnu.so"
         RPATH "")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/local/lib/python3.10/dist-packages/tidybot_msgs" TYPE SHARED_LIBRARY FILES "/home/locobot/Desktop/team1_guinea_pigs/collaborative-robotics-2026/build/tidybot_msgs/rosidl_generator_py/tidybot_msgs/tidybot_msgs_s__rosidl_typesupport_c.cpython-310-x86_64-linux-gnu.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/local/lib/python3.10/dist-packages/tidybot_msgs/tidybot_msgs_s__rosidl_typesupport_c.cpython-310-x86_64-linux-gnu.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/local/lib/python3.10/dist-packages/tidybot_msgs/tidybot_msgs_s__rosidl_typesupport_c.cpython-310-x86_64-linux-gnu.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/local/lib/python3.10/dist-packages/tidybot_msgs/tidybot_msgs_s__rosidl_typesupport_c.cpython-310-x86_64-linux-gnu.so"
         OLD_RPATH "/home/locobot/Desktop/team1_guinea_pigs/collaborative-robotics-2026/build/tidybot_msgs/rosidl_generator_py/tidybot_msgs:/home/locobot/Desktop/team1_guinea_pigs/collaborative-robotics-2026/build/tidybot_msgs:/opt/ros/humble/lib:"
         NEW_RPATH "")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/local/lib/python3.10/dist-packages/tidybot_msgs/tidybot_msgs_s__rosidl_typesupport_c.cpython-310-x86_64-linux-gnu.so")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libtidybot_msgs__rosidl_generator_py.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libtidybot_msgs__rosidl_generator_py.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libtidybot_msgs__rosidl_generator_py.so"
         RPATH "")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES "/home/locobot/Desktop/team1_guinea_pigs/collaborative-robotics-2026/build/tidybot_msgs/rosidl_generator_py/tidybot_msgs/libtidybot_msgs__rosidl_generator_py.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libtidybot_msgs__rosidl_generator_py.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libtidybot_msgs__rosidl_generator_py.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libtidybot_msgs__rosidl_generator_py.so"
         OLD_RPATH "/home/locobot/Desktop/team1_guinea_pigs/collaborative-robotics-2026/build/tidybot_msgs:/opt/ros/humble/lib:"
         NEW_RPATH "")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libtidybot_msgs__rosidl_generator_py.so")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/tidybot_msgs/msg" TYPE FILE FILES "/home/locobot/Desktop/team1_guinea_pigs/collaborative-robotics-2026/build/tidybot_msgs/rosidl_adapter/tidybot_msgs/msg/ArmCommand.idl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/tidybot_msgs/msg" TYPE FILE FILES "/home/locobot/Desktop/team1_guinea_pigs/collaborative-robotics-2026/build/tidybot_msgs/rosidl_adapter/tidybot_msgs/msg/ArmVelocityCommand.idl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/tidybot_msgs/msg" TYPE FILE FILES "/home/locobot/Desktop/team1_guinea_pigs/collaborative-robotics-2026/build/tidybot_msgs/rosidl_adapter/tidybot_msgs/msg/CartesianVelocityCommand.idl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/tidybot_msgs/msg" TYPE FILE FILES "/home/locobot/Desktop/team1_guinea_pigs/collaborative-robotics-2026/build/tidybot_msgs/rosidl_adapter/tidybot_msgs/msg/GripperCommand.idl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/tidybot_msgs/msg" TYPE FILE FILES "/home/locobot/Desktop/team1_guinea_pigs/collaborative-robotics-2026/build/tidybot_msgs/rosidl_adapter/tidybot_msgs/msg/PanTilt.idl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/tidybot_msgs/srv" TYPE FILE FILES "/home/locobot/Desktop/team1_guinea_pigs/collaborative-robotics-2026/build/tidybot_msgs/rosidl_adapter/tidybot_msgs/srv/SolveIK.idl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/tidybot_msgs/srv" TYPE FILE FILES "/home/locobot/Desktop/team1_guinea_pigs/collaborative-robotics-2026/build/tidybot_msgs/rosidl_adapter/tidybot_msgs/srv/GetJointState.idl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/tidybot_msgs/srv" TYPE FILE FILES "/home/locobot/Desktop/team1_guinea_pigs/collaborative-robotics-2026/build/tidybot_msgs/rosidl_adapter/tidybot_msgs/srv/PlanToTarget.idl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/tidybot_msgs/srv" TYPE FILE FILES "/home/locobot/Desktop/team1_guinea_pigs/collaborative-robotics-2026/build/tidybot_msgs/rosidl_adapter/tidybot_msgs/srv/AudioRecord.idl")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/tidybot_msgs/msg" TYPE FILE FILES "/home/locobot/Desktop/team1_guinea_pigs/collaborative-robotics-2026/ros2_ws/src/tidybot_msgs/msg/ArmCommand.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/tidybot_msgs/msg" TYPE FILE FILES "/home/locobot/Desktop/team1_guinea_pigs/collaborative-robotics-2026/ros2_ws/src/tidybot_msgs/msg/ArmVelocityCommand.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/tidybot_msgs/msg" TYPE FILE FILES "/home/locobot/Desktop/team1_guinea_pigs/collaborative-robotics-2026/ros2_ws/src/tidybot_msgs/msg/CartesianVelocityCommand.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/tidybot_msgs/msg" TYPE FILE FILES "/home/locobot/Desktop/team1_guinea_pigs/collaborative-robotics-2026/ros2_ws/src/tidybot_msgs/msg/GripperCommand.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/tidybot_msgs/msg" TYPE FILE FILES "/home/locobot/Desktop/team1_guinea_pigs/collaborative-robotics-2026/ros2_ws/src/tidybot_msgs/msg/PanTilt.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/tidybot_msgs/srv" TYPE FILE FILES "/home/locobot/Desktop/team1_guinea_pigs/collaborative-robotics-2026/ros2_ws/src/tidybot_msgs/srv/SolveIK.srv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/tidybot_msgs/srv" TYPE FILE FILES "/home/locobot/Desktop/team1_guinea_pigs/collaborative-robotics-2026/build/tidybot_msgs/rosidl_cmake/srv/SolveIK_Request.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/tidybot_msgs/srv" TYPE FILE FILES "/home/locobot/Desktop/team1_guinea_pigs/collaborative-robotics-2026/build/tidybot_msgs/rosidl_cmake/srv/SolveIK_Response.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/tidybot_msgs/srv" TYPE FILE FILES "/home/locobot/Desktop/team1_guinea_pigs/collaborative-robotics-2026/ros2_ws/src/tidybot_msgs/srv/GetJointState.srv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/tidybot_msgs/srv" TYPE FILE FILES "/home/locobot/Desktop/team1_guinea_pigs/collaborative-robotics-2026/build/tidybot_msgs/rosidl_cmake/srv/GetJointState_Request.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/tidybot_msgs/srv" TYPE FILE FILES "/home/locobot/Desktop/team1_guinea_pigs/collaborative-robotics-2026/build/tidybot_msgs/rosidl_cmake/srv/GetJointState_Response.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/tidybot_msgs/srv" TYPE FILE FILES "/home/locobot/Desktop/team1_guinea_pigs/collaborative-robotics-2026/ros2_ws/src/tidybot_msgs/srv/PlanToTarget.srv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/tidybot_msgs/srv" TYPE FILE FILES "/home/locobot/Desktop/team1_guinea_pigs/collaborative-robotics-2026/build/tidybot_msgs/rosidl_cmake/srv/PlanToTarget_Request.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/tidybot_msgs/srv" TYPE FILE FILES "/home/locobot/Desktop/team1_guinea_pigs/collaborative-robotics-2026/build/tidybot_msgs/rosidl_cmake/srv/PlanToTarget_Response.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/tidybot_msgs/srv" TYPE FILE FILES "/home/locobot/Desktop/team1_guinea_pigs/collaborative-robotics-2026/ros2_ws/src/tidybot_msgs/srv/AudioRecord.srv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/tidybot_msgs/srv" TYPE FILE FILES "/home/locobot/Desktop/team1_guinea_pigs/collaborative-robotics-2026/build/tidybot_msgs/rosidl_cmake/srv/AudioRecord_Request.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/tidybot_msgs/srv" TYPE FILE FILES "/home/locobot/Desktop/team1_guinea_pigs/collaborative-robotics-2026/build/tidybot_msgs/rosidl_cmake/srv/AudioRecord_Response.msg")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/ament_index/resource_index/package_run_dependencies" TYPE FILE FILES "/home/locobot/Desktop/team1_guinea_pigs/collaborative-robotics-2026/build/tidybot_msgs/ament_cmake_index/share/ament_index/resource_index/package_run_dependencies/tidybot_msgs")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/ament_index/resource_index/parent_prefix_path" TYPE FILE FILES "/home/locobot/Desktop/team1_guinea_pigs/collaborative-robotics-2026/build/tidybot_msgs/ament_cmake_index/share/ament_index/resource_index/parent_prefix_path/tidybot_msgs")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/tidybot_msgs/environment" TYPE FILE FILES "/opt/ros/humble/share/ament_cmake_core/cmake/environment_hooks/environment/ament_prefix_path.sh")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/tidybot_msgs/environment" TYPE FILE FILES "/home/locobot/Desktop/team1_guinea_pigs/collaborative-robotics-2026/build/tidybot_msgs/ament_cmake_environment_hooks/ament_prefix_path.dsv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/tidybot_msgs/environment" TYPE FILE FILES "/opt/ros/humble/share/ament_cmake_core/cmake/environment_hooks/environment/path.sh")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/tidybot_msgs/environment" TYPE FILE FILES "/home/locobot/Desktop/team1_guinea_pigs/collaborative-robotics-2026/build/tidybot_msgs/ament_cmake_environment_hooks/path.dsv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/tidybot_msgs" TYPE FILE FILES "/home/locobot/Desktop/team1_guinea_pigs/collaborative-robotics-2026/build/tidybot_msgs/ament_cmake_environment_hooks/local_setup.bash")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/tidybot_msgs" TYPE FILE FILES "/home/locobot/Desktop/team1_guinea_pigs/collaborative-robotics-2026/build/tidybot_msgs/ament_cmake_environment_hooks/local_setup.sh")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/tidybot_msgs" TYPE FILE FILES "/home/locobot/Desktop/team1_guinea_pigs/collaborative-robotics-2026/build/tidybot_msgs/ament_cmake_environment_hooks/local_setup.zsh")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/tidybot_msgs" TYPE FILE FILES "/home/locobot/Desktop/team1_guinea_pigs/collaborative-robotics-2026/build/tidybot_msgs/ament_cmake_environment_hooks/local_setup.dsv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/tidybot_msgs" TYPE FILE FILES "/home/locobot/Desktop/team1_guinea_pigs/collaborative-robotics-2026/build/tidybot_msgs/ament_cmake_environment_hooks/package.dsv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/ament_index/resource_index/packages" TYPE FILE FILES "/home/locobot/Desktop/team1_guinea_pigs/collaborative-robotics-2026/build/tidybot_msgs/ament_cmake_index/share/ament_index/resource_index/packages/tidybot_msgs")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/tidybot_msgs/cmake/export_tidybot_msgs__rosidl_generator_cExport.cmake")
    file(DIFFERENT EXPORT_FILE_CHANGED FILES
         "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/tidybot_msgs/cmake/export_tidybot_msgs__rosidl_generator_cExport.cmake"
         "/home/locobot/Desktop/team1_guinea_pigs/collaborative-robotics-2026/build/tidybot_msgs/CMakeFiles/Export/share/tidybot_msgs/cmake/export_tidybot_msgs__rosidl_generator_cExport.cmake")
    if(EXPORT_FILE_CHANGED)
      file(GLOB OLD_CONFIG_FILES "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/tidybot_msgs/cmake/export_tidybot_msgs__rosidl_generator_cExport-*.cmake")
      if(OLD_CONFIG_FILES)
        message(STATUS "Old export file \"$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/tidybot_msgs/cmake/export_tidybot_msgs__rosidl_generator_cExport.cmake\" will be replaced.  Removing files [${OLD_CONFIG_FILES}].")
        file(REMOVE ${OLD_CONFIG_FILES})
      endif()
    endif()
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/tidybot_msgs/cmake" TYPE FILE FILES "/home/locobot/Desktop/team1_guinea_pigs/collaborative-robotics-2026/build/tidybot_msgs/CMakeFiles/Export/share/tidybot_msgs/cmake/export_tidybot_msgs__rosidl_generator_cExport.cmake")
  if("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^()$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/tidybot_msgs/cmake" TYPE FILE FILES "/home/locobot/Desktop/team1_guinea_pigs/collaborative-robotics-2026/build/tidybot_msgs/CMakeFiles/Export/share/tidybot_msgs/cmake/export_tidybot_msgs__rosidl_generator_cExport-noconfig.cmake")
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/tidybot_msgs/cmake/export_tidybot_msgs__rosidl_typesupport_fastrtps_cExport.cmake")
    file(DIFFERENT EXPORT_FILE_CHANGED FILES
         "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/tidybot_msgs/cmake/export_tidybot_msgs__rosidl_typesupport_fastrtps_cExport.cmake"
         "/home/locobot/Desktop/team1_guinea_pigs/collaborative-robotics-2026/build/tidybot_msgs/CMakeFiles/Export/share/tidybot_msgs/cmake/export_tidybot_msgs__rosidl_typesupport_fastrtps_cExport.cmake")
    if(EXPORT_FILE_CHANGED)
      file(GLOB OLD_CONFIG_FILES "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/tidybot_msgs/cmake/export_tidybot_msgs__rosidl_typesupport_fastrtps_cExport-*.cmake")
      if(OLD_CONFIG_FILES)
        message(STATUS "Old export file \"$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/tidybot_msgs/cmake/export_tidybot_msgs__rosidl_typesupport_fastrtps_cExport.cmake\" will be replaced.  Removing files [${OLD_CONFIG_FILES}].")
        file(REMOVE ${OLD_CONFIG_FILES})
      endif()
    endif()
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/tidybot_msgs/cmake" TYPE FILE FILES "/home/locobot/Desktop/team1_guinea_pigs/collaborative-robotics-2026/build/tidybot_msgs/CMakeFiles/Export/share/tidybot_msgs/cmake/export_tidybot_msgs__rosidl_typesupport_fastrtps_cExport.cmake")
  if("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^()$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/tidybot_msgs/cmake" TYPE FILE FILES "/home/locobot/Desktop/team1_guinea_pigs/collaborative-robotics-2026/build/tidybot_msgs/CMakeFiles/Export/share/tidybot_msgs/cmake/export_tidybot_msgs__rosidl_typesupport_fastrtps_cExport-noconfig.cmake")
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/tidybot_msgs/cmake/tidybot_msgs__rosidl_typesupport_introspection_cExport.cmake")
    file(DIFFERENT EXPORT_FILE_CHANGED FILES
         "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/tidybot_msgs/cmake/tidybot_msgs__rosidl_typesupport_introspection_cExport.cmake"
         "/home/locobot/Desktop/team1_guinea_pigs/collaborative-robotics-2026/build/tidybot_msgs/CMakeFiles/Export/share/tidybot_msgs/cmake/tidybot_msgs__rosidl_typesupport_introspection_cExport.cmake")
    if(EXPORT_FILE_CHANGED)
      file(GLOB OLD_CONFIG_FILES "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/tidybot_msgs/cmake/tidybot_msgs__rosidl_typesupport_introspection_cExport-*.cmake")
      if(OLD_CONFIG_FILES)
        message(STATUS "Old export file \"$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/tidybot_msgs/cmake/tidybot_msgs__rosidl_typesupport_introspection_cExport.cmake\" will be replaced.  Removing files [${OLD_CONFIG_FILES}].")
        file(REMOVE ${OLD_CONFIG_FILES})
      endif()
    endif()
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/tidybot_msgs/cmake" TYPE FILE FILES "/home/locobot/Desktop/team1_guinea_pigs/collaborative-robotics-2026/build/tidybot_msgs/CMakeFiles/Export/share/tidybot_msgs/cmake/tidybot_msgs__rosidl_typesupport_introspection_cExport.cmake")
  if("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^()$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/tidybot_msgs/cmake" TYPE FILE FILES "/home/locobot/Desktop/team1_guinea_pigs/collaborative-robotics-2026/build/tidybot_msgs/CMakeFiles/Export/share/tidybot_msgs/cmake/tidybot_msgs__rosidl_typesupport_introspection_cExport-noconfig.cmake")
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/tidybot_msgs/cmake/tidybot_msgs__rosidl_typesupport_cExport.cmake")
    file(DIFFERENT EXPORT_FILE_CHANGED FILES
         "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/tidybot_msgs/cmake/tidybot_msgs__rosidl_typesupport_cExport.cmake"
         "/home/locobot/Desktop/team1_guinea_pigs/collaborative-robotics-2026/build/tidybot_msgs/CMakeFiles/Export/share/tidybot_msgs/cmake/tidybot_msgs__rosidl_typesupport_cExport.cmake")
    if(EXPORT_FILE_CHANGED)
      file(GLOB OLD_CONFIG_FILES "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/tidybot_msgs/cmake/tidybot_msgs__rosidl_typesupport_cExport-*.cmake")
      if(OLD_CONFIG_FILES)
        message(STATUS "Old export file \"$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/tidybot_msgs/cmake/tidybot_msgs__rosidl_typesupport_cExport.cmake\" will be replaced.  Removing files [${OLD_CONFIG_FILES}].")
        file(REMOVE ${OLD_CONFIG_FILES})
      endif()
    endif()
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/tidybot_msgs/cmake" TYPE FILE FILES "/home/locobot/Desktop/team1_guinea_pigs/collaborative-robotics-2026/build/tidybot_msgs/CMakeFiles/Export/share/tidybot_msgs/cmake/tidybot_msgs__rosidl_typesupport_cExport.cmake")
  if("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^()$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/tidybot_msgs/cmake" TYPE FILE FILES "/home/locobot/Desktop/team1_guinea_pigs/collaborative-robotics-2026/build/tidybot_msgs/CMakeFiles/Export/share/tidybot_msgs/cmake/tidybot_msgs__rosidl_typesupport_cExport-noconfig.cmake")
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/tidybot_msgs/cmake/export_tidybot_msgs__rosidl_generator_cppExport.cmake")
    file(DIFFERENT EXPORT_FILE_CHANGED FILES
         "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/tidybot_msgs/cmake/export_tidybot_msgs__rosidl_generator_cppExport.cmake"
         "/home/locobot/Desktop/team1_guinea_pigs/collaborative-robotics-2026/build/tidybot_msgs/CMakeFiles/Export/share/tidybot_msgs/cmake/export_tidybot_msgs__rosidl_generator_cppExport.cmake")
    if(EXPORT_FILE_CHANGED)
      file(GLOB OLD_CONFIG_FILES "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/tidybot_msgs/cmake/export_tidybot_msgs__rosidl_generator_cppExport-*.cmake")
      if(OLD_CONFIG_FILES)
        message(STATUS "Old export file \"$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/tidybot_msgs/cmake/export_tidybot_msgs__rosidl_generator_cppExport.cmake\" will be replaced.  Removing files [${OLD_CONFIG_FILES}].")
        file(REMOVE ${OLD_CONFIG_FILES})
      endif()
    endif()
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/tidybot_msgs/cmake" TYPE FILE FILES "/home/locobot/Desktop/team1_guinea_pigs/collaborative-robotics-2026/build/tidybot_msgs/CMakeFiles/Export/share/tidybot_msgs/cmake/export_tidybot_msgs__rosidl_generator_cppExport.cmake")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/tidybot_msgs/cmake/export_tidybot_msgs__rosidl_typesupport_fastrtps_cppExport.cmake")
    file(DIFFERENT EXPORT_FILE_CHANGED FILES
         "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/tidybot_msgs/cmake/export_tidybot_msgs__rosidl_typesupport_fastrtps_cppExport.cmake"
         "/home/locobot/Desktop/team1_guinea_pigs/collaborative-robotics-2026/build/tidybot_msgs/CMakeFiles/Export/share/tidybot_msgs/cmake/export_tidybot_msgs__rosidl_typesupport_fastrtps_cppExport.cmake")
    if(EXPORT_FILE_CHANGED)
      file(GLOB OLD_CONFIG_FILES "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/tidybot_msgs/cmake/export_tidybot_msgs__rosidl_typesupport_fastrtps_cppExport-*.cmake")
      if(OLD_CONFIG_FILES)
        message(STATUS "Old export file \"$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/tidybot_msgs/cmake/export_tidybot_msgs__rosidl_typesupport_fastrtps_cppExport.cmake\" will be replaced.  Removing files [${OLD_CONFIG_FILES}].")
        file(REMOVE ${OLD_CONFIG_FILES})
      endif()
    endif()
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/tidybot_msgs/cmake" TYPE FILE FILES "/home/locobot/Desktop/team1_guinea_pigs/collaborative-robotics-2026/build/tidybot_msgs/CMakeFiles/Export/share/tidybot_msgs/cmake/export_tidybot_msgs__rosidl_typesupport_fastrtps_cppExport.cmake")
  if("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^()$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/tidybot_msgs/cmake" TYPE FILE FILES "/home/locobot/Desktop/team1_guinea_pigs/collaborative-robotics-2026/build/tidybot_msgs/CMakeFiles/Export/share/tidybot_msgs/cmake/export_tidybot_msgs__rosidl_typesupport_fastrtps_cppExport-noconfig.cmake")
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/tidybot_msgs/cmake/tidybot_msgs__rosidl_typesupport_introspection_cppExport.cmake")
    file(DIFFERENT EXPORT_FILE_CHANGED FILES
         "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/tidybot_msgs/cmake/tidybot_msgs__rosidl_typesupport_introspection_cppExport.cmake"
         "/home/locobot/Desktop/team1_guinea_pigs/collaborative-robotics-2026/build/tidybot_msgs/CMakeFiles/Export/share/tidybot_msgs/cmake/tidybot_msgs__rosidl_typesupport_introspection_cppExport.cmake")
    if(EXPORT_FILE_CHANGED)
      file(GLOB OLD_CONFIG_FILES "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/tidybot_msgs/cmake/tidybot_msgs__rosidl_typesupport_introspection_cppExport-*.cmake")
      if(OLD_CONFIG_FILES)
        message(STATUS "Old export file \"$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/tidybot_msgs/cmake/tidybot_msgs__rosidl_typesupport_introspection_cppExport.cmake\" will be replaced.  Removing files [${OLD_CONFIG_FILES}].")
        file(REMOVE ${OLD_CONFIG_FILES})
      endif()
    endif()
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/tidybot_msgs/cmake" TYPE FILE FILES "/home/locobot/Desktop/team1_guinea_pigs/collaborative-robotics-2026/build/tidybot_msgs/CMakeFiles/Export/share/tidybot_msgs/cmake/tidybot_msgs__rosidl_typesupport_introspection_cppExport.cmake")
  if("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^()$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/tidybot_msgs/cmake" TYPE FILE FILES "/home/locobot/Desktop/team1_guinea_pigs/collaborative-robotics-2026/build/tidybot_msgs/CMakeFiles/Export/share/tidybot_msgs/cmake/tidybot_msgs__rosidl_typesupport_introspection_cppExport-noconfig.cmake")
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/tidybot_msgs/cmake/tidybot_msgs__rosidl_typesupport_cppExport.cmake")
    file(DIFFERENT EXPORT_FILE_CHANGED FILES
         "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/tidybot_msgs/cmake/tidybot_msgs__rosidl_typesupport_cppExport.cmake"
         "/home/locobot/Desktop/team1_guinea_pigs/collaborative-robotics-2026/build/tidybot_msgs/CMakeFiles/Export/share/tidybot_msgs/cmake/tidybot_msgs__rosidl_typesupport_cppExport.cmake")
    if(EXPORT_FILE_CHANGED)
      file(GLOB OLD_CONFIG_FILES "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/tidybot_msgs/cmake/tidybot_msgs__rosidl_typesupport_cppExport-*.cmake")
      if(OLD_CONFIG_FILES)
        message(STATUS "Old export file \"$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/tidybot_msgs/cmake/tidybot_msgs__rosidl_typesupport_cppExport.cmake\" will be replaced.  Removing files [${OLD_CONFIG_FILES}].")
        file(REMOVE ${OLD_CONFIG_FILES})
      endif()
    endif()
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/tidybot_msgs/cmake" TYPE FILE FILES "/home/locobot/Desktop/team1_guinea_pigs/collaborative-robotics-2026/build/tidybot_msgs/CMakeFiles/Export/share/tidybot_msgs/cmake/tidybot_msgs__rosidl_typesupport_cppExport.cmake")
  if("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^()$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/tidybot_msgs/cmake" TYPE FILE FILES "/home/locobot/Desktop/team1_guinea_pigs/collaborative-robotics-2026/build/tidybot_msgs/CMakeFiles/Export/share/tidybot_msgs/cmake/tidybot_msgs__rosidl_typesupport_cppExport-noconfig.cmake")
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/tidybot_msgs/cmake/export_tidybot_msgs__rosidl_generator_pyExport.cmake")
    file(DIFFERENT EXPORT_FILE_CHANGED FILES
         "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/tidybot_msgs/cmake/export_tidybot_msgs__rosidl_generator_pyExport.cmake"
         "/home/locobot/Desktop/team1_guinea_pigs/collaborative-robotics-2026/build/tidybot_msgs/CMakeFiles/Export/share/tidybot_msgs/cmake/export_tidybot_msgs__rosidl_generator_pyExport.cmake")
    if(EXPORT_FILE_CHANGED)
      file(GLOB OLD_CONFIG_FILES "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/tidybot_msgs/cmake/export_tidybot_msgs__rosidl_generator_pyExport-*.cmake")
      if(OLD_CONFIG_FILES)
        message(STATUS "Old export file \"$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/tidybot_msgs/cmake/export_tidybot_msgs__rosidl_generator_pyExport.cmake\" will be replaced.  Removing files [${OLD_CONFIG_FILES}].")
        file(REMOVE ${OLD_CONFIG_FILES})
      endif()
    endif()
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/tidybot_msgs/cmake" TYPE FILE FILES "/home/locobot/Desktop/team1_guinea_pigs/collaborative-robotics-2026/build/tidybot_msgs/CMakeFiles/Export/share/tidybot_msgs/cmake/export_tidybot_msgs__rosidl_generator_pyExport.cmake")
  if("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^()$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/tidybot_msgs/cmake" TYPE FILE FILES "/home/locobot/Desktop/team1_guinea_pigs/collaborative-robotics-2026/build/tidybot_msgs/CMakeFiles/Export/share/tidybot_msgs/cmake/export_tidybot_msgs__rosidl_generator_pyExport-noconfig.cmake")
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/tidybot_msgs/cmake" TYPE FILE FILES "/home/locobot/Desktop/team1_guinea_pigs/collaborative-robotics-2026/build/tidybot_msgs/rosidl_cmake/rosidl_cmake-extras.cmake")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/tidybot_msgs/cmake" TYPE FILE FILES "/home/locobot/Desktop/team1_guinea_pigs/collaborative-robotics-2026/build/tidybot_msgs/ament_cmake_export_dependencies/ament_cmake_export_dependencies-extras.cmake")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/tidybot_msgs/cmake" TYPE FILE FILES "/home/locobot/Desktop/team1_guinea_pigs/collaborative-robotics-2026/build/tidybot_msgs/ament_cmake_export_include_directories/ament_cmake_export_include_directories-extras.cmake")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/tidybot_msgs/cmake" TYPE FILE FILES "/home/locobot/Desktop/team1_guinea_pigs/collaborative-robotics-2026/build/tidybot_msgs/ament_cmake_export_libraries/ament_cmake_export_libraries-extras.cmake")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/tidybot_msgs/cmake" TYPE FILE FILES "/home/locobot/Desktop/team1_guinea_pigs/collaborative-robotics-2026/build/tidybot_msgs/ament_cmake_export_targets/ament_cmake_export_targets-extras.cmake")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/tidybot_msgs/cmake" TYPE FILE FILES "/home/locobot/Desktop/team1_guinea_pigs/collaborative-robotics-2026/build/tidybot_msgs/rosidl_cmake/rosidl_cmake_export_typesupport_targets-extras.cmake")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/tidybot_msgs/cmake" TYPE FILE FILES "/home/locobot/Desktop/team1_guinea_pigs/collaborative-robotics-2026/build/tidybot_msgs/rosidl_cmake/rosidl_cmake_export_typesupport_libraries-extras.cmake")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/tidybot_msgs/cmake" TYPE FILE FILES
    "/home/locobot/Desktop/team1_guinea_pigs/collaborative-robotics-2026/build/tidybot_msgs/ament_cmake_core/tidybot_msgsConfig.cmake"
    "/home/locobot/Desktop/team1_guinea_pigs/collaborative-robotics-2026/build/tidybot_msgs/ament_cmake_core/tidybot_msgsConfig-version.cmake"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/tidybot_msgs" TYPE FILE FILES "/home/locobot/Desktop/team1_guinea_pigs/collaborative-robotics-2026/ros2_ws/src/tidybot_msgs/package.xml")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("/home/locobot/Desktop/team1_guinea_pigs/collaborative-robotics-2026/build/tidybot_msgs/tidybot_msgs__py/cmake_install.cmake")

endif()

if(CMAKE_INSTALL_COMPONENT)
  set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
file(WRITE "/home/locobot/Desktop/team1_guinea_pigs/collaborative-robotics-2026/build/tidybot_msgs/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
