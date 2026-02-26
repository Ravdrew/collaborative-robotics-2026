cmake_minimum_required(VERSION 3.8)
project(tidybot_navigation)

find_package(ament_cmake REQUIRED)

# Install launch files
install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

# Install config files
install(DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config
)

# Install rviz config files
install(DIRECTORY rviz/
  DESTINATION share/${PROJECT_NAME}/rviz
)

# Install Python scripts
install(PROGRAMS
  scripts/startup_clear_scan.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
