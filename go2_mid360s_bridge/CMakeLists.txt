cmake_minimum_required(VERSION 3.5)
project(go2_mid360s_bridge)

find_package(ament_cmake REQUIRED)
find_package(livox_ros_driver2 REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)

add_executable(cloud_adapter src/cloud_adapter.cpp)
target_compile_features(cloud_adapter PRIVATE cxx_std_14)
ament_target_dependencies(cloud_adapter rclcpp sensor_msgs livox_ros_driver2)
add_executable(imu_si src/imu_si.cpp)
target_compile_features(imu_si PRIVATE cxx_std_14)
ament_target_dependencies(imu_si rclcpp sensor_msgs)

install(DIRECTORY launch config DESTINATION share/${PROJECT_NAME})
install(TARGETS imu_si cloud_adapter DESTINATION lib/${PROJECT_NAME})

ament_package()
