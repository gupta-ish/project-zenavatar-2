cmake_minimum_required(VERSION 3.1.0)

project(RealsenseExamplesHelloRealSense)

find_package(realsense2 REQUIRED)

add_executable(main main.cpp)
set_property(TARGET main PROPERTY CXX_STANDARD 11)
target_link_libraries(main ${DEPENDENCIES})
target_link_libraries(main realsense2::realsense2)
set_target_properties (main PROPERTIES FOLDER "Examples")

install(TARGETS main RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})