cmake_minimum_required(VERSION 2.8)

include(PICMake)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=x86-64 -mtune=generic")

pi_add_target(depthmap BIN src REQUIRED OpenCV GSLAM System OpenMP)
pi_report_target()

