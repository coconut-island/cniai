cmake_minimum_required(VERSION 3.13)

project(cniai_3rdparty)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

set(BUILD_SHARED_LIBS ON)

add_subdirectory(gflags)
add_subdirectory(json)
add_subdirectory(spdlog)
