cmake_minimum_required(VERSION 3.13)

project(tests)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

add_subdirectory(cuda)
add_subdirectory(cuda_kernel)
add_subdirectory(nvcodec)
add_subdirectory(nvjpeg)
add_subdirectory(pipeline)
add_subdirectory(other)
