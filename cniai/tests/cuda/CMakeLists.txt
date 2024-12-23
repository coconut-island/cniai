cmake_minimum_required(VERSION 3.13)

project(tests)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

include_directories(${CUDA_INCLUDE_DIR})
link_directories(${CUDA_LIB_DIR})

add_executable(cuda_stream_limit cuda_stream_limit.cc)
target_link_libraries(cuda_stream_limit PRIVATE cudart)
