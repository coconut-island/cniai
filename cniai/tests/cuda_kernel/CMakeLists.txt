cmake_minimum_required(VERSION 3.13)

project(tests)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

include_directories(${CUDA_INCLUDE_DIR})
link_directories(${CUDA_LIB_DIR})

set(CNIAI_REPO_DIR ${CMAKE_SOURCE_DIR})

include_directories(${CNIAI_REPO_DIR}/src)

aux_source_directory(${CNIAI_REPO_DIR}/src/common COMMON_SRC)

add_executable(rgb_packed_planar_swap rgb_packed_planar_swap.cc ${COMMON_SRC})
target_link_libraries(rgb_packed_planar_swap PRIVATE dl cudart spdlog::spdlog cniai_cuda_kernel)
