cmake_minimum_required(VERSION 3.13)


project(tests)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

set(CNIAI_REPO_DIR ${CMAKE_SOURCE_DIR})

include_directories(${CNIAI_REPO_DIR}/src)

include_directories(${CUDA_INCLUDE_DIR})
link_directories(${CUDA_LIB_DIR})

aux_source_directory(${CNIAI_REPO_DIR}/src/nvjpeg NVJPEG_SRC)
aux_source_directory(${CNIAI_REPO_DIR}/src/common COMMON_SRC)

add_executable(jpeg_decoder jpeg_decoder.cc ${NVJPEG_SRC} ${COMMON_SRC})
target_link_libraries(jpeg_decoder PRIVATE dl cudart nvjpeg spdlog::spdlog gflags::gflags ${CUDA_LIBS})
