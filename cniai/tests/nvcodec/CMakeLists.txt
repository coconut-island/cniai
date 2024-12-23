cmake_minimum_required(VERSION 3.13)

project(tests)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

set(CNIAI_REPO_DIR ${CMAKE_SOURCE_DIR})

include_directories(${CNIAI_REPO_DIR}/src)

include_directories(${CUDA_INCLUDE_DIR})
link_directories(${CUDA_LIB_DIR})

include_directories(${NVCODEC_INCLUDE_DIR})
link_directories(${NVCODEC_LIB_DIR})

include_directories(${FFMPEG_INCLUDE_DIR})
link_directories(${FFMPEG_LIB_DIR})

aux_source_directory(${CNIAI_REPO_DIR}/src/nvcodec NVCODEC_SRC)
aux_source_directory(${CNIAI_REPO_DIR}/src/nvcommon NVCOMMON_SRC)
aux_source_directory(${CNIAI_REPO_DIR}/src/common COMMON_SRC)

add_executable(nvcodec nvcodec.cc ${NVCOMMON_SRC} ${NVCODEC_SRC} ${COMMON_SRC})
target_link_libraries(nvcodec PRIVATE dl cniai_cuda_kernel spdlog::spdlog gflags::gflags ${FFMPEG_LIBS} nvcuvid nvidia-encode cudart cuda)
