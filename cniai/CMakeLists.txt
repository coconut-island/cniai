cmake_minimum_required(VERSION 3.13)

project(cniai)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)


set(CUDA_INCLUDE_DIR /usr/local/cuda/include)
set(CUDA_LIB_DIR /usr/local/cuda/lib64)

set(TENSORRT_INCLUDE_DIR /usr/local/tensorrt/include)
set(TENSORRT_LIB_DIR /usr/local/tensorrt/lib)

set(NVCODEC_INCLUDE_DIR /usr/local/nvcodec/include)
set(NVCODEC_LIB_DIR /usr/local/nvcodec/lib)

set(FFMPEG_INCLUDE_DIR /usr/local/ffmpeg/include)
set(FFMPEG_LIB_DIR /usr/local/ffmpeg/lib)
set(FFMPEG_LIBS avformat avcodec avutil swscale)


set(RESOURCES_SOURCE_DIR ${CMAKE_SOURCE_DIR}/resources)
set(RESOURCES_BINARY_DIR ${CMAKE_BINARY_DIR}/resources)
if (NOT EXISTS ${RESOURCES_BINARY_DIR})
    file(COPY ${RESOURCES_SOURCE_DIR} DESTINATION ${CMAKE_BINARY_DIR})
else ()
    message(STATUS "Destination directory ${RESOURCES_BINARY_DIR} already exists, skipping copy operation.")
endif ()

set(MODEL_SOURCE_DIR ${CMAKE_SOURCE_DIR}/models)
set(MODEL_BINARY_DIR ${CMAKE_BINARY_DIR}/models)
if (NOT EXISTS ${MODEL_BINARY_DIR})
    file(COPY ${MODEL_SOURCE_DIR} DESTINATION ${CMAKE_BINARY_DIR})
else ()
    message(STATUS "Destination directory ${MODEL_BINARY_DIR} already exists, skipping copy operation.")
endif ()

set(PIPELINE_REPOSITORY_SOURCE_DIR ${CMAKE_SOURCE_DIR}/pipeline_repository)
set(PIPELINE_REPOSITORY_BINARY_DIR ${CMAKE_BINARY_DIR}/pipeline_repository)
if (NOT EXISTS ${PIPELINE_REPOSITORY_BINARY_DIR})
    file(COPY ${PIPELINE_REPOSITORY_SOURCE_DIR} DESTINATION ${CMAKE_BINARY_DIR})
else ()
    message(STATUS "Destination directory ${PIPELINE_REPOSITORY_BINARY_DIR} already exists, skipping copy operation.")
endif ()

add_subdirectory(3rdparty)
add_subdirectory(cniai_cuda_kernel)
add_subdirectory(src cniai)
add_subdirectory(tests)

set(CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/bin)