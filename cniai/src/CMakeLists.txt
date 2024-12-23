cmake_minimum_required(VERSION 3.13)

project(cniai)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

include_directories(${CUDA_INCLUDE_DIR})
link_directories(${CUDA_LIB_DIR})

include_directories(${TENSORRT_INCLUDE_DIR})
link_directories(${TENSORRT_LIB_DIR})

include_directories(${NVCODEC_INCLUDE_DIR})
link_directories(${NVCODEC_LIB_DIR})

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

set(CUDA_LIBS cudart nvjpeg)
set(TENSORRT_LIBS nvinfer)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/src/common SRCS)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/src/nvcodec SRCS)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/src/nvjpeg SRCS)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/src/nvcommon SRCS)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/src/tensorrt SRCS)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/src/pipeline SRCS)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/src/pipeline/yolov5 SRCS)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/src/pipeline/paddleocr SRCS)

add_executable(cniai cniai.cc ${SRCS})
target_link_libraries(cniai PRIVATE dl pthread spdlog::spdlog gflags::gflags nlohmann_json::nlohmann_json cniai_cuda_kernel nvcuvid nvidia-encode cudart cuda ${CUDA_LIBS} ${TENSORRT_LIBS} ${FFMPEG_LIBS} ${OpenCV_LIBS})