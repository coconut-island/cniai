cmake_minimum_required(VERSION 3.13)

project(tests)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

set(CNIAI_REPO_DIR ${CMAKE_SOURCE_DIR})

include_directories(${CNIAI_REPO_DIR}/src)

include_directories(${CUDA_INCLUDE_DIR})
link_directories(${CUDA_LIB_DIR})

include_directories(${TENSORRT_INCLUDE_DIR})
link_directories(${TENSORRT_LIB_DIR})

include_directories(${NVCODEC_INCLUDE_DIR})
link_directories(${NVCODEC_LIB_DIR})

include_directories(${FFMPEG_INCLUDE_DIR})
link_directories(${FFMPEG_LIB_DIR})

aux_source_directory(${CNIAI_REPO_DIR}/src/common COMMON_SRC)
aux_source_directory(${CNIAI_REPO_DIR}/src/nvcodec NVCODEC_SRC)
aux_source_directory(${CNIAI_REPO_DIR}/src/nvcommon NVCOMMON_SRC)
aux_source_directory(${CNIAI_REPO_DIR}/src/tensorrt TENSORRT_SRC)
aux_source_directory(${CNIAI_REPO_DIR}/src/nvjpeg NVJPEG_SRC)
aux_source_directory(${CNIAI_REPO_DIR}/src/pipeline PIPELINE_SRC)
aux_source_directory(${CNIAI_REPO_DIR}/src/pipeline/yolov5 PIPELINE_YOLOV5_SRC)
aux_source_directory(${CNIAI_REPO_DIR}/src/pipeline/paddleocr PIPELINE_PADDLEOCR_SRC)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(yolov5_detect_video yolov5_detect_video.cc ${COMMON_SRC} ${NVCODEC_SRC} ${NVCOMMON_SRC} ${TENSORRT_SRC} ${PIPELINE_SRC} ${PIPELINE_YOLOV5_SRC})
target_link_libraries(yolov5_detect_video PRIVATE dl spdlog::spdlog gflags::gflags nlohmann_json::nlohmann_json nvonnxparser nvinfer cudart cuda cniai_cuda_kernel nvcuvid nvidia-encode ${FFMPEG_LIBS} ${OpenCV_LIBS})

add_executable(paddleocr paddleocr.cc ${COMMON_SRC} ${UTIL_SRC} ${TENSORRT_SRC} ${NVJPEG_SRC} ${PIPELINE_SRC} ${PIPELINE_PADDLEOCR_SRC} ${NVJPEG_SRC})
target_link_libraries(paddleocr PRIVATE dl pthread spdlog::spdlog gflags::gflags nlohmann_json::nlohmann_json nvonnxparser nvinfer cudart cniai_cuda_kernel nvjpeg ${OpenCV_LIBS})
