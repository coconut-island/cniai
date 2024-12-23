cmake_minimum_required(VERSION 3.13)

project(tests)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

set(CNIAI_REPO_DIR ${CMAKE_SOURCE_DIR})

include_directories(${CNIAI_REPO_DIR}/src)

add_executable(thread_pool thread_pool.cc)
target_link_libraries(thread_pool PRIVATE dl spdlog::spdlog pthread)

add_executable(log log.cc)
target_link_libraries(log PRIVATE dl spdlog::spdlog gflags::gflags)

add_executable(json json.cc)
target_link_libraries(json PRIVATE nlohmann_json::nlohmann_json)

add_executable(coroutine coroutine.cc)

