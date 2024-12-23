cmake_minimum_required(VERSION 3.13)

include(cmake/utils.cmake)
include(cmake/ide.cmake)

if (NOT DEFINED CMAKE_INSTALL_INCLUDEDIR)
    set(CMAKE_INSTALL_INCLUDEDIR include)
endif ()

if (NOT DEFINED CMAKE_INSTALL_LIBDIR)
    set(CMAKE_INSTALL_LIBDIR lib)
endif ()

cniai_cuda_kernel_extract_version()

project(cniai_cuda_kernel VERSION ${CNIAI_CUDA_KERNEL_VERSION} LANGUAGES CXX)
message(STATUS "Build cniai_cuda_kernel: ${CNIAI_CUDA_KERNEL_VERSION}")

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose Release or Debug" FORCE)
endif ()

if (NOT CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 20)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif ()


find_package(CUDA REQUIRED)
message(STATUS "Build type: " ${CMAKE_BUILD_TYPE})

set(GENCODE_SM53 "--generate-code arch=compute_53,code=sm_53")
set(GENCODE_SM62 "--generate-code arch=compute_62,code=sm_62")
set(GENCODE_SM72 "--generate-code arch=compute_72,code=sm_72")
set(GENCODE_SM87 "--generate-code arch=compute_87,code=sm_87")
set(GENCODE_SM_PTX "--generate-code arch=compute_72,code=compute_72")

set(GENCODE_FLAGS
        ${GENCODE_SM53}
        ${GENCODE_SM62}
        ${GENCODE_SM72}
        ${GENCODE_SM87}
        ${GENCODE_SM_PTX}
        )

set(CUDA_NVCC_FLAGS ${GENCODE_FLAGS} -O3)

# 4070 sm_89 compute_89 Ada arch
# 2080\T4 sm_75 compute_75 Turing arch

#sm_50 sm_52 and sm_53  Maxwell support
#sm_60 sm_61 and sm_62  Pascal support
#sm_70 and sm_72        Volta support
#sm_75                  Turing support
#sm_80 sm_86 and sm_87  NVIDIA Ampere GPU architecture support
#sm_89                  Ada support
#sm_90, sm_90a          Hopper support

include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/osd)

option(BUILD_TRUETYPE_BACKEND "Enable the TrueType text backend" ON)

if (BUILD_TRUETYPE_BACKEND)
    add_compile_definitions(ENABLE_TEXT_BACKEND_STB)
endif ()

aux_source_directory(src/imgproc CNIAI_CUDA_KERNEL_SOURCES)
aux_source_directory(src/osd CNIAI_CUDA_KERNEL_SOURCES)
aux_source_directory(src/osd/textbackend CNIAI_CUDA_KERNEL_SOURCES)

set(CUDA_LIBS cudart)

cuda_add_library(cniai_cuda_kernel SHARED ${CNIAI_CUDA_KERNEL_SOURCES})
target_link_libraries(cniai_cuda_kernel ${CUDA_LIBS})


target_compile_definitions(cniai_cuda_kernel PUBLIC CNI_CUDA_KERNEL_COMPILED_LIB)
target_include_directories(cniai_cuda_kernel PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")

set_target_properties(cniai_cuda_kernel PROPERTIES VERSION ${CNIAI_CUDA_KERNEL_VERSION} SOVERSION ${CNIAI_CUDA_KERNEL_VERSION_MAJOR}.${CNIAI_CUDA_KERNEL_VERSION_MINOR})

add_library(cniai_cuda_kernel_header_only INTERFACE)
add_library(cniai::cniai_cuda_kernel_header_only ALIAS cniai_cuda_kernel_header_only)
target_include_directories(cniai_cuda_kernel_header_only INTERFACE "$<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")


message(STATUS "Generating install")
set(project_config_in "${CMAKE_CURRENT_LIST_DIR}/cmake/CniaiCudaKernelConfig.cmake.in")
set(project_config_out "${CMAKE_CURRENT_BINARY_DIR}/CniaiCudaKernelConfig.cmake")
set(config_targets_file "CniaiCudaKernelConfigTargets.cmake")
set(version_config_file "${CMAKE_CURRENT_BINARY_DIR}/CniaiCudaKernelConfigVersion.cmake")
set(export_dest_dir "${CMAKE_INSTALL_LIBDIR}/cmake/CniaiCudaKernel")
set(pkgconfig_install_dir "${CMAKE_INSTALL_LIBDIR}/pkgconfig")
set(pkg_config "${CMAKE_BINARY_DIR}/CniaiCudaKernel.pc")

# ---------------------------------------------------------------------------------------
# Include files
# ---------------------------------------------------------------------------------------
install(DIRECTORY include/ DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
install(
        TARGETS cniai_cuda_kernel cniai_cuda_kernel_header_only
        EXPORT cniai_cuda_kernel
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

# ---------------------------------------------------------------------------------------
# Install pkg-config file
# ---------------------------------------------------------------------------------------
if (IS_ABSOLUTE "${CMAKE_INSTALL_INCLUDEDIR}")
    set(PKG_CONFIG_INCLUDEDIR "${CMAKE_INSTALL_INCLUDEDIR}")
else ()
    set(PKG_CONFIG_INCLUDEDIR "\${prefix}/${CMAKE_INSTALL_INCLUDEDIR}")
endif ()
if (IS_ABSOLUTE "${CMAKE_INSTALL_LIBDIR}")
    set(PKG_CONFIG_LIBDIR "${CMAKE_INSTALL_LIBDIR}")
else ()
    set(PKG_CONFIG_LIBDIR "\${exec_prefix}/${CMAKE_INSTALL_LIBDIR}")
endif ()
get_target_property(PKG_CONFIG_DEFINES cniai_cuda_kernel INTERFACE_COMPILE_DEFINITIONS)
string(REPLACE ";" " -D" PKG_CONFIG_DEFINES "${PKG_CONFIG_DEFINES}")
string(CONCAT PKG_CONFIG_DEFINES "-D" "${PKG_CONFIG_DEFINES}")
configure_file("cmake/CniaiCudaKernel.pc.in" "${pkg_config}" @ONLY)
install(FILES "${pkg_config}" DESTINATION "${pkgconfig_install_dir}")

# ---------------------------------------------------------------------------------------
# Install CMake config files
# ---------------------------------------------------------------------------------------
export(
        TARGETS cniai_cuda_kernel
        NAMESPACE cniai::
        FILE "${CMAKE_CURRENT_BINARY_DIR}/${config_targets_file}")
install(EXPORT cniai_cuda_kernel DESTINATION ${export_dest_dir} NAMESPACE cniai:: FILE ${config_targets_file})

include(CMakePackageConfigHelpers)
configure_package_config_file("${project_config_in}" "${project_config_out}"
        INSTALL_DESTINATION ${export_dest_dir})

write_basic_package_version_file("${version_config_file}" COMPATIBILITY SameMajorVersion)
install(FILES "${project_config_out}" "${version_config_file}" DESTINATION "${export_dest_dir}")

# ---------------------------------------------------------------------------------------
# Support creation of installable packages
# ---------------------------------------------------------------------------------------
include(cmake/CniaiCudaKernelCPack.cmake)