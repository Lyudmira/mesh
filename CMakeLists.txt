# 室内点云重建项目 - CMake配置文件
# 版本: 1.0
# 日期: 2025-08-12

cmake_minimum_required(VERSION 3.16)
project(PointCloudReconstruction VERSION 1.0.0 LANGUAGES CXX)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 设置构建类型
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# 编译选项
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall -Wextra")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# 设置RPATH
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

# 查找依赖包
find_package(PkgConfig REQUIRED)

# OpenVDB (手动查找)
find_path(OPENVDB_INCLUDE_DIR 
    NAMES openvdb/openvdb.h
    PATHS ${CMAKE_PREFIX_PATH}/include $ENV{CONDA_PREFIX}/include
    NO_DEFAULT_PATH
)

find_library(OPENVDB_LIBRARY 
    NAMES openvdb
    PATHS ${CMAKE_PREFIX_PATH}/lib $ENV{CONDA_PREFIX}/lib
    NO_DEFAULT_PATH
)

if(OPENVDB_INCLUDE_DIR AND OPENVDB_LIBRARY)
    message(STATUS "Found OpenVDB:")
    message(STATUS "  Include: ${OPENVDB_INCLUDE_DIR}")
    message(STATUS "  Library: ${OPENVDB_LIBRARY}")
    set(OPENVDB_FOUND TRUE)
else()
    message(FATAL_ERROR "OpenVDB not found")
endif()

# PCL
find_package(PCL REQUIRED COMPONENTS 
    common 
    io 
    filters 
    features 
    surface 
    kdtree
    search
)
if(PCL_FOUND)
    message(STATUS "Found PCL: ${PCL_VERSION}")
    include_directories(${PCL_INCLUDE_DIRS})
    link_directories(${PCL_LIBRARY_DIRS})
    add_definitions(${PCL_DEFINITIONS})
else()
    message(FATAL_ERROR "PCL not found")
endif()

# CGAL
find_package(CGAL REQUIRED)
if(CGAL_FOUND)
    message(STATUS "Found CGAL: ${CGAL_VERSION}")
else()
    message(FATAL_ERROR "CGAL not found")
endif()

# Eigen3
find_package(Eigen3 REQUIRED)
if(Eigen3_FOUND)
    message(STATUS "Found Eigen3: ${EIGEN3_VERSION}")
else()
    message(FATAL_ERROR "Eigen3 not found")
endif()

# TBB (通常由OpenVDB依赖)
find_package(TBB REQUIRED)
if(TBB_FOUND)
    message(STATUS "Found TBB")
else()
    message(WARNING "TBB not found, may cause linking issues")
endif()

# 包含目录
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/udf_builder
    ${CMAKE_CURRENT_SOURCE_DIR}/src/graph_cut
    ${CMAKE_CURRENT_SOURCE_DIR}/src/dual_contouring
    ${CMAKE_CURRENT_SOURCE_DIR}/src/detail_layer
    ${CMAKE_CURRENT_SOURCE_DIR}/src/fusion
    ${OPENVDB_INCLUDE_DIR}
)

# 源文件
set(SOURCES
    src/simple_pipeline.cpp
)

# 创建可执行文件
add_executable(pipeline ${SOURCES})

# 链接库
target_link_libraries(pipeline PRIVATE
    ${OPENVDB_LIBRARY}
    ${PCL_LIBRARIES}
    CGAL::CGAL
    Eigen3::Eigen
)

# 如果TBB可用，链接TBB
if(TBB_FOUND)
    target_link_libraries(pipeline PRIVATE TBB::tbb)
endif()

# 设置目标属性
set_target_properties(pipeline PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
    INSTALL_RPATH_USE_LINK_PATH TRUE
)

# 编译定义
target_compile_definitions(pipeline PRIVATE
    ${PCL_DEFINITIONS}
)

# 安装规则
install(TARGETS pipeline
    RUNTIME DESTINATION bin
)

# 可选：创建测试目标
option(BUILD_TESTS "Build test programs" OFF)
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# 可选：创建文档
option(BUILD_DOCS "Build documentation" OFF)
if(BUILD_DOCS)
    find_package(Doxygen)
    if(DOXYGEN_FOUND)
        add_subdirectory(docs)
    endif()
endif()

# 显示配置信息
message(STATUS "")
message(STATUS "Configuration Summary:")
message(STATUS "  Project: ${PROJECT_NAME} ${PROJECT_VERSION}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Install Prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "")
message(STATUS "Dependencies:")
message(STATUS "  OpenVDB: ${OpenVDB_VERSION}")
message(STATUS "  PCL: ${PCL_VERSION}")
message(STATUS "  CGAL: ${CGAL_VERSION}")
message(STATUS "  Eigen3: ${EIGEN3_VERSION}")
message(STATUS "")

# 创建配置头文件
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/src/config.h.in"
    "${CMAKE_CURRENT_BINARY_DIR}/config.h"
)
include_directories(${CMAKE_CURRENT_BINARY_DIR})

