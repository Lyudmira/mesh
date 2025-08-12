cmake_minimum_required(VERSION 3.16)

# 项目信息
project(IndoorPointCloudReconstruction
    VERSION 1.0.0
    DESCRIPTION "Advanced Indoor Point Cloud Reconstruction System"
    LANGUAGES CXX
)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 设置构建类型
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# 编译选项
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -DDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g -DNDEBUG")

# 添加编译器特定选项
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    add_compile_options(-Wall -Wextra -Wpedantic)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    add_compile_options(/W4)
endif()

# 设置输出目录
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# 包含目录
include_directories(${CMAKE_SOURCE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/recon/src)

# 查找依赖包
find_package(PkgConfig REQUIRED)

# 查找线程库
find_package(Threads REQUIRED)

# 查找OpenMP
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "Found OpenMP")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
else()
    message(WARNING "OpenMP not found, parallel processing will be disabled")
endif()

# 查找Python
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
message(STATUS "Found Python: ${Python3_EXECUTABLE}")

# 查找Boost
find_package(Boost REQUIRED COMPONENTS system filesystem thread)
if(Boost_FOUND)
    message(STATUS "Found Boost: ${Boost_VERSION}")
    include_directories(${Boost_INCLUDE_DIRS})
else()
    message(FATAL_ERROR "Boost not found")
endif()

# 查找Eigen3
find_package(Eigen3 REQUIRED)
if(Eigen3_FOUND)
    message(STATUS "Found Eigen3: ${EIGEN3_VERSION}")
    include_directories(${EIGEN3_INCLUDE_DIR})
else()
    message(FATAL_ERROR "Eigen3 not found")
endif()

# 查找CGAL
find_package(CGAL QUIET)
if(CGAL_FOUND)
    message(STATUS "Found CGAL: ${CGAL_VERSION}")
    include(${CGAL_USE_FILE})
else()
    message(WARNING "CGAL not found, some features will be disabled")
    add_definitions(-DNO_CGAL)
endif()

# 查找OpenVDB
find_package(OpenVDB QUIET)
if(OpenVDB_FOUND)
    message(STATUS "Found OpenVDB: ${OpenVDB_VERSION}")
else()
    message(WARNING "OpenVDB not found, some features will be disabled")
    add_definitions(-DNO_OPENVDB)
endif()

# 查找libigl
find_package(libigl QUIET)
if(libigl_FOUND)
    message(STATUS "Found libigl")
else()
    message(WARNING "libigl not found, some features will be disabled")
    add_definitions(-DNO_LIBIGL)
endif()

# 查找PCL
find_package(PCL QUIET)
if(PCL_FOUND)
    message(STATUS "Found PCL: ${PCL_VERSION}")
    include_directories(${PCL_INCLUDE_DIRS})
    link_directories(${PCL_LIBRARY_DIRS})
    add_definitions(${PCL_DEFINITIONS})
else()
    message(WARNING "PCL not found, some features will be disabled")
    add_definitions(-DNO_PCL)
endif()

# 查找nlohmann/json
find_package(nlohmann_json QUIET)
if(nlohmann_json_FOUND)
    message(STATUS "Found nlohmann/json: ${nlohmann_json_VERSION}")
else()
    message(WARNING "nlohmann/json not found, using fallback")
    add_definitions(-DNO_NLOHMANN_JSON)
endif()

# 查找yaml-cpp
find_package(yaml-cpp QUIET)
if(yaml-cpp_FOUND)
    message(STATUS "Found yaml-cpp: ${yaml-cpp_VERSION}")
else()
    message(WARNING "yaml-cpp not found, configuration system will be disabled")
    add_definitions(-DNO_YAML_CPP)
endif()

# 查找OpenCV (可选)
find_package(OpenCV QUIET)
if(OpenCV_FOUND)
    message(STATUS "Found OpenCV: ${OpenCV_VERSION}")
    include_directories(${OpenCV_INCLUDE_DIRS})
else()
    message(STATUS "OpenCV not found, image processing features will be disabled")
    add_definitions(-DNO_OPENCV)
endif()

# 定义源文件
set(RECON_SOURCES
    # 基础类型和工具
    recon/src/base/types.h
    
    # 配置管理
    recon/src/config/config_manager.h
    
    # 工具类
    recon/src/utils/logger.h
    recon/src/utils/data_validator.h
    recon/src/utils/performance_monitor.h
    
    # 图割优化 (推荐使用)
    recon/src/graph_cut/boykov_kolmogorov_solver.h
    recon/src/graph_cut/boykov_kolmogorov_solver.cpp
    recon/src/graph_cut/minimal_graph_cut.h
    recon/src/graph_cut/minimal_graph_cut.cpp
    
    # UDF构建 (推荐使用)
    recon/src/udf_builder/enhanced_udf_builder.h
    recon/src/udf_builder/enhanced_udf_builder.cpp
    
    # 集成模块
    recon/src/integration/udf_graphcut_integrator.h
    recon/src/integration/udf_graphcut_integrator.cpp
    
    # 细节层重建 (推荐使用)
    recon/src/detail_layer/adaptive_detail_selector.h
    recon/src/detail_layer/adaptive_detail_selector.cpp
    recon/src/detail_layer/hybrid_reconstructor.h
    recon/src/detail_layer/hybrid_reconstructor.cpp
    recon/src/detail_layer/detail_reconstruction_integrator.h
    recon/src/detail_layer/detail_reconstruction_integrator.cpp
    
    # 融合与合法化 (推荐使用)
    recon/src/fusion/alpha_wrapping_fusion.h
    recon/src/fusion/alpha_wrapping_fusion.cpp
    recon/src/fusion/legalization_processor.h
    recon/src/fusion/legalization_processor.cpp
    recon/src/fusion/fusion_legalization_integrator.h
    recon/src/fusion/fusion_legalization_integrator.cpp
    
    # 主重建器
    recon/src/main_reconstructor.h
    recon/src/main_reconstructor.cpp
    
    # 统一接口
    recon/src/unified_interface.h
    recon/src/unified_interface.cpp
    
    # 兼容性模块 (不推荐，但保留)
    recon/src/graph_cut/graph_cut.h
    recon/src/graph_cut/graph_cut.cpp
    recon/src/graph_cut/pymaxflow_solver.h
    recon/src/graph_cut/pymaxflow_solver.cpp
    recon/src/graph_cut/thread_safe_pymaxflow_solver.h
    recon/src/graph_cut/thread_safe_pymaxflow_solver.cpp
    recon/src/udf_builder/udf_builder.h
    recon/src/udf_builder/udf_builder.cpp
    recon/src/udf_builder/optimized_udf_builder.h
    recon/src/udf_builder/optimized_udf_builder.cpp
    recon/src/detail_reconstruction.h
    recon/src/detail_reconstruction.cpp
    recon/src/enhanced_detail_reconstruction.h
    recon/src/enhanced_detail_reconstruction.cpp
    recon/src/fusion/mesh_fusion.h
    recon/src/fusion/mesh_fusion.cpp
    recon/src/fusion/enhanced_mesh_fusion.h
    recon/src/fusion/enhanced_mesh_fusion.cpp
    
    # 双重轮廓
    recon/src/dual_contouring/dual_contouring.h
    recon/src/dual_contouring/dual_contouring.cpp
    recon/src/dual_contouring/enhanced_dual_contouring.h
    recon/src/dual_contouring/enhanced_dual_contouring.cpp
    recon/src/pipeline.cpp
)

# 过滤存在的源文件
set(EXISTING_SOURCES)
foreach(source ${RECON_SOURCES})
    if(EXISTS ${CMAKE_SOURCE_DIR}/${source})
        list(APPEND EXISTING_SOURCES ${source})
    else()
        message(STATUS "Source file not found: ${source}")
    endif()
endforeach()

# 创建主库
add_library(recon_lib STATIC ${EXISTING_SOURCES})

# 设置库的包含目录
target_include_directories(recon_lib PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/recon/src>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/external/maxflow>
    $<INSTALL_INTERFACE:include>
)

# 链接依赖库
target_link_libraries(recon_lib PUBLIC
    Threads::Threads
    ${Boost_LIBRARIES}
)

# 条件性链接可选依赖
if(CGAL_FOUND)
    target_link_libraries(recon_lib PUBLIC ${CGAL_LIBRARIES})
endif()

if(OpenVDB_FOUND)
    target_link_libraries(recon_lib PUBLIC OpenVDB::openvdb)
endif()

if(libigl_FOUND)
    target_link_libraries(recon_lib PUBLIC igl::core)
    if(CGAL_FOUND)
        target_link_libraries(recon_lib PUBLIC igl::cgal)
    endif()
endif()

if(PCL_FOUND)
    target_link_libraries(recon_lib PUBLIC ${PCL_LIBRARIES})
endif()

if(nlohmann_json_FOUND)
    target_link_libraries(recon_lib PUBLIC nlohmann_json::nlohmann_json)
endif()

if(yaml-cpp_FOUND)
    target_link_libraries(recon_lib PUBLIC yaml-cpp)
endif()

if(OpenCV_FOUND)
    target_link_libraries(recon_lib PUBLIC ${OpenCV_LIBS})
endif()

if(OpenMP_CXX_FOUND)
    target_link_libraries(recon_lib PUBLIC OpenMP::OpenMP_CXX)
endif()

# 创建主可执行文件
if(EXISTS ${CMAKE_SOURCE_DIR}/recon/src/main.cpp)
    add_executable(mesh_reconstruction recon/src/main.cpp)
    target_link_libraries(mesh_reconstruction recon_lib)
endif()

# 创建测试可执行文件
set(TEST_PROGRAMS
    test_enhanced_logging_system
    test_complete_system_integration
    test_thread_safe_solver
    test_optimized_udf_simple
    test_enhanced_dual_contouring_simple
    test_enhanced_detail_reconstruction_simple
    test_enhanced_mesh_fusion_simple
    integration_test
    minimal_test
)

foreach(test_prog ${TEST_PROGRAMS})
    if(test_prog STREQUAL "integration_test")
        if(EXISTS ${CMAKE_SOURCE_DIR}/recon/tests/${test_prog}.cpp)
            add_executable(${test_prog} recon/tests/${test_prog}.cpp)
            target_link_libraries(${test_prog} recon_lib)
        endif()
    elseif(test_prog STREQUAL "minimal_test")
        if(EXISTS ${CMAKE_SOURCE_DIR}/examples/${test_prog}.cpp)
            add_executable(${test_prog} examples/${test_prog}.cpp)
            target_link_libraries(${test_prog} recon_lib)
        endif()
    else()
        if(EXISTS ${CMAKE_SOURCE_DIR}/${test_prog}.cpp)
            add_executable(${test_prog} ${test_prog}.cpp)
            target_link_libraries(${test_prog} recon_lib)
        endif()
    endif()
endforeach()

# Python模块支持
option(BUILD_PYTHON_MODULE "Build Python module" OFF)
if(BUILD_PYTHON_MODULE)
    find_package(pybind11 QUIET)
    if(pybind11_FOUND)
        pybind11_add_module(pyrecon recon/python/python_bindings.cpp)
        target_link_libraries(pyrecon PRIVATE recon_lib)
    else()
        message(WARNING "pybind11 not found, Python module will not be built")
    endif()
endif()

# 包含必要的CMake模块
include(CMakePackageConfigHelpers)

# 安装目标
install(TARGETS recon_lib
    EXPORT ReconTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
)

# 安装头文件
install(DIRECTORY recon/src/
    DESTINATION include/recon
    FILES_MATCHING PATTERN "*.h"
)

## 导出目标
install(EXPORT ReconTargets
    FILE ReconTargets.cmake
    NAMESPACE recon::
    DESTINATION lib/cmake/recon
)

# 安装配置文件
configure_package_config_file(
    cmake/ReconConfig.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/ReconConfig.cmake"
    INSTALL_DESTINATION lib/cmake/recon
)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/ReconConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/ReconConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/ReconConfigVersion.cmake"
    DESTINATION lib/cmake/recon
)

# 安装pkg-config文件
configure_file(
    cmake/recon.pc.in
    "${CMAKE_CURRENT_BINARY_DIR}/recon.pc"
    @ONLY
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/recon.pc"
    DESTINATION lib/pkgconfig
)

export(EXPORT ReconTargets
    FILE "${CMAKE_CURRENT_BINARY_DIR}/ReconTargets.cmake"
    NAMESPACE Recon::
)

# 创建pkg-config文件
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/recon.pc.in"
    "${CMAKE_CURRENT_BINARY_DIR}/recon.pc"
    @ONLY
)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/recon.pc"
    DESTINATION lib/pkgconfig
)

# 文档生成 (可选)
option(BUILD_DOCUMENTATION "Build documentation" OFF)
if(BUILD_DOCUMENTATION)
    find_package(Doxygen QUIET)
    if(DOXYGEN_FOUND)
        set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/docs/Doxyfile.in)
        set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)
        
        configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)
        
        add_custom_target(doc_doxygen ALL
            COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMENT "Generating API documentation with Doxygen"
            VERBATIM
        )
        
        install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/docs/html/
            DESTINATION share/doc/recon
        )
    else()
        message(WARNING "Doxygen not found, documentation will not be built")
    endif()
endif()

# 测试支持
option(BUILD_TESTS "Build tests" ON)
if(BUILD_TESTS)
    enable_testing()
    
    # 添加测试
    foreach(test_prog ${TEST_PROGRAMS})
        if(TARGET ${test_prog})
            add_test(NAME ${test_prog} COMMAND ${test_prog})
        endif()
    endforeach()
    
    # 性能测试
    if(TARGET test_complete_system_integration)
        add_test(NAME performance_test COMMAND test_complete_system_integration)
        set_tests_properties(performance_test PROPERTIES TIMEOUT 300)
    endif()
endif()

# 打包支持
set(CPACK_PACKAGE_NAME "IndoorPointCloudReconstruction")
set(CPACK_PACKAGE_VERSION ${PROJECT_VERSION})
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY ${PROJECT_DESCRIPTION})
set(CPACK_PACKAGE_VENDOR "Manus AI")
set(CPACK_PACKAGE_CONTACT "support@manus.ai")

set(CPACK_SOURCE_GENERATOR "TGZ")
set(CPACK_GENERATOR "TGZ;DEB;RPM")

set(CPACK_DEBIAN_PACKAGE_DEPENDS "libboost-all-dev, libeigen3-dev, python3")
set(CPACK_RPM_PACKAGE_REQUIRES "boost-devel, eigen3-devel, python3")

include(CPack)

# 显示配置摘要
message(STATUS "")
message(STATUS "=== Configuration Summary ===")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Install prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "")
message(STATUS "=== Dependencies ===")
message(STATUS "Threads: ${Threads_FOUND}")
message(STATUS "OpenMP: ${OpenMP_CXX_FOUND}")
message(STATUS "Boost: ${Boost_FOUND} (${Boost_VERSION})")
message(STATUS "Eigen3: ${Eigen3_FOUND} (${EIGEN3_VERSION})")
message(STATUS "CGAL: ${CGAL_FOUND}")
message(STATUS "OpenVDB: ${OpenVDB_FOUND}")
message(STATUS "libigl: ${libigl_FOUND}")
message(STATUS "PCL: ${PCL_FOUND}")
message(STATUS "nlohmann/json: ${nlohmann_json_FOUND}")
message(STATUS "yaml-cpp: ${yaml-cpp_FOUND}")
message(STATUS "OpenCV: ${OpenCV_FOUND}")
message(STATUS "")
message(STATUS "=== Build Options ===")
message(STATUS "Build Python module: ${BUILD_PYTHON_MODULE}")
message(STATUS "Build documentation: ${BUILD_DOCUMENTATION}")
message(STATUS "Build tests: ${BUILD_TESTS}")
message(STATUS "")

# 警告信息
if(NOT CGAL_FOUND OR NOT OpenVDB_FOUND OR NOT libigl_FOUND)
    message(STATUS "=== WARNINGS ===")
    if(NOT CGAL_FOUND)
        message(STATUS "CGAL not found - geometric algorithms will be limited")
    endif()
    if(NOT OpenVDB_FOUND)
        message(STATUS "OpenVDB not found - volumetric operations will be disabled")
    endif()
    if(NOT libigl_FOUND)
        message(STATUS "libigl not found - mesh processing will be limited")
    endif()
    message(STATUS "Run 'scripts/install_dependencies.sh' to install missing dependencies")
    message(STATUS "")
endif()

