/**
 * 室内点云重建项目 - 配置头文件
 * 由CMake自动生成
 */

#ifndef CONFIG_H
#define CONFIG_H

// 项目信息
#define PROJECT_NAME "PointCloudReconstruction"
#define PROJECT_VERSION "1.0.0"
#define PROJECT_VERSION_MAJOR 1
#define PROJECT_VERSION_MINOR 0
#define PROJECT_VERSION_PATCH 0

// 构建信息
#define BUILD_TYPE "Release"
#define CXX_STANDARD 17

// 依赖库版本
/* #undef OPENVDB_VERSION */
#define PCL_VERSION "1.14.1"
#define CGAL_VERSION "5.6.1"
#define EIGEN3_VERSION "3.4.0"

// 功能开关
/* #undef BUILD_TESTS */
/* #undef BUILD_DOCS */

// 路径信息
#define INSTALL_PREFIX "/usr/local"

#endif // CONFIG_H

