/**
 * 简化版点云重建管道
 * 专注于UDF模块的自适应细化实现
 */

#include <iostream>
#include <string>
#include <chrono>

// PCL核心库
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/obj_io.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/surface/poisson.h>

// OpenVDB
#include <openvdb/openvdb.h>
#include <openvdb/tools/LevelSetSphere.h>
#include <openvdb/tools/VolumeToMesh.h>

// UDF构建器
#include "udf_builder/udf_builder.h"
#include "udf_builder/enhanced_udf_builder.h"

using PointT = pcl::PointXYZRGB;
using PointCloudT = pcl::PointCloud<PointT>;

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "用法: " << argv[0] << " <输入PLY文件> <输出OBJ文件>" << std::endl;
        return -1;
    }

    std::string input_file = argv[1];
    std::string output_file = argv[2];

    // 初始化OpenVDB
    openvdb::initialize();

    try {
        auto start_time = std::chrono::high_resolution_clock::now();

        // 1. 加载点云
        std::cout << "加载点云: " << input_file << std::endl;
        PointCloudT::Ptr cloud(new PointCloudT);
        if (pcl::io::loadPLYFile<PointT>(input_file, *cloud) == -1) {
            std::cerr << "无法加载文件: " << input_file << std::endl;
            return -1;
        }
        std::cout << "加载了 " << cloud->size() << " 个点" << std::endl;

        // 2. 基础预处理
        std::cout << "执行统计去噪..." << std::endl;
        pcl::StatisticalOutlierRemoval<PointT> sor;
        sor.setInputCloud(cloud);
        sor.setMeanK(50);
        sor.setStddevMulThresh(1.0);
        PointCloudT::Ptr filtered_cloud(new PointCloudT);
        sor.filter(*filtered_cloud);
        std::cout << "去噪后剩余 " << filtered_cloud->size() << " 个点" << std::endl;

        // 3. 法向量估计
        std::cout << "估计法向量..." << std::endl;
        pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_with_normals(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
        pcl::copyPointCloud(*filtered_cloud, *cloud_with_normals);
        
        pcl::NormalEstimationOMP<PointT, pcl::PointXYZRGBNormal> ne;
        ne.setInputCloud(filtered_cloud);
        pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>());
        ne.setSearchMethod(tree);
        ne.setKSearch(20);
        ne.compute(*cloud_with_normals);

        // 4. 构建增强的UDF
        std::cout << "构建自适应UDF..." << std::endl;
        
        // 配置增强UDF构建器
        recon::EnhancedUDFConfig udf_config;
        udf_config.base_voxel_size = 0.05f;          // 5cm基础体素
        udf_config.finest_voxel_size = 0.01f;        // 1cm最细体素
        udf_config.coarsest_voxel_size = 0.1f;       // 10cm最粗体素
        udf_config.max_refinement_levels = 3;        // 最大3级细化
        udf_config.curvature_threshold = 0.1f;       // 曲率阈值
        udf_config.color_gradient_threshold = 25.0f; // 颜色梯度阈值
        udf_config.plane_distance_threshold = 0.03f; // 3cm平面距离阈值
        udf_config.use_parallel_processing = true;   // 启用并行处理
        
        recon::EnhancedUDFBuilder enhanced_builder(udf_config);
        
        recon::EnhancedUDFBuilder::GridT::Ptr udf_grid;
        recon::EnhancedUDFBuilder::ConfidenceGridT::Ptr confidence_grid;
        recon::EnhancedUDFBuilder::RefinementGridT::Ptr refinement_grid;
        
        if (!enhanced_builder.buildEnhancedUDF(cloud_with_normals, udf_grid, confidence_grid, refinement_grid)) {
            std::cerr << "增强UDF构建失败" << std::endl;
            return -1;
        }
        
        // 显示统计信息
        const auto& stats = enhanced_builder.getStatistics();
        std::cout << "UDF构建统计:" << std::endl;
        std::cout << "  总体素数: " << stats.total_voxels << std::endl;
        std::cout << "  细化体素数: " << stats.refined_voxels << std::endl;
        std::cout << "  平面体素数: " << stats.planar_voxels << std::endl;
        std::cout << "  边缘体素数: " << stats.edge_voxels << std::endl;
        std::cout << "  构建时间: " << stats.build_time_seconds << " 秒" << std::endl;

        // 5. 双重轮廓网格提取
        std::cout << "提取网格..." << std::endl;
        std::vector<openvdb::Vec3s> points;
        std::vector<openvdb::Vec3I> triangles;
        std::vector<openvdb::Vec4I> quads;
        
        openvdb::tools::volumeToMesh(*udf_grid, points, triangles, quads, 0.0);
        
        // 6. 转换为PCL格式并保存
        std::cout << "转换并保存网格..." << std::endl;
        pcl::PolygonMesh mesh;
        
        // 转换顶点
        pcl::PointCloud<pcl::PointXYZ>::Ptr mesh_vertices(new pcl::PointCloud<pcl::PointXYZ>);
        for (const auto& point : points) {
            pcl::PointXYZ p;
            p.x = point[0];
            p.y = point[1];
            p.z = point[2];
            mesh_vertices->push_back(p);
        }
        pcl::toPCLPointCloud2(*mesh_vertices, mesh.cloud);
        
        // 转换三角形
        for (const auto& tri : triangles) {
            pcl::Vertices vertices;
            vertices.vertices.resize(3);
            vertices.vertices[0] = tri[0];
            vertices.vertices[1] = tri[1];
            vertices.vertices[2] = tri[2];
            mesh.polygons.push_back(vertices);
        }
        
        // 转换四边形
        for (const auto& quad : quads) {
            pcl::Vertices vertices;
            vertices.vertices.resize(4);
            vertices.vertices[0] = quad[0];
            vertices.vertices[1] = quad[1];
            vertices.vertices[2] = quad[2];
            vertices.vertices[3] = quad[3];
            mesh.polygons.push_back(vertices);
        }

        // 保存结果
        if (pcl::io::saveOBJFile(output_file, mesh) == -1) {
            std::cerr << "无法保存文件: " << output_file << std::endl;
            return -1;
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
        
        std::cout << "重建完成!" << std::endl;
        std::cout << "输出顶点数: " << points.size() << std::endl;
        std::cout << "输出三角形数: " << triangles.size() << std::endl;
        std::cout << "输出四边形数: " << quads.size() << std::endl;
        std::cout << "总耗时: " << duration.count() << " 秒" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}

