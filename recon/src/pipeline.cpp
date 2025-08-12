/**
 * 室内点云重建项目 - 主程序入口
 * 基于双层混合管道的点云到网格转换演示
 * 
 * 版本: 1.0
 * 日期: 2025-08-12
 */

#include <iostream>
#include <string>
#include <chrono>
#include <memory>

// PCL库
#include <pcl/io/ply_io.h>
#include <pcl/io/obj_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/surface/gp3.h>
#include <pcl/surface/marching_cubes_hoppe.h>

// OpenVDB库
#include <openvdb/openvdb.h>
#include <openvdb/tools/ParticlesToLevelSet.h>
#include <openvdb/tools/VolumeToMesh.h>
#include <openvdb/tools/LevelSetFilter.h>

// 标准库
#include <vector>
#include <algorithm>
#include <cmath>

// 类型定义
using PointT = pcl::PointXYZRGB;
using PointCloudT = pcl::PointCloud<PointT>;
using PointNormalT = pcl::PointXYZRGBNormal;
using PointNormalCloudT = pcl::PointCloud<PointNormalT>;

/**
 * 数据预处理类
 * 实现Stage 0的功能：坐标统一、异常值去除、法向量估计等
 */
class DataPreprocessor {
public:
    struct Config {
        double unit_scale = 1.0;
        int knn_neighbors = 16;
        double stddev_thresh = 2.0;
        double radius_multiplier = 2.0;
        int min_neighbors = 8;
        int normal_k = 32;
    };

    DataPreprocessor(const Config& config) : config_(config) {}

    /**
     * 执行完整的预处理管道
     */
    bool process(PointCloudT::Ptr input, PointNormalCloudT::Ptr output) {
        std::cout << "开始数据预处理..." << std::endl;
        
        // 1. 坐标统一和单位标准化
        if (!normalizeCoordinates(input)) {
            std::cerr << "坐标标准化失败" << std::endl;
            return false;
        }
        
        // 2. 统计异常值去除
        PointCloudT::Ptr filtered(new PointCloudT);
        if (!removeStatisticalOutliers(input, filtered)) {
            std::cerr << "统计异常值去除失败" << std::endl;
            return false;
        }
        
        // 3. 半径异常值去除
        PointCloudT::Ptr radius_filtered(new PointCloudT);
        if (!removeRadiusOutliers(filtered, radius_filtered)) {
            std::cerr << "半径异常值去除失败" << std::endl;
            return false;
        }
        
        // 4. 法向量估计
        if (!estimateNormals(radius_filtered, output)) {
            std::cerr << "法向量估计失败" << std::endl;
            return false;
        }
        
        std::cout << "预处理完成，输入点数: " << input->size() 
                  << ", 输出点数: " << output->size() << std::endl;
        return true;
    }

private:
    Config config_;

    bool normalizeCoordinates(PointCloudT::Ptr cloud) {
        for (auto& point : cloud->points) {
            point.x *= config_.unit_scale;
            point.y *= config_.unit_scale;
            point.z *= config_.unit_scale;
        }
        return true;
    }

    bool removeStatisticalOutliers(PointCloudT::Ptr input, PointCloudT::Ptr output) {
        pcl::StatisticalOutlierRemoval<PointT> sor;
        sor.setInputCloud(input);
        sor.setMeanK(config_.knn_neighbors);
        sor.setStddevMulThresh(config_.stddev_thresh);
        sor.filter(*output);
        return true;
    }

    bool removeRadiusOutliers(PointCloudT::Ptr input, PointCloudT::Ptr output) {
        // 计算中位数距离
        pcl::KdTreeFLANN<PointT> tree;
        tree.setInputCloud(input);
        std::vector<float> distances;
        
        for (size_t i = 0; i < input->size() && i < 1000; ++i) {
            std::vector<int> idx(config_.knn_neighbors);
            std::vector<float> sqd(config_.knn_neighbors);
            if (tree.nearestKSearch(input->points[i], config_.knn_neighbors, idx, sqd) > 0) {
                distances.push_back(std::sqrt(sqd.back()));
            }
        }
        
        if (distances.empty()) return false;
        
        std::nth_element(distances.begin(), distances.begin() + distances.size()/2, distances.end());
        double median_distance = distances[distances.size()/2];
        
        pcl::RadiusOutlierRemoval<PointT> ror;
        ror.setInputCloud(input);
        ror.setRadiusSearch(config_.radius_multiplier * median_distance);
        ror.setMinNeighborsInRadius(config_.min_neighbors);
        ror.filter(*output);
        return true;
    }

    bool estimateNormals(PointCloudT::Ptr input, PointNormalCloudT::Ptr output) {
        // 估计法向量
        pcl::NormalEstimationOMP<PointT, pcl::Normal> ne;
        pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
        pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>());
        
        ne.setInputCloud(input);
        ne.setSearchMethod(tree);
        ne.setKSearch(config_.normal_k);
        ne.setNumberOfThreads(4);
        ne.compute(*normals);
        
        // 合并点和法向量
        output->clear();
        output->reserve(input->size());
        
        for (size_t i = 0; i < input->size(); ++i) {
            PointNormalT point;
            point.x = input->points[i].x;
            point.y = input->points[i].y;
            point.z = input->points[i].z;
            point.r = input->points[i].r;
            point.g = input->points[i].g;
            point.b = input->points[i].b;
            
            if (i < normals->size()) {
                point.normal_x = normals->points[i].normal_x;
                point.normal_y = normals->points[i].normal_y;
                point.normal_z = normals->points[i].normal_z;
                point.curvature = normals->points[i].curvature;
            }
            
            output->push_back(point);
        }
        
        return true;
    }
};

/**
 * 简化的外壳重建类
 * 实现Stage 1的基本功能：UDF构建和表面提取
 */
class ShellReconstructor {
public:
    struct Config {
        float voxel_size = 0.01f;  // 1厘米体素
        float truncation_distance = 0.03f;  // 3厘米截断距离
        bool use_gaussian_filter = true;
    };

    ShellReconstructor(const Config& config) : config_(config) {}

    /**
     * 从点云构建外壳网格
     */
    bool reconstruct(PointNormalCloudT::Ptr input, pcl::PolygonMesh& output) {
        std::cout << "开始外壳重建..." << std::endl;
        
        // 初始化OpenVDB
        openvdb::initialize();
        
        // 1. 构建距离场
        openvdb::FloatGrid::Ptr grid = buildDistanceField(input);
        if (!grid) {
            std::cerr << "距离场构建失败" << std::endl;
            return false;
        }
        
        // 2. 提取网格
        if (!extractMesh(grid, output)) {
            std::cerr << "网格提取失败" << std::endl;
            return false;
        }
        
        std::cout << "外壳重建完成，三角形数: " << output.polygons.size() << std::endl;
        return true;
    }

private:
    Config config_;

    openvdb::FloatGrid::Ptr buildDistanceField(PointNormalCloudT::Ptr cloud) {
        using GridT = openvdb::FloatGrid;
        GridT::Ptr grid = GridT::create(config_.truncation_distance);
        grid->setTransform(openvdb::math::Transform::createLinearTransform(config_.voxel_size));
        
        // 创建点列表
        struct PointList {
            using PosType = openvdb::Vec3R;
            using value_type = openvdb::Vec3R;
            
            std::vector<openvdb::Vec3R> pts;
            double radius = 0.01;  // 使用double类型
            
            size_t size() const { return pts.size(); }
            void getPos(size_t i, PosType& xyz) const { xyz = pts[i]; }
            void getPosRad(size_t i, PosType& xyz, double& rad) const { 
                xyz = pts[i]; 
                rad = radius; 
            }
        } plist;
        
        for (const auto& p : cloud->points) {
            plist.pts.emplace_back(p.x, p.y, p.z);
        }
        
        // 光栅化为距离场
        openvdb::tools::ParticlesToLevelSet<GridT> raster(*grid);
        raster.setGrainSize(1);
        raster.setRmin(config_.voxel_size * 0.5f);
        raster.rasterizeSpheres(plist);
        raster.finalize();
        
        // 可选的高斯滤波
        if (config_.use_gaussian_filter) {
            openvdb::tools::LevelSetFilter<GridT> filter(*grid);
            filter.gaussian(1.0);
        }
        
        return grid;
    }

    bool extractMesh(openvdb::FloatGrid::Ptr grid, pcl::PolygonMesh& mesh) {
        // 使用OpenVDB的VolumeToMesh提取网格
        std::vector<openvdb::Vec3s> points;
        std::vector<openvdb::Vec3I> triangles;
        std::vector<openvdb::Vec4I> quads;
        
        openvdb::tools::volumeToMesh(*grid, points, triangles, quads, 0.0);
        
        // 转换为PCL格式
        pcl::PointCloud<pcl::PointXYZ> vertices;
        vertices.reserve(points.size());
        
        for (const auto& p : points) {
            vertices.push_back(pcl::PointXYZ(p.x(), p.y(), p.z()));
        }
        
        // 设置顶点
        pcl::toPCLPointCloud2(vertices, mesh.cloud);
        
        // 设置三角形
        mesh.polygons.clear();
        mesh.polygons.reserve(triangles.size() + quads.size() * 2);
        
        for (const auto& tri : triangles) {
            pcl::Vertices vertices;
            vertices.vertices = {static_cast<int>(tri.x()), 
                               static_cast<int>(tri.y()), 
                               static_cast<int>(tri.z())};
            mesh.polygons.push_back(vertices);
        }
        
        // 将四边形分解为三角形
        for (const auto& quad : quads) {
            pcl::Vertices tri1, tri2;
            tri1.vertices = {static_cast<int>(quad.x()), 
                           static_cast<int>(quad.y()), 
                           static_cast<int>(quad.z())};
            tri2.vertices = {static_cast<int>(quad.x()), 
                           static_cast<int>(quad.z()), 
                           static_cast<int>(quad.w())};
            mesh.polygons.push_back(tri1);
            mesh.polygons.push_back(tri2);
        }
        
        return true;
    }
};

/**
 * 主函数
 */
int main(int argc, char** argv) {
    if (argc != 3) {
        std::cout << "用法: " << argv[0] << " <输入.ply> <输出.obj>" << std::endl;
        return -1;
    }
    
    std::string input_file = argv[1];
    std::string output_file = argv[2];
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    try {
        // 1. 加载点云
        std::cout << "加载点云: " << input_file << std::endl;
        PointCloudT::Ptr cloud(new PointCloudT);
        if (pcl::io::loadPLYFile<PointT>(input_file, *cloud) == -1) {
            std::cerr << "无法加载文件: " << input_file << std::endl;
            return -1;
        }
        std::cout << "加载完成，点数: " << cloud->size() << std::endl;
        
        // 2. 数据预处理
        DataPreprocessor::Config preprocess_config;
        DataPreprocessor preprocessor(preprocess_config);
        
        PointNormalCloudT::Ptr processed_cloud(new PointNormalCloudT);
        if (!preprocessor.process(cloud, processed_cloud)) {
            std::cerr << "预处理失败" << std::endl;
            return -1;
        }
        
        // 3. 外壳重建
        ShellReconstructor::Config shell_config;
        ShellReconstructor reconstructor(shell_config);
        
        pcl::PolygonMesh mesh;
        if (!reconstructor.reconstruct(processed_cloud, mesh)) {
            std::cerr << "外壳重建失败" << std::endl;
            return -1;
        }
        
        // 4. 保存结果
        std::cout << "保存网格: " << output_file << std::endl;
        if (pcl::io::saveOBJFile(output_file, mesh) == -1) {
            std::cerr << "无法保存文件: " << output_file << std::endl;
            return -1;
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
        
        std::cout << "重建完成！" << std::endl;
        std::cout << "处理时间: " << duration.count() << " 秒" << std::endl;
        std::cout << "输出顶点数: " << mesh.cloud.width << std::endl;
        std::cout << "输出三角形数: " << mesh.polygons.size() << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}

