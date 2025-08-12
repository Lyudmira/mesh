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
#include <pcl/conversions.h>
#include <pcl/PolygonMesh.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/crop_box.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/bilateral.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/surface/mls.h>
#include <pcl/surface/gp3.h>
#include <pcl/surface/marching_cubes_hoppe.h>

// OpenVDB库
#include <openvdb/openvdb.h>

#include "udf_builder/udf_builder.h"
#include "graph_cut/graph_cut.h"
#include "dual_contouring/dual_contouring.h"
// 暂时注释掉有问题的模块，专注于UDF实现
// #include "detail_reconstruction.h"
// #include "detail_reconstruction.cpp"
// #include "fusion/mesh_fusion.h"
// #include "fusion/mesh_fusion.cpp"

// 标准库
#include <vector>
#include <algorithm>
#include <cmath>
#include <queue>
#include <limits>
#include <Eigen/Dense>

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
        double unit_scale = 1.0;              // 单位转换比例
        bool enable_crop = false;            // 是否启用裁剪
        Eigen::Vector4f crop_min = Eigen::Vector4f(-std::numeric_limits<float>::max(),
                                                  -std::numeric_limits<float>::max(),
                                                  -std::numeric_limits<float>::max(), 1.0f);
        Eigen::Vector4f crop_max = Eigen::Vector4f(std::numeric_limits<float>::max(),
                                                  std::numeric_limits<float>::max(),
                                                  std::numeric_limits<float>::max(), 1.0f);
        int knn_neighbors = 16;              // kNN邻居数
        double stddev_thresh = 2.0;          // 统计异常值阈值
        double radius_multiplier = 2.0;      // 半径异常值倍数
        int min_neighbors = 8;               // 半径内最小邻居数
        bool use_mls = false;                // 是否使用MLS平滑
        double mls_radius = 0.0;             // MLS搜索半径
        bool use_bilateral = false;          // 是否使用双边滤波
        double bilateral_sigma_s = 0.03;     // 双边滤波空间sigma
        double bilateral_sigma_r = 0.05;     // 双边滤波强度sigma
        bool use_voxel_downsample = false;   // 是否进行体素下采样
        float voxel_size = 0.0f;             // 下采样体素大小
        int normal_k = 32;                   // 法向估计的邻居数
    };

    struct Statistics {
        double median_knn_distance = 0.0;    // kNN距离中位数
        double p95_knn_distance = 0.0;       // kNN距离95百分位
        size_t input_point_count = 0;        // 输入点数量
        size_t output_point_count = 0;       // 输出点数量
    };

    explicit DataPreprocessor(const Config& config) : config_(config) {}

    const Statistics& stats() const { return stats_; }

    /**
     * 执行完整的预处理管道
     */
    bool process(PointCloudT::Ptr input, PointNormalCloudT::Ptr output) {
        stats_ = Statistics{};
        stats_.input_point_count = input->size();
        std::cout << "开始数据预处理..." << std::endl;

        // 1. 坐标统一和单位标准化
        if (!normalizeCoordinates(input)) {
            std::cerr << "坐标标准化失败" << std::endl;
            return false;
        }

        // 2. 可选裁剪
        if (config_.enable_crop) {
            if (!cropPointCloud(input)) {
                std::cerr << "裁剪失败" << std::endl;
                return false;
            }
        }

        // 3. 计算密度统计
        if (!computeDensityStats(input, stats_.median_knn_distance, stats_.p95_knn_distance)) {
            std::cerr << "密度统计计算失败" << std::endl;
            return false;
        }
        std::cout << "密度统计：median=" << stats_.median_knn_distance
                  << "  p95=" << stats_.p95_knn_distance << std::endl;

        // 4. 统计异常值去除
        PointCloudT::Ptr filtered(new PointCloudT);
        if (!removeStatisticalOutliers(input, filtered)) {
            std::cerr << "统计异常值去除失败" << std::endl;
            return false;
        }

        // 5. 半径异常值去除
        PointCloudT::Ptr radius_filtered(new PointCloudT);
        if (!removeRadiusOutliers(filtered, radius_filtered, stats_.median_knn_distance)) {
            std::cerr << "半径异常值去除失败" << std::endl;
            return false;
        }

        // 6. 特征保持去噪
        PointCloudT::Ptr denoised(new PointCloudT);
        if (!applyDenoising(radius_filtered, denoised, stats_.median_knn_distance)) {
            std::cerr << "去噪失败" << std::endl;
            return false;
        }

        // 7. 密度平衡采样
        PointCloudT::Ptr sampled(new PointCloudT);
        if (!applySampling(denoised, sampled)) {
            std::cerr << "采样失败" << std::endl;
            return false;
        }

        // 8. 法向量估计
        if (!estimateNormals(sampled, output)) {
            std::cerr << "法向量估计失败" << std::endl;
            return false;
        }

        // 9. 法向量全局定向
        orientNormals(output);

        stats_.output_point_count = output->size();
        std::cout << "预处理完成，输入点数: " << stats_.input_point_count
                  << ", 输出点数: " << stats_.output_point_count << std::endl;
        return true;
    }

private:
    Config config_;
    Statistics stats_;

    bool normalizeCoordinates(PointCloudT::Ptr cloud) {
        for (auto& point : cloud->points) {
            point.x *= config_.unit_scale;
            point.y *= config_.unit_scale;
            point.z *= config_.unit_scale;
        }
        return true;
    }

    bool cropPointCloud(PointCloudT::Ptr cloud) {
        pcl::CropBox<PointT> crop;
        crop.setMin(config_.crop_min);
        crop.setMax(config_.crop_max);
        crop.setInputCloud(cloud);
        PointCloudT cropped;
        crop.filter(cropped);
        *cloud = cropped;
        return true;
    }

    bool computeDensityStats(PointCloudT::Ptr cloud, double& median_d, double& p95_d) {
        pcl::KdTreeFLANN<PointT> tree;
        tree.setInputCloud(cloud);
        std::vector<float> distances;
        distances.reserve(cloud->size());

        std::vector<int> idx(config_.knn_neighbors);
        std::vector<float> sqd(config_.knn_neighbors);
        for (size_t i = 0; i < cloud->size(); ++i) {
            if (tree.nearestKSearch(cloud->points[i], config_.knn_neighbors, idx, sqd) > 0) {
                distances.push_back(std::sqrt(sqd.back()));
            }
        }

        if (distances.empty()) return false;

        auto mid_it = distances.begin() + distances.size() / 2;
        std::nth_element(distances.begin(), mid_it, distances.end());
        median_d = distances[distances.size() / 2];

        auto p95_it = distances.begin() + static_cast<size_t>(0.95 * distances.size());
        std::nth_element(distances.begin(), p95_it, distances.end());
        p95_d = distances[static_cast<size_t>(0.95 * distances.size())];
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

    bool removeRadiusOutliers(PointCloudT::Ptr input, PointCloudT::Ptr output, double median_distance) {
        pcl::RadiusOutlierRemoval<PointT> ror;
        ror.setInputCloud(input);
        ror.setRadiusSearch(config_.radius_multiplier * median_distance);
        ror.setMinNeighborsInRadius(config_.min_neighbors);
        ror.filter(*output);
        return true;
    }

    bool applyDenoising(PointCloudT::Ptr input, PointCloudT::Ptr output, double median_distance) {
        if (config_.use_mls) {
            pcl::MovingLeastSquares<PointT, PointT> mls;
            mls.setInputCloud(input);
            pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>);
            mls.setSearchMethod(tree);
            double radius = config_.mls_radius > 0.0 ? config_.mls_radius : 3.0 * median_distance;
            mls.setSearchRadius(radius);
            mls.setPolynomialFit(true);
            mls.setComputeNormals(false);
            mls.process(*output);
            return true;
        } else if (config_.use_bilateral) {
            pcl::BilateralFilter<PointT> bf;
            bf.setInputCloud(input);
            bf.setSigmaS(config_.bilateral_sigma_s);
            bf.setSigmaR(config_.bilateral_sigma_r);
            bf.filter(*output);
            return true;
        }

        // 无去噪，直接复制
        *output = *input;
        return true;
    }

    bool applySampling(PointCloudT::Ptr input, PointCloudT::Ptr output) {
        if (config_.use_voxel_downsample && config_.voxel_size > 0.0f) {
            pcl::VoxelGrid<PointT> vg;
            vg.setInputCloud(input);
            vg.setLeafSize(config_.voxel_size, config_.voxel_size, config_.voxel_size);
            vg.filter(*output);
            return true;
        }

        *output = *input;
        return true;
    }

    bool estimateNormals(PointCloudT::Ptr input, PointNormalCloudT::Ptr output) {
        pcl::NormalEstimationOMP<PointT, pcl::Normal> ne;
        pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
        pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>());

        ne.setInputCloud(input);
        ne.setSearchMethod(tree);
        ne.setKSearch(config_.normal_k);
        ne.setNumberOfThreads(4);
        ne.compute(*normals);

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

    void orientNormals(PointNormalCloudT::Ptr cloud) {
        if (cloud->empty()) return;

        pcl::KdTreeFLANN<PointNormalT> tree;
        tree.setInputCloud(cloud);
        std::vector<bool> visited(cloud->size(), false);
        std::queue<int> q;

        q.push(0);
        visited[0] = true;

        std::vector<int> idx(config_.normal_k);
        std::vector<float> sqd(config_.normal_k);
        while (!q.empty()) {
            int curr = q.front();
            q.pop();
            auto& curr_pt = cloud->points[curr];

            int n = tree.nearestKSearch(curr_pt, config_.normal_k, idx, sqd);
            for (int i = 0; i < n; ++i) {
                int ni = idx[i];
                if (visited[ni]) continue;

                auto& neigh = cloud->points[ni];
                float dot = curr_pt.normal_x * neigh.normal_x +
                            curr_pt.normal_y * neigh.normal_y +
                            curr_pt.normal_z * neigh.normal_z;
                if (dot < 0.0f) {
                    neigh.normal_x *= -1.0f;
                    neigh.normal_y *= -1.0f;
                    neigh.normal_z *= -1.0f;
                }

                visited[ni] = true;
                q.push(ni);
            }
        }
    }
};

/**
 * 网格后处理：简单合法化和退化三角形移除
 */
class MeshPostProcessor {
public:
    bool process(pcl::PolygonMesh& mesh) {
        pcl::PointCloud<pcl::PointXYZ> cloud;
        pcl::fromPCLPointCloud2(mesh.cloud, cloud);
        std::vector<pcl::Vertices> cleaned;
        cleaned.reserve(mesh.polygons.size());
        for (const auto& poly : mesh.polygons) {
            if (poly.vertices.size() != 3) continue;
            const auto& a = cloud[poly.vertices[0]];
            const auto& b = cloud[poly.vertices[1]];
            const auto& c = cloud[poly.vertices[2]];
            Eigen::Vector3f ab(b.x - a.x, b.y - a.y, b.z - a.z);
            Eigen::Vector3f ac(c.x - a.x, c.y - a.y, c.z - a.z);
            if (ab.cross(ac).norm() < 1e-6f) continue; // degenerate
            cleaned.push_back(poly);
        }
        mesh.polygons.swap(cleaned);
        return true;
    }
};

/**
 * 外壳重建器
 * 组合UDF构建、图割优化和双重轮廓提取
 */
class ShellReconstructor {
public:
    struct Config {
        recon::UDFConfig udf_config;
        recon::GraphCutConfig graph_config;
        recon::DualContouringConfig dc_config;
    };

    explicit ShellReconstructor(const Config& config) : config_(config) {}

    bool reconstruct(PointNormalCloudT::Ptr input, pcl::PolygonMesh& output) {
        std::cout << "开始外壳重建..." << std::endl;
        openvdb::initialize();

        recon::UDFBuilder udf_builder(config_.udf_config);
        openvdb::FloatGrid::Ptr udf_grid;
        openvdb::FloatGrid::Ptr conf_grid;
        if (!udf_builder.buildUDF(input, udf_grid, conf_grid)) {
            std::cerr << "UDF构建失败" << std::endl;
            return false;
        }

        recon::GraphCutOptimizer optimizer(config_.graph_config);
        openvdb::Int32Grid::Ptr label_grid;
        if (!optimizer.optimize(udf_grid, conf_grid, input, label_grid)) {
            std::cerr << "图割优化失败" << std::endl;
            return false;
        }

        recon::DualContouringExtractor extractor(config_.dc_config);
        if (!extractor.extractSurface(udf_grid, label_grid, input, output)) {
            std::cerr << "表面提取失败" << std::endl;
            return false;
        }

        MeshPostProcessor post;
        post.process(output);

        std::cout << "外壳重建完成，三角形数: " << output.polygons.size() << std::endl;
        return true;
    }

private:
    Config config_;
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

        // 4. 细节重建
        recon::DetailReconstructionConfig detail_cfg;
        recon::DetailReconstructor detail(detail_cfg);
        pcl::PolygonMesh detail_mesh;
        if (!detail.reconstructDetails(mesh, processed_cloud, detail_mesh)) {
            std::cerr << "细节重建失败，继续使用外壳网格" << std::endl;
        }

        // 5. 融合与布尔运算 - 暂时注释，待实现
        /*
        MeshFusionConfig fusion_cfg;
        MeshFuser fuser(fusion_cfg);
        pcl::PolygonMesh fused;
        if (!detail_mesh.polygons.empty() && fuser.fuse(mesh, detail_mesh, processed_cloud, fused)) {
            mesh.swap(fused);
        } else {
            std::cerr << "融合失败，使用简单拼接" << std::endl;
            pcl::concatenatePolygonMeshes(mesh, detail_mesh, fused);
            mesh.swap(fused);
        }
        */
        
        // 暂时跳过融合，直接使用外壳网格
        std::cout << "暂时跳过融合步骤，使用外壳网格" << std::endl;

        // 6. 保存结果
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

