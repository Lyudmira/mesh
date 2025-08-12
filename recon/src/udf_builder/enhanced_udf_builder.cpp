/**
 * 增强版UDF构建器实现
 * 实现基于曲率、颜色梯度、平面距离及局部密度的动态体素细化
 */

#include "enhanced_udf_builder.h"
#include <openvdb/tools/LevelSetFilter.h>
#include <openvdb/tools/ParticlesToLevelSet.h>
#include <openvdb/tree/LeafManager.h>
#include <pcl/common/common.h>
#include <pcl/features/normal_3d.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <chrono>
#include <omp.h>

namespace recon {

EnhancedUDFBuilder::EnhancedUDFBuilder(const EnhancedUDFConfig& config) 
    : config_(config) {
    // 初始化OpenVDB
    openvdb::initialize();
    
    // 设置OpenMP线程数
    if (config_.use_parallel_processing) {
        if (config_.num_threads > 0) {
            omp_set_num_threads(config_.num_threads);
        }
    }
}

EnhancedUDFBuilder::~EnhancedUDFBuilder() = default;

bool EnhancedUDFBuilder::buildEnhancedUDF(PointCloudT::Ptr cloud, 
                                         GridT::Ptr& udf_grid, 
                                         ConfidenceGridT::Ptr& confidence_grid,
                                         RefinementGridT::Ptr& refinement_grid) {
    if (!cloud || cloud->empty()) {
        std::cerr << "输入点云为空" << std::endl;
        return false;
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    std::cout << "开始构建增强UDF，点云大小: " << cloud->size() << std::endl;
    
    // 1. 初始化
    if (!initialize(*cloud)) {
        std::cerr << "初始化失败" << std::endl;
        return false;
    }
    
    // 2. 构建自适应网格
    if (!buildAdaptiveGrid(*cloud, udf_grid, confidence_grid, refinement_grid)) {
        std::cerr << "自适应网格构建失败" << std::endl;
        return false;
    }
    
    // 3. 应用多级滤波
    if (config_.use_gaussian_filter) {
        std::cout << "应用多级高斯滤波..." << std::endl;
        applyMultiLevelFiltering(udf_grid, *refinement_grid);
    }
    
    // 4. 优化网格拓扑
    std::cout << "优化网格拓扑..." << std::endl;
    optimizeGridTopology(udf_grid);
    optimizeTreeTraversal(udf_grid);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    stats_.build_time_seconds = std::chrono::duration<double>(end_time - start_time).count();
    
    std::cout << "增强UDF构建完成" << std::endl;
    std::cout << "统计信息:" << std::endl;
    std::cout << "  总体素数: " << stats_.total_voxels << std::endl;
    std::cout << "  细化体素数: " << stats_.refined_voxels << std::endl;
    std::cout << "  平面体素数: " << stats_.planar_voxels << std::endl;
    std::cout << "  边缘体素数: " << stats_.edge_voxels << std::endl;
    std::cout << "  角点体素数: " << stats_.corner_voxels << std::endl;
    std::cout << "  平均细化级别: " << stats_.avg_refinement_level << std::endl;
    std::cout << "  构建时间: " << stats_.build_time_seconds << " 秒" << std::endl;
    
    return true;
}

bool EnhancedUDFBuilder::initialize(const PointCloudT& cloud) {
    // 初始化KD树
    kdtree_ = std::make_unique<pcl::KdTreeFLANN<PointT>>();
    PointCloudT::Ptr cloud_ptr(new PointCloudT(cloud));
    kdtree_->setInputCloud(cloud_ptr);
    
    // 清空缓存
    plane_cache_.clear();
    curvature_cache_.clear();
    
    // 重置统计信息
    stats_ = Statistics{};
    
    std::cout << "初始化完成，KD树构建成功" << std::endl;
    return true;
}

bool EnhancedUDFBuilder::buildAdaptiveGrid(const PointCloudT& cloud,
                                          GridT::Ptr& udf_grid,
                                          ConfidenceGridT::Ptr& confidence_grid,
                                          RefinementGridT::Ptr& refinement_grid) {
    
    // 创建网格
    udf_grid = GridT::create(config_.truncation_distance);
    udf_grid->setName("enhanced_udf");
    udf_grid->setGridClass(openvdb::GRID_LEVEL_SET);
    
    confidence_grid = ConfidenceGridT::create(0.0f);
    confidence_grid->setName("confidence");
    confidence_grid->setGridClass(openvdb::GRID_FOG_VOLUME);
    
    refinement_grid = RefinementGridT::create(0);
    refinement_grid->setName("refinement_level");
    refinement_grid->setGridClass(openvdb::GRID_UNKNOWN);
    
    // 设置变换
    float base_size = config_.base_voxel_size;
    openvdb::math::Transform::Ptr transform = 
        openvdb::math::Transform::createLinearTransform(base_size);
    
    udf_grid->setTransform(transform);
    confidence_grid->setTransform(transform);
    refinement_grid->setTransform(transform);
    
    // 计算边界框
    PointT min_pt, max_pt;
    pcl::getMinMax3D(cloud, min_pt, max_pt);
    
    openvdb::Vec3f min_bound(min_pt.x - config_.max_search_radius,
                            min_pt.y - config_.max_search_radius,
                            min_pt.z - config_.max_search_radius);
    openvdb::Vec3f max_bound(max_pt.x + config_.max_search_radius,
                            max_pt.y + config_.max_search_radius,
                            max_pt.z + config_.max_search_radius);
    
    std::cout << "边界框: (" << min_bound.x() << "," << min_bound.y() << "," << min_bound.z() 
              << ") 到 (" << max_bound.x() << "," << max_bound.y() << "," << max_bound.z() << ")" << std::endl;
    
    // 收集需要处理的体素位置
    std::vector<openvdb::Vec3f> voxel_positions;
    
    for (float x = min_bound.x(); x <= max_bound.x(); x += base_size) {
        for (float y = min_bound.y(); y <= max_bound.y(); y += base_size) {
            for (float z = min_bound.z(); z <= max_bound.z(); z += base_size) {
                openvdb::Vec3f pos(x, y, z);
                
                // 快速距离检查：只处理截断距离内的体素
                auto [udf_value, _] = computeEnhancedVoxelValues(pos, cloud, 0);
                if (udf_value <= config_.truncation_distance) {
                    voxel_positions.push_back(pos);
                }
            }
        }
    }
    
    std::cout << "需要处理的基础体素数: " << voxel_positions.size() << std::endl;
    
    // 并行处理体素
    if (config_.use_parallel_processing) {
        processVoxelsParallel(voxel_positions, cloud, udf_grid, confidence_grid, refinement_grid);
    } else {
        // 串行处理
        for (const auto& pos : voxel_positions) {
            refineVoxelRecursive(pos, base_size, 0, cloud, udf_grid, confidence_grid, refinement_grid);
        }
    }
    
    stats_.total_voxels = static_cast<int>(udf_grid->activeVoxelCount());
    
    return true;
}

VoxelRefinementDecision EnhancedUDFBuilder::computeRefinementDecision(const openvdb::Vec3f& world_pos,
                                                                     const PointCloudT& cloud,
                                                                     int current_level) {
    VoxelRefinementDecision decision;
    
    // 计算各项指标
    decision.curvature = computePrincipalCurvature(world_pos, cloud);
    decision.color_gradient = computeEnhancedColorGradient(world_pos, cloud);
    decision.plane_distance = computeAccuratePlaneDistance(world_pos, cloud);
    
    auto density_stats = computeDensityStatistics(world_pos, cloud);
    decision.local_density = density_stats.density;
    decision.normal_variation = computeNormalVariation(world_pos, cloud);
    
    // 判断区域类型
    decision.is_planar_region = (decision.curvature < config_.curvature_threshold * 0.5f &&
                                decision.plane_distance < config_.plane_distance_threshold);
    
    decision.is_edge_region = (decision.color_gradient > config_.color_gradient_threshold ||
                              decision.normal_variation > 0.3f);
    
    decision.is_corner_region = (decision.curvature > config_.curvature_threshold * 2.0f);
    
    decision.is_high_detail_region = (decision.color_gradient > config_.color_gradient_threshold * 0.7f ||
                                     decision.curvature > config_.curvature_threshold * 1.5f);
    
    // 计算细化优先级
    float priority = 0.0f;
    priority += (decision.curvature / config_.curvature_threshold) * 0.3f;
    priority += (decision.color_gradient / config_.color_gradient_threshold) * 0.3f;
    priority += (1.0f - decision.plane_distance / config_.plane_distance_threshold) * 0.2f;
    priority += (density_stats.variance / config_.density_variance_threshold) * 0.2f;
    
    decision.refinement_priority = std::min(1.0f, priority);
    
    // 判断是否需要细化
    decision.needs_refinement = (current_level < config_.max_refinement_levels) &&
                               (decision.refinement_priority > 0.5f ||
                                decision.is_edge_region ||
                                decision.is_corner_region ||
                                decision.is_high_detail_region);
    
    // 计算目标体素大小
    if (decision.needs_refinement) {
        float size_factor = 1.0f - decision.refinement_priority * 0.8f;
        decision.target_voxel_size = config_.base_voxel_size * size_factor;
        decision.target_voxel_size = std::max(decision.target_voxel_size, config_.finest_voxel_size);
    } else {
        decision.target_voxel_size = config_.base_voxel_size;
    }
    
    return decision;
}

void EnhancedUDFBuilder::refineVoxelRecursive(const openvdb::Vec3f& center,
                                             float voxel_size,
                                             int level,
                                             const PointCloudT& cloud,
                                             GridT::Ptr& udf_grid,
                                             ConfidenceGridT::Ptr& confidence_grid,
                                             RefinementGridT::Ptr& refinement_grid) {
    
    // 计算当前体素的UDF值和置信度
    auto [udf_value, confidence] = computeEnhancedVoxelValues(center, cloud, level);
    
    // 如果超出截断距离，跳过
    if (udf_value > config_.truncation_distance) {
        return;
    }
    
    // 设置体素值
    openvdb::Coord coord = udf_grid->transform().worldToIndexCellCentered(center);
    auto udf_accessor = udf_grid->getAccessor();
    auto conf_accessor = confidence_grid->getAccessor();
    auto ref_accessor = refinement_grid->getAccessor();
    
    udf_accessor.setValue(coord, udf_value);
    conf_accessor.setValue(coord, confidence);
    ref_accessor.setValue(coord, level);
    
    stats_.total_voxels++;
    
    // 计算细化决策
    VoxelRefinementDecision decision = computeRefinementDecision(center, cloud, level);
    
    // 更新统计信息
    if (decision.is_planar_region) stats_.planar_voxels++;
    if (decision.is_edge_region) stats_.edge_voxels++;
    if (decision.is_corner_region) stats_.corner_voxels++;
    
    // 如果需要细化且未达到最大级别
    if (decision.needs_refinement && level < config_.max_refinement_levels) {
        stats_.refined_voxels++;
        
        // 计算子体素大小
        float child_size = voxel_size * 0.5f;
        child_size = std::max(child_size, config_.finest_voxel_size);
        
        // 递归细化8个子体素
        float offset = child_size * 0.5f;
        std::vector<openvdb::Vec3f> child_centers = {
            openvdb::Vec3f(center.x() - offset, center.y() - offset, center.z() - offset),
            openvdb::Vec3f(center.x() + offset, center.y() - offset, center.z() - offset),
            openvdb::Vec3f(center.x() - offset, center.y() + offset, center.z() - offset),
            openvdb::Vec3f(center.x() + offset, center.y() + offset, center.z() - offset),
            openvdb::Vec3f(center.x() - offset, center.y() - offset, center.z() + offset),
            openvdb::Vec3f(center.x() + offset, center.y() - offset, center.z() + offset),
            openvdb::Vec3f(center.x() - offset, center.y() + offset, center.z() + offset),
            openvdb::Vec3f(center.x() + offset, center.y() + offset, center.z() + offset)
        };
        
        for (const auto& child_center : child_centers) {
            refineVoxelRecursive(child_center, child_size, level + 1, 
                               cloud, udf_grid, confidence_grid, refinement_grid);
        }
    }
}

std::pair<float, float> EnhancedUDFBuilder::computeEnhancedVoxelValues(const openvdb::Vec3f& world_pos,
                                                                      const PointCloudT& cloud,
                                                                      int refinement_level) {
    PointT search_point;
    search_point.x = world_pos.x();
    search_point.y = world_pos.y();
    search_point.z = world_pos.z();
    
    // 自适应搜索半径
    float search_radius = config_.base_voxel_size * (3.0f - refinement_level * 0.5f);
    search_radius = std::max(search_radius, config_.finest_voxel_size * 2.0f);
    
    std::vector<int> indices;
    std::vector<float> distances;
    
    float udf_value = config_.truncation_distance;
    float confidence = 0.0f;
    
    if (kdtree_->radiusSearch(search_point, search_radius, indices, distances) > 0) {
        // 找到最近点的距离
        udf_value = std::sqrt(distances[0]);
        udf_value = std::min(udf_value, config_.truncation_distance);
        
        // 计算增强置信度
        confidence = computeEnhancedConfidence(world_pos, cloud, indices, distances, refinement_level);
    }
    
    return {udf_value, confidence};
}

PlaneInfo EnhancedUDFBuilder::fitPlaneRANSAC(const openvdb::Vec3f& center,
                                            const PointCloudT& cloud,
                                            float radius) {
    PlaneInfo plane_info;
    
    // 检查缓存
    openvdb::Coord cache_key = openvdb::Coord(
        static_cast<int>(center.x() / config_.plane_fitting_radius),
        static_cast<int>(center.y() / config_.plane_fitting_radius),
        static_cast<int>(center.z() / config_.plane_fitting_radius)
    );
    
    auto cache_it = plane_cache_.find(cache_key);
    if (cache_it != plane_cache_.end()) {
        return cache_it->second;
    }
    
    // 查找邻域点
    PointT search_point;
    search_point.x = center.x();
    search_point.y = center.y();
    search_point.z = center.z();
    
    std::vector<int> indices;
    std::vector<float> distances;
    
    if (kdtree_->radiusSearch(search_point, radius, indices, distances) < config_.min_plane_points) {
        plane_cache_[cache_key] = plane_info;
        return plane_info;
    }
    
    // 创建邻域点云
    PointCloudT::Ptr neighborhood(new PointCloudT);
    for (int idx : indices) {
        neighborhood->push_back(cloud.points[idx]);
    }
    
    // 使用RANSAC拟合平面
    pcl::SampleConsensusModelPlane<PointT>::Ptr model(
        new pcl::SampleConsensusModelPlane<PointT>(neighborhood));
    pcl::RandomSampleConsensus<PointT> ransac(model);
    ransac.setDistanceThreshold(config_.plane_inlier_threshold);
    ransac.setMaxIterations(1000);
    
    if (ransac.computeModel()) {
        Eigen::VectorXf coefficients;
        ransac.getModelCoefficients(coefficients);
        
        if (coefficients.size() == 4) {
            plane_info.coefficients = coefficients;
            plane_info.normal = Eigen::Vector3f(coefficients[0], coefficients[1], coefficients[2]);
            plane_info.normal.normalize();
            
            // 计算内点数
            std::vector<int> inliers;
            ransac.getInliers(inliers);
            plane_info.inlier_count = static_cast<int>(inliers.size());
            plane_info.confidence = static_cast<float>(inliers.size()) / indices.size();
            
            // 计算平面中心点
            Eigen::Vector3f centroid(0, 0, 0);
            for (int idx : inliers) {
                const auto& pt = cloud.points[indices[idx]];
                centroid += Eigen::Vector3f(pt.x, pt.y, pt.z);
            }
            plane_info.centroid = centroid / inliers.size();
        }
    }
    
    // 缓存结果
    plane_cache_[cache_key] = plane_info;
    return plane_info;
}

float EnhancedUDFBuilder::computePrincipalCurvature(const openvdb::Vec3f& pos, 
                                                   const PointCloudT& cloud) {
    // 检查缓存
    openvdb::Coord cache_key = openvdb::Coord(
        static_cast<int>(pos.x() / config_.curvature_radius),
        static_cast<int>(pos.y() / config_.curvature_radius),
        static_cast<int>(pos.z() / config_.curvature_radius)
    );
    
    auto cache_it = curvature_cache_.find(cache_key);
    if (cache_it != curvature_cache_.end()) {
        return cache_it->second;
    }
    
    PointT search_point;
    search_point.x = pos.x();
    search_point.y = pos.y();
    search_point.z = pos.z();
    
    std::vector<int> indices;
    std::vector<float> distances;
    
    if (kdtree_->radiusSearch(search_point, config_.curvature_radius, indices, distances) < config_.min_curvature_neighbors) {
        curvature_cache_[cache_key] = 0.0f;
        return 0.0f;
    }
    
    // 使用协方差矩阵估计曲率
    Eigen::Matrix3f covariance = Eigen::Matrix3f::Zero();
    Eigen::Vector3f centroid(0, 0, 0);
    
    // 计算质心
    for (int idx : indices) {
        const auto& pt = cloud.points[idx];
        centroid += Eigen::Vector3f(pt.x, pt.y, pt.z);
    }
    centroid /= indices.size();
    
    // 计算协方差矩阵
    for (int idx : indices) {
        const auto& pt = cloud.points[idx];
        Eigen::Vector3f diff(pt.x - centroid.x(), pt.y - centroid.y(), pt.z - centroid.z());
        covariance += diff * diff.transpose();
    }
    covariance /= indices.size();
    
    // 计算特征值
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver(covariance);
    Eigen::Vector3f eigenvalues = solver.eigenvalues();
    
    // 曲率估计：最小特征值与最大特征值的比值
    float curvature = eigenvalues(0) / (eigenvalues(2) + 1e-8f);
    
    curvature_cache_[cache_key] = curvature;
    return curvature;
}

float EnhancedUDFBuilder::computeEnhancedColorGradient(const openvdb::Vec3f& pos, 
                                                      const PointCloudT& cloud) {
    PointT search_point;
    search_point.x = pos.x();
    search_point.y = pos.y();
    search_point.z = pos.z();
    
    std::vector<int> indices;
    std::vector<float> distances;
    float search_radius = config_.base_voxel_size * 2.0f;
    
    if (kdtree_->radiusSearch(search_point, search_radius, indices, distances) < 3) {
        return 0.0f;
    }
    
    // 计算颜色梯度的幅度
    float max_gradient = 0.0f;
    
    for (size_t i = 0; i < indices.size(); ++i) {
        for (size_t j = i + 1; j < indices.size(); ++j) {
            const auto& pt1 = cloud.points[indices[i]];
            const auto& pt2 = cloud.points[indices[j]];
            
            // 计算空间距离
            float spatial_dist = std::sqrt(
                (pt1.x - pt2.x) * (pt1.x - pt2.x) +
                (pt1.y - pt2.y) * (pt1.y - pt2.y) +
                (pt1.z - pt2.z) * (pt1.z - pt2.z)
            );
            
            if (spatial_dist > 1e-6f) {
                // 计算颜色距离
                float color_dist = std::sqrt(
                    (pt1.r - pt2.r) * (pt1.r - pt2.r) +
                    (pt1.g - pt2.g) * (pt1.g - pt2.g) +
                    (pt1.b - pt2.b) * (pt1.b - pt2.b)
                );
                
                // 颜色梯度 = 颜色距离 / 空间距离
                float gradient = color_dist / spatial_dist;
                max_gradient = std::max(max_gradient, gradient);
            }
        }
    }
    
    return max_gradient;
}

float EnhancedUDFBuilder::computeAccuratePlaneDistance(const openvdb::Vec3f& pos, 
                                                      const PointCloudT& cloud) {
    // 拟合局部平面
    PlaneInfo plane = fitPlaneRANSAC(pos, cloud, config_.plane_fitting_radius);
    
    if (plane.confidence < 0.3f) {
        // 平面拟合质量不好，返回大距离
        return config_.plane_distance_threshold * 2.0f;
    }
    
    // 计算点到平面的距离
    float distance = std::abs(
        plane.coefficients[0] * pos.x() +
        plane.coefficients[1] * pos.y() +
        plane.coefficients[2] * pos.z() +
        plane.coefficients[3]
    );
    
    return distance;
}

EnhancedUDFBuilder::DensityStats EnhancedUDFBuilder::computeDensityStatistics(const openvdb::Vec3f& pos, 
                                                                             const PointCloudT& cloud) {
    DensityStats stats;
    
    PointT search_point;
    search_point.x = pos.x();
    search_point.y = pos.y();
    search_point.z = pos.z();
    
    std::vector<int> indices;
    std::vector<float> distances;
    float search_radius = config_.base_voxel_size * 3.0f;
    
    int count = kdtree_->radiusSearch(search_point, search_radius, indices, distances);
    
    if (count == 0) {
        return stats;
    }
    
    // 计算密度
    float volume = (4.0f / 3.0f) * M_PI * search_radius * search_radius * search_radius;
    stats.density = static_cast<float>(count) / volume;
    
    // 计算距离方差（密度均匀性的指标）
    float mean_distance = 0.0f;
    for (float dist : distances) {
        mean_distance += std::sqrt(dist);
    }
    mean_distance /= count;
    
    float variance = 0.0f;
    for (float dist : distances) {
        float d = std::sqrt(dist) - mean_distance;
        variance += d * d;
    }
    stats.variance = variance / count;
    
    // 均匀性：方差越小越均匀
    stats.uniformity = 1.0f / (1.0f + stats.variance);
    
    return stats;
}

float EnhancedUDFBuilder::computeNormalVariation(const openvdb::Vec3f& pos, 
                                                const PointCloudT& cloud) {
    PointT search_point;
    search_point.x = pos.x();
    search_point.y = pos.y();
    search_point.z = pos.z();
    
    std::vector<int> indices;
    std::vector<float> distances;
    float search_radius = config_.base_voxel_size * 2.0f;
    
    if (kdtree_->radiusSearch(search_point, search_radius, indices, distances) < 3) {
        return 0.0f;
    }
    
    // 计算法向量的平均值
    Eigen::Vector3f mean_normal(0, 0, 0);
    int valid_normals = 0;
    
    for (int idx : indices) {
        const auto& pt = cloud.points[idx];
        if (std::isfinite(pt.normal_x) && std::isfinite(pt.normal_y) && std::isfinite(pt.normal_z)) {
            mean_normal += Eigen::Vector3f(pt.normal_x, pt.normal_y, pt.normal_z);
            valid_normals++;
        }
    }
    
    if (valid_normals == 0) {
        return 0.0f;
    }
    
    mean_normal /= valid_normals;
    mean_normal.normalize();
    
    // 计算法向量变化
    float variation = 0.0f;
    for (int idx : indices) {
        const auto& pt = cloud.points[idx];
        if (std::isfinite(pt.normal_x) && std::isfinite(pt.normal_y) && std::isfinite(pt.normal_z)) {
            Eigen::Vector3f normal(pt.normal_x, pt.normal_y, pt.normal_z);
            normal.normalize();
            
            // 计算角度差异
            float dot_product = mean_normal.dot(normal);
            dot_product = std::max(-1.0f, std::min(1.0f, dot_product));
            float angle = std::acos(std::abs(dot_product));
            variation += angle;
        }
    }
    
    return variation / valid_normals;
}

void EnhancedUDFBuilder::applyMultiLevelFiltering(GridT::Ptr grid, 
                                                 const RefinementGridT& refinement_grid) {
    // 对不同细化级别应用不同强度的滤波
    openvdb::tools::LevelSetFilter<GridT> filter(*grid);
    
    for (int level = 0; level <= config_.max_refinement_levels; ++level) {
        // 细化级别越高，滤波强度越小
        float filter_width = config_.gaussian_width * (1.0f - level * 0.2f);
        filter_width = std::max(filter_width, 0.5f);
        
        // 应用滤波
        for (int iter = 0; iter < config_.filter_iterations; ++iter) {
            filter.gaussian(filter_width);
        }
    }
}

void EnhancedUDFBuilder::optimizeGridTopology(GridT::Ptr grid) {
    // 使用OpenVDB的内置优化
    grid->pruneGrid();
    grid->tree().voxelizeActiveTiles();
}

void EnhancedUDFBuilder::optimizeTreeTraversal(GridT::Ptr grid) {
    // 重新组织树结构以优化遍历性能
    grid->tree().prune();
    
    // 使用LeafManager优化叶节点访问
    openvdb::tree::LeafManager<GridT::TreeType> leafManager(grid->tree());
    leafManager.rebuild();
}

void EnhancedUDFBuilder::processVoxelsParallel(const std::vector<openvdb::Vec3f>& positions,
                                              const PointCloudT& cloud,
                                              GridT::Ptr& udf_grid,
                                              ConfidenceGridT::Ptr& confidence_grid,
                                              RefinementGridT::Ptr& refinement_grid) {
    
    std::cout << "开始并行处理 " << positions.size() << " 个体素..." << std::endl;
    
    #pragma omp parallel for if(config_.use_parallel_processing)
    for (size_t i = 0; i < positions.size(); ++i) {
        const auto& pos = positions[i];
        
        // 每个线程处理一个体素的递归细化
        refineVoxelRecursive(pos, config_.base_voxel_size, 0, 
                           cloud, udf_grid, confidence_grid, refinement_grid);
        
        // 进度报告
        if (i % 1000 == 0) {
            #pragma omp critical
            {
                std::cout << "已处理 " << i << "/" << positions.size() << " 个体素" << std::endl;
            }
        }
    }
    
    // 计算平均细化级别
    auto ref_accessor = refinement_grid->getConstAccessor();
    int total_levels = 0;
    int count = 0;
    
    for (auto iter = refinement_grid->cbeginValueOn(); iter; ++iter) {
        total_levels += iter.getValue();
        count++;
    }
    
    if (count > 0) {
        stats_.avg_refinement_level = static_cast<float>(total_levels) / count;
    }
}

float EnhancedUDFBuilder::computeEnhancedConfidence(const openvdb::Vec3f& world_pos,
                                                   const PointCloudT& cloud,
                                                   const std::vector<int>& indices,
                                                   const std::vector<float>& distances,
                                                   int refinement_level) {
    if (indices.empty()) {
        return 0.0f;
    }
    
    // 1. 距离因子
    float min_distance = std::sqrt(distances[0]);
    float distance_factor = 1.0f - (min_distance / config_.truncation_distance);
    distance_factor = std::max(0.0f, distance_factor);
    
    // 2. 密度因子
    auto density_stats = computeDensityStatistics(world_pos, cloud);
    float density_factor = std::min(1.0f, density_stats.density / 1000.0f);  // 归一化
    
    // 3. 颜色一致性因子
    float color_gradient = computeEnhancedColorGradient(world_pos, cloud);
    float color_factor = 1.0f - (color_gradient / config_.color_gradient_threshold);
    color_factor = std::max(0.0f, std::min(1.0f, color_factor));
    
    // 4. 平面性因子
    float curvature = computePrincipalCurvature(world_pos, cloud);
    float planarity_factor = 1.0f / (1.0f + curvature * 10.0f);
    
    // 5. 法向量一致性因子
    float normal_variation = computeNormalVariation(world_pos, cloud);
    float normal_factor = 1.0f - (normal_variation / M_PI);
    normal_factor = std::max(0.0f, normal_factor);
    
    // 6. 细化级别因子（细化级别越高，置信度权重越大）
    float level_factor = 1.0f + refinement_level * 0.1f;
    
    // 加权组合
    float confidence = (
        config_.density_weight * density_factor +
        config_.color_weight * color_factor +
        config_.planarity_weight * planarity_factor +
        config_.normal_weight * normal_factor
    ) * distance_factor * level_factor;
    
    return std::max(0.0f, std::min(1.0f, confidence));
}

// UDF构建器工厂实现
std::unique_ptr<EnhancedUDFBuilder> UDFBuilderFactory::create(BuilderType type,
                                                             const EnhancedUDFConfig& config) {
    switch (type) {
        case BuilderType::ENHANCED:
        case BuilderType::ADAPTIVE:
            return std::make_unique<EnhancedUDFBuilder>(config);
        case BuilderType::BASIC:
        default:
            return std::make_unique<EnhancedUDFBuilder>(config);
    }
}

} // namespace recon

