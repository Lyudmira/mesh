/**
 * UDF构建模块实现文件
 * 
 * 版本: 1.0
 * 日期: 2025-08-12
 */

#include "udf_builder.h"
#include <openvdb/tools/LevelSetFilter.h>
#include <openvdb/tools/ParticlesToLevelSet.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/common.h>
#include <algorithm>
#include <cmath>
#include <iostream>

namespace recon {

UDFBuilder::UDFBuilder(const UDFConfig& config) : config_(config) {
    // 初始化OpenVDB
    openvdb::initialize();
}

UDFBuilder::~UDFBuilder() = default;

bool UDFBuilder::buildUDF(PointCloudT::Ptr cloud, 
                         GridT::Ptr& udf_grid, 
                         ConfidenceGridT::Ptr& confidence_grid) {
    if (!cloud || cloud->empty()) {
        std::cerr << "输入点云为空" << std::endl;
        return false;
    }
    
    std::cout << "开始构建UDF，点云大小: " << cloud->size() << std::endl;
    
    // 1. 初始化网格
    if (!initializeGrids(*cloud, udf_grid, confidence_grid)) {
        std::cerr << "网格初始化失败" << std::endl;
        return false;
    }
    
    // 2. 计算边界框
    PointT min_pt, max_pt;
    pcl::getMinMax3D(*cloud, min_pt, max_pt);
    
    // 扩展边界框
    openvdb::Vec3f min_bound(min_pt.x - config_.expansion_distance,
                            min_pt.y - config_.expansion_distance,
                            min_pt.z - config_.expansion_distance);
    openvdb::Vec3f max_bound(max_pt.x + config_.expansion_distance,
                            max_pt.y + config_.expansion_distance,
                            max_pt.z + config_.expansion_distance);
    
    // 3. 遍历体素并计算UDF值
    auto udf_accessor = udf_grid->getAccessor();
    auto conf_accessor = confidence_grid->getAccessor();
    
    // 使用粗体素大小进行初始遍历
    float voxel_size = config_.coarse_voxel_size;
    openvdb::math::Transform::Ptr transform = 
        openvdb::math::Transform::createLinearTransform(voxel_size);
    
    udf_grid->setTransform(transform);
    confidence_grid->setTransform(transform);
    
    int processed_voxels = 0;
    int refined_voxels = 0;
    
    // 遍历边界框内的体素
    for (float x = min_bound.x(); x <= max_bound.x(); x += voxel_size) {
        for (float y = min_bound.y(); y <= max_bound.y(); y += voxel_size) {
            for (float z = min_bound.z(); z <= max_bound.z(); z += voxel_size) {
                openvdb::Vec3f world_pos(x, y, z);
                openvdb::Coord coord = transform->worldToIndexCellCentered(world_pos);
                
                // 计算UDF值和置信度
                auto [udf_value, confidence] = computeVoxelValues(world_pos, *cloud);
                
                // 检查是否在截断距离内
                if (udf_value <= config_.truncation_distance) {
                    udf_accessor.setValue(coord, udf_value);
                    conf_accessor.setValue(coord, confidence);
                    processed_voxels++;
                    
                    // 检查是否需要细化
                    VoxelRefinementInfo refine_info = computeRefinementInfo(coord, *cloud, *udf_grid);
                    if (needsRefinement(refine_info)) {
                        // 在此体素周围进行细化
                        float fine_size = config_.fine_voxel_size;
                        for (float fx = x - voxel_size/2; fx <= x + voxel_size/2; fx += fine_size) {
                            for (float fy = y - voxel_size/2; fy <= y + voxel_size/2; fy += fine_size) {
                                for (float fz = z - voxel_size/2; fz <= z + voxel_size/2; fz += fine_size) {
                                    openvdb::Vec3f fine_pos(fx, fy, fz);
                                    auto [fine_udf, fine_conf] = computeVoxelValues(fine_pos, *cloud);
                                    
                                    if (fine_udf <= config_.truncation_distance) {
                                        openvdb::Coord fine_coord = transform->worldToIndexCellCentered(fine_pos);
                                        udf_accessor.setValue(fine_coord, fine_udf);
                                        conf_accessor.setValue(fine_coord, fine_conf);
                                        refined_voxels++;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    std::cout << "处理体素数: " << processed_voxels << ", 细化体素数: " << refined_voxels << std::endl;
    
    // 4. 应用高斯滤波
    if (config_.use_gaussian_filter) {
        applyGaussianFilter(udf_grid);
        std::cout << "应用高斯滤波完成" << std::endl;
    }
    
    std::cout << "UDF构建完成" << std::endl;
    return true;
}

bool UDFBuilder::initializeGrids(const PointCloudT& cloud, 
                                GridT::Ptr& udf_grid, 
                                ConfidenceGridT::Ptr& confidence_grid) {
    // 创建UDF网格
    udf_grid = GridT::create(config_.truncation_distance);
    udf_grid->setName("udf");
    udf_grid->setGridClass(openvdb::GRID_LEVEL_SET);
    
    // 创建置信度网格
    confidence_grid = ConfidenceGridT::create(0.0f);
    confidence_grid->setName("confidence");
    confidence_grid->setGridClass(openvdb::GRID_FOG_VOLUME);
    
    return true;
}

VoxelRefinementInfo UDFBuilder::computeRefinementInfo(const openvdb::Coord& coord,
                                                     const PointCloudT& cloud,
                                                     const GridT& grid) {
    VoxelRefinementInfo info;
    
    // 获取世界坐标
    openvdb::Vec3f world_pos = grid.transform().indexToWorld(coord);
    
    // 计算各种指标
    info.curvature = computeLocalCurvature(world_pos, cloud);
    info.color_gradient = computeColorGradient(world_pos, cloud);
    info.plane_distance = computePlaneDistance(world_pos, cloud);
    info.local_density = computeLocalDensity(world_pos, cloud);
    
    // 判断是否需要细化
    info.needs_refinement = needsRefinement(info);
    
    return info;
}

bool UDFBuilder::needsRefinement(const VoxelRefinementInfo& info) {
    return (info.curvature > config_.curvature_threshold ||
            info.color_gradient > config_.color_gradient_threshold ||
            info.plane_distance < config_.plane_distance_threshold ||
            info.local_density > config_.density_threshold_multiplier);
}

std::pair<float, float> UDFBuilder::computeVoxelValues(const openvdb::Vec3f& world_pos,
                                                      const PointCloudT& cloud) {
    // 使用KD树查找最近点
    pcl::KdTreeFLANN<PointT> kdtree;
    PointCloudT::Ptr cloud_ptr(new PointCloudT(cloud));
    kdtree.setInputCloud(cloud_ptr);
    
    PointT search_point;
    search_point.x = world_pos.x();
    search_point.y = world_pos.y();
    search_point.z = world_pos.z();
    
    std::vector<int> indices(1);
    std::vector<float> distances(1);
    
    float udf_value = config_.truncation_distance;
    float confidence = 0.0f;
    
    if (kdtree.nearestKSearch(search_point, 1, indices, distances) > 0) {
        udf_value = std::sqrt(distances[0]);
        udf_value = std::min(udf_value, config_.truncation_distance);
        
        // 计算置信度
        confidence = computeConfidence(world_pos, cloud, udf_value);
    }
    
    return {udf_value, confidence};
}

float UDFBuilder::computePointDistance(const openvdb::Vec3f& voxel_center,
                                      const PointT& point) {
    float dx = voxel_center.x() - point.x;
    float dy = voxel_center.y() - point.y;
    float dz = voxel_center.z() - point.z;
    return std::sqrt(dx*dx + dy*dy + dz*dz);
}

float UDFBuilder::computeConfidence(const openvdb::Vec3f& world_pos,
                                   const PointCloudT& cloud,
                                   float distance) {
    // 基于距离、密度和颜色一致性计算置信度
    float density_factor = computeLocalDensity(world_pos, cloud);
    float color_factor = 1.0f - computeColorGradient(world_pos, cloud);
    float planarity_factor = 1.0f / (1.0f + computeLocalCurvature(world_pos, cloud));
    
    // 距离因子（距离越近置信度越高）
    float distance_factor = 1.0f - (distance / config_.truncation_distance);
    
    // 加权组合
    float confidence = config_.density_weight * density_factor +
                      config_.color_weight * color_factor +
                      config_.planarity_weight * planarity_factor;
    
    confidence *= distance_factor;
    
    return std::max(0.0f, std::min(1.0f, confidence));
}

void UDFBuilder::applyGaussianFilter(GridT::Ptr grid) {
    openvdb::tools::LevelSetFilter<GridT> filter(*grid);
    filter.gaussian(config_.gaussian_width);
}

float UDFBuilder::computeLocalCurvature(const openvdb::Vec3f& pos, const PointCloudT& cloud) {
    // 简化的曲率计算：基于邻域点的法向量变化
    pcl::KdTreeFLANN<PointT> kdtree;
    PointCloudT::Ptr cloud_ptr(new PointCloudT(cloud));
    kdtree.setInputCloud(cloud_ptr);
    
    PointT search_point;
    search_point.x = pos.x();
    search_point.y = pos.y();
    search_point.z = pos.z();
    
    std::vector<int> indices;
    std::vector<float> distances;
    float search_radius = config_.coarse_voxel_size * 3.0f;
    
    if (kdtree.radiusSearch(search_point, search_radius, indices, distances) < 3) {
        return 0.0f;
    }
    
    // 计算法向量的方差作为曲率的近似
    float sum_nx = 0, sum_ny = 0, sum_nz = 0;
    for (int idx : indices) {
        sum_nx += cloud.points[idx].normal_x;
        sum_ny += cloud.points[idx].normal_y;
        sum_nz += cloud.points[idx].normal_z;
    }
    
    float mean_nx = sum_nx / indices.size();
    float mean_ny = sum_ny / indices.size();
    float mean_nz = sum_nz / indices.size();
    
    float variance = 0.0f;
    for (int idx : indices) {
        float dx = cloud.points[idx].normal_x - mean_nx;
        float dy = cloud.points[idx].normal_y - mean_ny;
        float dz = cloud.points[idx].normal_z - mean_nz;
        variance += dx*dx + dy*dy + dz*dz;
    }
    
    return variance / indices.size();
}

float UDFBuilder::computeColorGradient(const openvdb::Vec3f& pos, const PointCloudT& cloud) {
    // 计算颜色梯度
    pcl::KdTreeFLANN<PointT> kdtree;
    PointCloudT::Ptr cloud_ptr(new PointCloudT(cloud));
    kdtree.setInputCloud(cloud_ptr);
    
    PointT search_point;
    search_point.x = pos.x();
    search_point.y = pos.y();
    search_point.z = pos.z();
    
    std::vector<int> indices;
    std::vector<float> distances;
    float search_radius = config_.coarse_voxel_size * 2.0f;
    
    if (kdtree.radiusSearch(search_point, search_radius, indices, distances) < 2) {
        return 0.0f;
    }
    
    // 计算颜色方差
    float sum_r = 0, sum_g = 0, sum_b = 0;
    for (int idx : indices) {
        sum_r += cloud.points[idx].r;
        sum_g += cloud.points[idx].g;
        sum_b += cloud.points[idx].b;
    }
    
    float mean_r = sum_r / indices.size();
    float mean_g = sum_g / indices.size();
    float mean_b = sum_b / indices.size();
    
    float variance = 0.0f;
    for (int idx : indices) {
        float dr = cloud.points[idx].r - mean_r;
        float dg = cloud.points[idx].g - mean_g;
        float db = cloud.points[idx].b - mean_b;
        variance += dr*dr + dg*dg + db*db;
    }
    
    return std::sqrt(variance / indices.size()) / 255.0f;  // 归一化到[0,1]
}

float UDFBuilder::computePlaneDistance(const openvdb::Vec3f& pos, const PointCloudT& cloud) {
    // 简化实现：返回固定值
    // 实际实现应该使用RANSAC拟合局部平面
    return config_.plane_distance_threshold + 0.01f;
}

float UDFBuilder::computeLocalDensity(const openvdb::Vec3f& pos, const PointCloudT& cloud) {
    pcl::KdTreeFLANN<PointT> kdtree;
    PointCloudT::Ptr cloud_ptr(new PointCloudT(cloud));
    kdtree.setInputCloud(cloud_ptr);
    
    PointT search_point;
    search_point.x = pos.x();
    search_point.y = pos.y();
    search_point.z = pos.z();
    
    std::vector<int> indices;
    std::vector<float> distances;
    float search_radius = config_.coarse_voxel_size * 2.0f;
    
    int count = kdtree.radiusSearch(search_point, search_radius, indices, distances);
    
    // 密度 = 点数 / 体积
    float volume = (4.0f / 3.0f) * M_PI * search_radius * search_radius * search_radius;
    return static_cast<float>(count) / volume;
}

} // namespace recon

