/**
 * UDF与图割集成器实现
 */

#include "udf_graphcut_integrator.h"
#include <pcl/common/common.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/principal_curvatures.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <chrono>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace recon {

// ============================================================================
// UDFGraphCutIntegrator实现
// ============================================================================

UDFGraphCutIntegrator::UDFGraphCutIntegrator(const IntegrationConfig& config)
    : config_(config) {
}

UDFGraphCutIntegrator::~UDFGraphCutIntegrator() = default;

bool UDFGraphCutIntegrator::integrateUDFWithGraphCut(
    const GridT& udf_grid,
    const ConfidenceGridT& confidence_grid,
    const RefinementGridT& refinement_grid,
    const PointCloudT& cloud,
    EnergyParameterMapping& mapping) {
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    std::cout << "开始UDF与图割集成..." << std::endl;
    
    // 1. 初始化
    if (!initialize(cloud)) {
        std::cerr << "集成器初始化失败" << std::endl;
        return false;
    }
    
    // 2. 收集所有活跃体素坐标
    std::vector<openvdb::Coord> voxel_coords;
    for (auto iter = udf_grid.cbeginValueOn(); iter; ++iter) {
        voxel_coords.push_back(iter.getCoord());
    }
    
    std::cout << "收集到 " << voxel_coords.size() << " 个活跃体素" << std::endl;
    
    // 3. 建立能量参数映射
    if (!buildEnergyParameterMapping(voxel_coords, udf_grid, confidence_grid, 
                                   refinement_grid, cloud, mapping)) {
        std::cerr << "能量参数映射构建失败" << std::endl;
        return false;
    }
    
    // 4. 验证映射结果
    if (!validateMapping(mapping)) {
        std::cerr << "映射结果验证失败" << std::endl;
        return false;
    }
    
    // 5. 更新统计信息
    updateStatistics(mapping);
    
    // 6. 优化缓存
    optimizeCacheUsage();
    
    auto end_time = std::chrono::high_resolution_clock::now();
    double integration_time = std::chrono::duration<double>(end_time - start_time).count();
    
    std::cout << "UDF与图割集成完成" << std::endl;
    std::cout << "统计信息:" << std::endl;
    std::cout << "  总体素数: " << mapping.total_voxels << std::endl;
    std::cout << "  平面体素数: " << mapping.planar_voxels << std::endl;
    std::cout << "  边缘体素数: " << mapping.edge_voxels << std::endl;
    std::cout << "  角点体素数: " << mapping.corner_voxels << std::endl;
    std::cout << "  细节体素数: " << mapping.detail_voxels << std::endl;
    std::cout << "  平均α权重: " << mapping.avg_alpha << std::endl;
    std::cout << "  平均λ权重: " << mapping.avg_lambda << std::endl;
    std::cout << "  平均置信度: " << mapping.avg_confidence << std::endl;
    std::cout << "  集成时间: " << integration_time << " 秒" << std::endl;
    
    last_mapping_ = mapping;
    return true;
}

bool UDFGraphCutIntegrator::initialize(const PointCloudT& cloud) {
    // 初始化KD树
    kdtree_ = std::make_unique<pcl::KdTreeFLANN<PointT>>();
    PointCloudT::Ptr cloud_ptr(new PointCloudT(cloud));
    kdtree_->setInputCloud(cloud_ptr);
    
    // 清空缓存
    attribute_cache_.clear();
    
    std::cout << "集成器初始化完成，KD树构建成功" << std::endl;
    return true;
}

bool UDFGraphCutIntegrator::buildEnergyParameterMapping(
    const std::vector<openvdb::Coord>& voxel_coords,
    const GridT& udf_grid,
    const ConfidenceGridT& confidence_grid,
    const RefinementGridT& refinement_grid,
    const PointCloudT& cloud,
    EnergyParameterMapping& mapping) {
    
    mapping.voxel_attributes.clear();
    mapping.voxel_attributes.reserve(voxel_coords.size());
    
    std::cout << "开始构建能量参数映射..." << std::endl;
    
    // 并行处理体素属性计算
    #pragma omp parallel for if(voxel_coords.size() > 1000)
    for (size_t i = 0; i < voxel_coords.size(); ++i) {
        const auto& coord = voxel_coords[i];
        
        // 计算体素属性
        VoxelAttributes attributes = computeVoxelAttributes(
            coord, udf_grid, confidence_grid, refinement_grid, cloud);
        
        // 应用自适应权重调整
        if (config_.enable_adaptive_weighting) {
            applyAdaptiveWeighting(attributes);
        }
        
        // 应用置信度缩放
        if (config_.apply_confidence_scaling) {
            applyConfidenceScaling(attributes);
        }
        
        // 线程安全地添加到映射中
        #pragma omp critical
        {
            mapping.voxel_attributes[coord] = attributes;
        }
        
        // 进度报告
        if (i % 10000 == 0) {
            #pragma omp critical
            {
                std::cout << "已处理 " << i << "/" << voxel_coords.size() << " 个体素" << std::endl;
            }
        }
    }
    
    std::cout << "能量参数映射构建完成" << std::endl;
    return true;
}

VoxelAttributes UDFGraphCutIntegrator::computeVoxelAttributes(
    const openvdb::Coord& coord,
    const GridT& udf_grid,
    const ConfidenceGridT& confidence_grid,
    const RefinementGridT& refinement_grid,
    const PointCloudT& cloud) {
    
    VoxelAttributes attributes;
    
    // 获取世界坐标
    openvdb::Vec3f world_pos = udf_grid.transform().indexToWorld(coord);
    
    // 获取基础信息
    auto udf_accessor = udf_grid.getConstAccessor();
    auto conf_accessor = confidence_grid.getConstAccessor();
    auto ref_accessor = refinement_grid.getConstAccessor();
    
    attributes.confidence = conf_accessor.getValue(coord);
    attributes.refinement_level = ref_accessor.getValue(coord);
    
    // 计算几何属性
    attributes.curvature = computeCurvatureAttribute(world_pos, cloud);
    attributes.color_gradient = computeColorGradientAttribute(world_pos, cloud);
    attributes.plane_distance = computePlaneDistanceAttribute(world_pos, cloud);
    attributes.local_density = computeLocalDensityAttribute(world_pos, cloud);
    attributes.normal_variation = computeNormalVariationAttribute(world_pos, cloud);
    
    // 区域分类
    classifyVoxelRegion(attributes);
    
    // 计算图割参数
    attributes.alpha_weight = computeAlphaMapping(attributes);
    attributes.lambda_weight = computeLambdaMapping(attributes);
    attributes.visibility_penalty = computeVisibilityPenaltyMapping(world_pos, attributes, cloud);
    
    return attributes;
}

void UDFGraphCutIntegrator::classifyVoxelRegion(VoxelAttributes& attributes) const {
    // 基于几何属性进行区域分类
    
    // 平面区域判断
    if (attributes.plane_distance < 0.02f && attributes.normal_variation < 0.1f) {
        attributes.is_planar = true;
    }
    
    // 边缘区域判断
    if (attributes.curvature > config_.edge_threshold && 
        attributes.color_gradient > 20.0f) {
        attributes.is_edge = true;
    }
    
    // 角点区域判断
    if (attributes.curvature > config_.corner_threshold && 
        attributes.normal_variation > 0.6f) {
        attributes.is_corner = true;
    }
    
    // 细节区域判断
    if (attributes.refinement_level > 2 || 
        (attributes.curvature > config_.detail_threshold && 
         attributes.local_density > 1.0f)) {
        attributes.is_detail = true;
    }
}

float UDFGraphCutIntegrator::computeAlphaMapping(const VoxelAttributes& attributes) const {
    float alpha = config_.alpha_base;
    
    // 基于曲率调整
    alpha *= (1.0f + attributes.curvature * config_.alpha_curvature_factor);
    
    // 基于区域类型调整
    if (attributes.is_planar) {
        alpha *= config_.alpha_planar_boost;
    } else if (attributes.is_edge) {
        alpha *= config_.alpha_edge_reduction;
    }
    
    // 基于细化级别调整
    if (attributes.refinement_level < static_cast<int>(config_.refinement_weights.size())) {
        alpha *= config_.refinement_weights[attributes.refinement_level];
    }
    
    return std::max(0.1f, alpha);
}

float UDFGraphCutIntegrator::computeLambdaMapping(const VoxelAttributes& attributes) const {
    float lambda = config_.lambda_base;
    
    // 基于区域类型调整
    if (attributes.is_planar) {
        lambda *= config_.lambda_planar_boost;
    } else if (attributes.is_edge) {
        lambda *= config_.lambda_edge_reduction;
    }
    
    // 基于颜色一致性调整
    float color_consistency = 1.0f - (attributes.color_gradient / 255.0f);
    lambda *= (1.0f + color_consistency * config_.lambda_color_factor);
    
    // 基于细化级别调整
    if (attributes.refinement_level > 0) {
        lambda *= (1.0f - 0.1f * attributes.refinement_level);
    }
    
    return std::max(0.05f, lambda);
}

float UDFGraphCutIntegrator::computeRefinementBasedSmoothness(
    const openvdb::Coord& coord1,
    const openvdb::Coord& coord2,
    const RefinementGridT& refinement_grid) const {
    
    auto accessor = refinement_grid.getConstAccessor();
    
    int level1 = accessor.getValue(coord1);
    int level2 = accessor.getValue(coord2);
    
    // 细化级别差异越大，平滑权重越小
    int level_diff = std::abs(level1 - level2);
    float smoothness_factor = 1.0f / (1.0f + level_diff * 0.5f);
    
    // 基于平均细化级别调整
    int avg_level = (level1 + level2) / 2;
    if (avg_level < static_cast<int>(config_.refinement_weights.size())) {
        smoothness_factor *= config_.refinement_weights[avg_level];
    }
    
    return smoothness_factor * config_.lambda_base;
}

void UDFGraphCutIntegrator::applyAdaptiveWeighting(VoxelAttributes& attributes) const {
    // 基于置信度的自适应调整
    float confidence_factor = std::max(config_.min_confidence, attributes.confidence);
    
    attributes.alpha_weight *= confidence_factor;
    attributes.lambda_weight *= confidence_factor;
    
    // 基于局部密度的调整
    if (attributes.local_density > 2.0f) {
        attributes.alpha_weight *= 1.2f;  // 高密度区域增强数据项
    } else if (attributes.local_density < 0.5f) {
        attributes.lambda_weight *= 1.5f; // 低密度区域增强平滑项
    }
}

void UDFGraphCutIntegrator::applyConfidenceScaling(VoxelAttributes& attributes) const {
    if (attributes.confidence < config_.min_confidence) {
        // 低置信度区域，增强平滑约束
        attributes.lambda_weight *= 2.0f;
        attributes.alpha_weight *= 0.5f;
    } else if (attributes.confidence > config_.high_confidence) {
        // 高置信度区域，增强数据约束
        attributes.alpha_weight *= 1.5f;
        attributes.lambda_weight *= 0.8f;
    }
}

// ============================================================================
// 几何属性计算方法
// ============================================================================

float UDFGraphCutIntegrator::computeCurvatureAttribute(
    const openvdb::Vec3f& world_pos,
    const PointCloudT& cloud) const {
    
    PointT search_point;
    search_point.x = world_pos.x();
    search_point.y = world_pos.y();
    search_point.z = world_pos.z();
    
    std::vector<int> indices;
    std::vector<float> distances;
    
    if (kdtree_->radiusSearch(search_point, 0.08f, indices, distances) < 10) {
        return 0.0f;
    }
    
    // 计算主曲率
    Eigen::Vector3f centroid(0, 0, 0);
    for (int idx : indices) {
        const auto& pt = cloud.points[idx];
        centroid += Eigen::Vector3f(pt.x, pt.y, pt.z);
    }
    centroid /= indices.size();
    
    // 构建协方差矩阵
    Eigen::Matrix3f covariance = Eigen::Matrix3f::Zero();
    for (int idx : indices) {
        const auto& pt = cloud.points[idx];
        Eigen::Vector3f diff(pt.x - centroid.x(), pt.y - centroid.y(), pt.z - centroid.z());
        covariance += diff * diff.transpose();
    }
    covariance /= indices.size();
    
    // 计算特征值
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver(covariance);
    Eigen::Vector3f eigenvalues = solver.eigenvalues();
    
    // 主曲率近似
    float curvature = eigenvalues(0) / (eigenvalues(0) + eigenvalues(1) + eigenvalues(2));
    return std::min(curvature, 1.0f);
}

float UDFGraphCutIntegrator::computeColorGradientAttribute(
    const openvdb::Vec3f& world_pos,
    const PointCloudT& cloud) const {
    
    PointT search_point;
    search_point.x = world_pos.x();
    search_point.y = world_pos.y();
    search_point.z = world_pos.z();
    
    std::vector<int> indices;
    std::vector<float> distances;
    
    if (kdtree_->radiusSearch(search_point, 0.05f, indices, distances) < 5) {
        return 0.0f;
    }
    
    // 计算颜色梯度
    Eigen::Vector3f avg_color(0, 0, 0);
    for (int idx : indices) {
        const auto& pt = cloud.points[idx];
        avg_color += Eigen::Vector3f(pt.r, pt.g, pt.b);
    }
    avg_color /= indices.size();
    
    float color_variance = 0.0f;
    for (int idx : indices) {
        const auto& pt = cloud.points[idx];
        Eigen::Vector3f color(pt.r, pt.g, pt.b);
        color_variance += (color - avg_color).norm();
    }
    
    return color_variance / indices.size();
}

float UDFGraphCutIntegrator::computePlaneDistanceAttribute(
    const openvdb::Vec3f& world_pos,
    const PointCloudT& cloud) const {
    
    PointT search_point;
    search_point.x = world_pos.x();
    search_point.y = world_pos.y();
    search_point.z = world_pos.z();
    
    std::vector<int> indices;
    std::vector<float> distances;
    
    if (kdtree_->radiusSearch(search_point, 0.1f, indices, distances) < 10) {
        return 1.0f; // 无法拟合平面，返回最大距离
    }
    
    // 创建邻域点云
    PointCloudT::Ptr neighborhood(new PointCloudT);
    for (int idx : indices) {
        neighborhood->push_back(cloud.points[idx]);
    }
    
    // RANSAC平面拟合
    pcl::SampleConsensusModelPlane<PointT>::Ptr model(
        new pcl::SampleConsensusModelPlane<PointT>(neighborhood));
    pcl::RandomSampleConsensus<PointT> ransac(model);
    ransac.setDistanceThreshold(0.02f);
    ransac.setMaxIterations(100);
    
    if (ransac.computeModel()) {
        Eigen::VectorXf coefficients;
        ransac.getModelCoefficients(coefficients);
        
        // 计算点到平面的距离
        float distance = std::abs(coefficients[0] * world_pos.x() + 
                                coefficients[1] * world_pos.y() + 
                                coefficients[2] * world_pos.z() + 
                                coefficients[3]);
        return std::min(distance, 1.0f);
    }
    
    return 1.0f;
}

float UDFGraphCutIntegrator::computeLocalDensityAttribute(
    const openvdb::Vec3f& world_pos,
    const PointCloudT& cloud) const {
    
    PointT search_point;
    search_point.x = world_pos.x();
    search_point.y = world_pos.y();
    search_point.z = world_pos.z();
    
    std::vector<int> indices;
    std::vector<float> distances;
    
    float radius = 0.05f;
    int neighbor_count = kdtree_->radiusSearch(search_point, radius, indices, distances);
    
    // 计算局部密度
    float volume = (4.0f / 3.0f) * M_PI * radius * radius * radius;
    float density = static_cast<float>(neighbor_count) / volume;
    
    return std::min(density / 1000.0f, 5.0f); // 归一化
}

float UDFGraphCutIntegrator::computeNormalVariationAttribute(
    const openvdb::Vec3f& world_pos,
    const PointCloudT& cloud) const {
    
    PointT search_point;
    search_point.x = world_pos.x();
    search_point.y = world_pos.y();
    search_point.z = world_pos.z();
    
    std::vector<int> indices;
    std::vector<float> distances;
    
    if (kdtree_->radiusSearch(search_point, 0.08f, indices, distances) < 5) {
        return 0.0f;
    }
    
    // 计算法向量变化
    Eigen::Vector3f avg_normal(0, 0, 0);
    for (int idx : indices) {
        const auto& pt = cloud.points[idx];
        avg_normal += Eigen::Vector3f(pt.normal_x, pt.normal_y, pt.normal_z);
    }
    avg_normal /= indices.size();
    avg_normal.normalize();
    
    float normal_variance = 0.0f;
    for (int idx : indices) {
        const auto& pt = cloud.points[idx];
        Eigen::Vector3f normal(pt.normal_x, pt.normal_y, pt.normal_z);
        normal.normalize();
        float dot_product = std::abs(avg_normal.dot(normal));
        normal_variance += (1.0f - dot_product);
    }
    
    return normal_variance / indices.size();
}

float UDFGraphCutIntegrator::computeVisibilityPenaltyMapping(
    const openvdb::Vec3f& world_pos,
    const VoxelAttributes& attributes,
    const PointCloudT& cloud) const {
    
    // 简化的可见性惩罚计算
    // 基于局部密度和遮挡情况
    
    float visibility_penalty = 0.0f;
    
    // 低密度区域可能存在遮挡
    if (attributes.local_density < 0.5f) {
        visibility_penalty += 0.3f;
    }
    
    // 高曲率区域可能存在自遮挡
    if (attributes.curvature > 0.7f) {
        visibility_penalty += 0.2f;
    }
    
    // 边缘区域可见性较差
    if (attributes.is_edge) {
        visibility_penalty += 0.1f;
    }
    
    return std::min(visibility_penalty, 1.0f);
}

// ============================================================================
// 验证和统计方法
// ============================================================================

bool UDFGraphCutIntegrator::validateMapping(const EnergyParameterMapping& mapping) const {
    if (mapping.voxel_attributes.empty()) {
        std::cerr << "映射为空" << std::endl;
        return false;
    }
    
    // 检查权重范围
    for (const auto& [coord, attributes] : mapping.voxel_attributes) {
        if (attributes.alpha_weight <= 0 || attributes.alpha_weight > 10.0f) {
            std::cerr << "α权重超出合理范围: " << attributes.alpha_weight << std::endl;
            return false;
        }
        
        if (attributes.lambda_weight <= 0 || attributes.lambda_weight > 10.0f) {
            std::cerr << "λ权重超出合理范围: " << attributes.lambda_weight << std::endl;
            return false;
        }
        
        if (attributes.confidence < 0 || attributes.confidence > 1.0f) {
            std::cerr << "置信度超出范围: " << attributes.confidence << std::endl;
            return false;
        }
    }
    
    return true;
}

void UDFGraphCutIntegrator::updateStatistics(EnergyParameterMapping& mapping) const {
    mapping.total_voxels = static_cast<int>(mapping.voxel_attributes.size());
    mapping.planar_voxels = 0;
    mapping.edge_voxels = 0;
    mapping.corner_voxels = 0;
    mapping.detail_voxels = 0;
    
    float total_alpha = 0.0f;
    float total_lambda = 0.0f;
    float total_confidence = 0.0f;
    
    for (const auto& [coord, attributes] : mapping.voxel_attributes) {
        if (attributes.is_planar) mapping.planar_voxels++;
        if (attributes.is_edge) mapping.edge_voxels++;
        if (attributes.is_corner) mapping.corner_voxels++;
        if (attributes.is_detail) mapping.detail_voxels++;
        
        total_alpha += attributes.alpha_weight;
        total_lambda += attributes.lambda_weight;
        total_confidence += attributes.confidence;
    }
    
    mapping.avg_alpha = total_alpha / mapping.total_voxels;
    mapping.avg_lambda = total_lambda / mapping.total_voxels;
    mapping.avg_confidence = total_confidence / mapping.total_voxels;
}

void UDFGraphCutIntegrator::optimizeCacheUsage() {
    // 清理过大的缓存
    if (attribute_cache_.size() > 100000) {
        attribute_cache_.clear();
        std::cout << "缓存已清理" << std::endl;
    }
}

// ============================================================================
// IntegratorFactory实现
// ============================================================================

std::unique_ptr<UDFGraphCutIntegrator> IntegratorFactory::createStandardIntegrator() {
    IntegrationConfig config;
    // 使用默认配置
    return std::make_unique<UDFGraphCutIntegrator>(config);
}

std::unique_ptr<UDFGraphCutIntegrator> IntegratorFactory::createHighPrecisionIntegrator() {
    IntegrationConfig config;
    
    // 高精度配置
    config.alpha_curvature_factor = 0.8f;
    config.lambda_planar_boost = 3.0f;
    config.lambda_edge_reduction = 0.1f;
    config.planar_threshold = 0.8f;
    config.edge_threshold = 0.3f;
    config.min_confidence = 0.05f;
    
    return std::make_unique<UDFGraphCutIntegrator>(config);
}

std::unique_ptr<UDFGraphCutIntegrator> IntegratorFactory::createFastIntegrator() {
    IntegrationConfig config;
    
    // 快速配置
    config.enable_adaptive_weighting = false;
    config.use_refinement_guidance = false;
    config.apply_confidence_scaling = false;
    
    return std::make_unique<UDFGraphCutIntegrator>(config);
}

} // namespace recon

