/**
 * 优化版UDF构建模块实现文件
 * 
 * 版本: 2.0 - 高性能优化版本
 * 日期: 2025-08-12
 */

#include "optimized_udf_builder.h"
#include <openvdb/tools/LevelSetFilter.h>
#include <openvdb/tools/ParticlesToLevelSet.h>
#include <pcl/common/common.h>
#include <pcl/features/normal_3d.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <chrono>
#include <omp.h>
#include <random>

namespace recon {

// 自适应细化策略实现
void AdaptiveRefinementCriteria::adaptThresholds(const LocalFeatures& features) {
    // 根据局部特征动态调整阈值
    if (features.is_planar) {
        // 平面区域降低曲率阈值，提高颜色阈值
        curvature_threshold *= 0.5f;
        color_gradient_threshold *= 1.5f;
    } else {
        // 非平面区域提高曲率阈值
        curvature_threshold *= 1.2f;
    }
    
    if (features.is_edge) {
        // 边缘区域降低所有阈值以获得更高精度
        curvature_threshold *= 0.3f;
        edge_threshold *= 0.5f;
        color_gradient_threshold *= 0.7f;
    }
    
    // 根据密度调整阈值
    if (features.density > 1000.0f) {  // 高密度区域
        density_variation_threshold *= 1.5f;
    } else if (features.density < 100.0f) {  // 低密度区域
        density_variation_threshold *= 0.5f;
    }
}

float AdaptiveRefinementCriteria::computeRefinementScore(const LocalFeatures& features) const {
    float score = 0.0f;
    
    // 曲率贡献
    if (features.curvature > curvature_threshold) {
        score += (features.curvature / curvature_threshold) * 0.3f;
    }
    
    // 边缘贡献
    if (features.is_edge) {
        score += 0.4f;
    }
    
    // 颜色梯度贡献
    if (features.color_variance > color_gradient_threshold) {
        score += (features.color_variance / color_gradient_threshold) * 0.2f;
    }
    
    // 密度变化贡献
    if (features.density > density_variation_threshold) {
        score += (features.density / density_variation_threshold) * 0.1f;
    }
    
    return score;
}

// OptimizedUDFBuilder实现
OptimizedUDFBuilder::OptimizedUDFBuilder(const OptimizedUDFConfig& config) 
    : config_(config) {
    // 初始化OpenVDB
    openvdb::initialize();
    
    // 初始化KD树
    kdtree_ = KdTreePtr(new pcl::KdTreeFLANN<PointT>);
    
    // 设置线程数
    if (config_.num_threads > 0) {
        omp_set_num_threads(config_.num_threads);
    }
    
    std::cout << "OptimizedUDFBuilder初始化完成" << std::endl;
    std::cout << "  并行处理: " << (config_.use_parallel_processing ? "启用" : "禁用") << std::endl;
    std::cout << "  特征缓存: " << (config_.use_feature_cache ? "启用" : "禁用") << std::endl;
    std::cout << "  线程数: " << omp_get_max_threads() << std::endl;
}

OptimizedUDFBuilder::~OptimizedUDFBuilder() = default;

bool OptimizedUDFBuilder::buildUDF(PointCloudT::Ptr cloud, 
                                  GridT::Ptr& udf_grid, 
                                  ConfidenceGridT::Ptr& confidence_grid) {
    if (!cloud || cloud->empty()) {
        std::cerr << "输入点云为空" << std::endl;
        return false;
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    std::cout << "开始构建优化版UDF，点云大小: " << cloud->size() << std::endl;
    
    // 1. 预计算特征
    std::cout << "1. 预计算点云特征..." << std::endl;
    auto precompute_start = std::chrono::high_resolution_clock::now();
    
    if (!precomputeFeatures(*cloud)) {
        std::cerr << "特征预计算失败" << std::endl;
        return false;
    }
    
    auto precompute_end = std::chrono::high_resolution_clock::now();
    stats_.precompute_time = std::chrono::duration<double>(precompute_end - precompute_start).count();
    
    // 2. 初始化网格
    std::cout << "2. 初始化体素网格..." << std::endl;
    if (!initializeGrids(*cloud, udf_grid, confidence_grid)) {
        std::cerr << "网格初始化失败" << std::endl;
        return false;
    }
    
    // 3. 分块处理
    std::cout << "3. 分块处理体素空间..." << std::endl;
    auto chunks = partitionVoxelSpace(*cloud);
    stats_.parallel_chunks = static_cast<int>(chunks.size());
    
    // 4. 并行处理体素
    std::cout << "4. 并行处理体素块 (" << chunks.size() << " 块)..." << std::endl;
    auto voxel_start = std::chrono::high_resolution_clock::now();
    
    if (!processChunksParallel(chunks, *cloud, udf_grid, confidence_grid)) {
        std::cerr << "体素处理失败" << std::endl;
        return false;
    }
    
    auto voxel_end = std::chrono::high_resolution_clock::now();
    stats_.voxel_processing_time = std::chrono::duration<double>(voxel_end - voxel_start).count();
    
    // 5. 应用滤波
    if (config_.use_gaussian_filter) {
        std::cout << "5. 应用优化高斯滤波..." << std::endl;
        auto filter_start = std::chrono::high_resolution_clock::now();
        
        applyOptimizedGaussianFilter(udf_grid);
        
        auto filter_end = std::chrono::high_resolution_clock::now();
        stats_.filtering_time = std::chrono::duration<double>(filter_end - filter_start).count();
    }
    
    // 计算总时间和性能统计
    auto end_time = std::chrono::high_resolution_clock::now();
    stats_.total_time = std::chrono::duration<double>(end_time - start_time).count();
    
    if (stats_.voxel_processing_time > 0) {
        stats_.voxels_per_second = stats_.total_voxels / stats_.voxel_processing_time;
    }
    
    std::cout << "优化版UDF构建完成!" << std::endl;
    std::cout << "性能统计:" << std::endl;
    std::cout << "  总时间: " << stats_.total_time << "s" << std::endl;
    std::cout << "  预计算时间: " << stats_.precompute_time << "s" << std::endl;
    std::cout << "  体素处理时间: " << stats_.voxel_processing_time << "s" << std::endl;
    std::cout << "  滤波时间: " << stats_.filtering_time << "s" << std::endl;
    std::cout << "  总体素数: " << stats_.total_voxels << std::endl;
    std::cout << "  细化体素数: " << stats_.refined_voxels << std::endl;
    std::cout << "  处理速度: " << static_cast<int>(stats_.voxels_per_second) << " 体素/秒" << std::endl;
    std::cout << "  并行块数: " << stats_.parallel_chunks << std::endl;
    
    return true;
}

bool OptimizedUDFBuilder::precomputeFeatures(const PointCloudT& cloud) {
    // 设置KD树
    PointCloudT::Ptr cloud_ptr(new PointCloudT(cloud));
    kdtree_->setInputCloud(cloud_ptr);
    
    if (!config_.use_feature_cache) {
        return true;  // 如果不使用缓存，直接返回成功
    }
    
    std::cout << "  预计算 " << cloud.size() << " 个点的特征..." << std::endl;
    
    // 并行预计算特征
    feature_cache_.clear();
    feature_cache_.reserve(cloud.size());
    
    std::atomic<int> processed_count(0);
    
    #pragma omp parallel for if(config_.use_parallel_processing)
    for (int i = 0; i < static_cast<int>(cloud.size()); ++i) {
        LocalFeatures features;
        const PointT& point = cloud.points[i];
        openvdb::Vec3f pos(point.x, point.y, point.z);
        
        // 计算几何曲率
        features.curvature = computeGeometricCurvature(pos, cloud);
        
        // 计算局部密度
        std::vector<int> indices;
        std::vector<float> distances;
        int count = kdtree_->radiusSearch(point, config_.density_search_radius, indices, distances);
        float volume = (4.0f / 3.0f) * M_PI * std::pow(config_.density_search_radius, 3);
        features.density = static_cast<float>(count) / volume;
        features.neighbor_count = count;
        
        // 计算颜色方差
        features.color_variance = computeEnhancedColorGradient(pos, cloud);
        
        // 检测边缘
        features.is_edge = detectEdge(pos, cloud);
        
        // 计算平面距离
        Eigen::Vector4f plane_coeffs;
        features.plane_distance = computePlaneDistanceRANSAC(pos, cloud, plane_coeffs);
        features.plane_coeffs = plane_coeffs;
        features.is_planar = (features.plane_distance < config_.plane_distance_threshold);
        
        // 计算法向量（如果点云没有法向量）
        if (std::isnan(point.normal_x) || std::isnan(point.normal_y) || std::isnan(point.normal_z)) {
            // 从协方差矩阵计算法向量
            Eigen::Matrix3f covariance = computeCovarianceMatrix(pos, cloud, config_.curvature_search_radius);
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver(covariance);
            features.normal = solver.eigenvectors().col(0);  // 最小特征值对应的特征向量
        } else {
            features.normal = Eigen::Vector3f(point.normal_x, point.normal_y, point.normal_z);
        }
        
        // 计算特征置信度
        features.feature_confidence = std::min(1.0f, static_cast<float>(count) / 50.0f);  // 基于邻居数量
        
        // 线程安全地存储特征
        setCachedFeatures(i, features);
        
        int current_count = processed_count.fetch_add(1) + 1;
        if (current_count % 10000 == 0) {
            std::cout << "    已处理: " << current_count << "/" << cloud.size() << " 点" << std::endl;
        }
    }
    
    std::cout << "  特征预计算完成，缓存大小: " << feature_cache_.size() << std::endl;
    return true;
}


bool OptimizedUDFBuilder::initializeGrids(const PointCloudT& cloud, 
                                         GridT::Ptr& udf_grid, 
                                         ConfidenceGridT::Ptr& confidence_grid) {
    // 创建UDF网格
    udf_grid = GridT::create(config_.truncation_distance);
    udf_grid->setName("optimized_udf");
    udf_grid->setGridClass(openvdb::GRID_LEVEL_SET);
    
    // 创建置信度网格
    confidence_grid = ConfidenceGridT::create(0.0f);
    confidence_grid->setName("optimized_confidence");
    confidence_grid->setGridClass(openvdb::GRID_FOG_VOLUME);
    
    // 设置变换
    float voxel_size = config_.coarse_voxel_size;
    openvdb::math::Transform::Ptr transform = 
        openvdb::math::Transform::createLinearTransform(voxel_size);
    
    udf_grid->setTransform(transform);
    confidence_grid->setTransform(transform);
    
    std::cout << "  网格初始化完成，体素大小: " << voxel_size << std::endl;
    return true;
}

std::vector<VoxelChunk> OptimizedUDFBuilder::partitionVoxelSpace(const PointCloudT& cloud) {
    // 计算点云边界框
    PointT min_pt, max_pt;
    pcl::getMinMax3D(cloud, min_pt, max_pt);
    
    // 扩展边界框
    openvdb::Vec3f min_bound(min_pt.x - config_.expansion_distance,
                            min_pt.y - config_.expansion_distance,
                            min_pt.z - config_.expansion_distance);
    openvdb::Vec3f max_bound(max_pt.x + config_.expansion_distance,
                            max_pt.y + config_.expansion_distance,
                            max_pt.z + config_.expansion_distance);
    
    // 计算网格尺寸
    float voxel_size = config_.coarse_voxel_size;
    int grid_x = static_cast<int>(std::ceil((max_bound.x() - min_bound.x()) / voxel_size));
    int grid_y = static_cast<int>(std::ceil((max_bound.y() - min_bound.y()) / voxel_size));
    int grid_z = static_cast<int>(std::ceil((max_bound.z() - min_bound.z()) / voxel_size));
    
    // 计算块大小
    int chunk_size = config_.chunk_size;
    int chunks_x = (grid_x + chunk_size - 1) / chunk_size;
    int chunks_y = (grid_y + chunk_size - 1) / chunk_size;
    int chunks_z = (grid_z + chunk_size - 1) / chunk_size;
    
    std::vector<VoxelChunk> chunks;
    chunks.reserve(chunks_x * chunks_y * chunks_z);
    
    int chunk_id = 0;
    for (int cx = 0; cx < chunks_x; ++cx) {
        for (int cy = 0; cy < chunks_y; ++cy) {
            for (int cz = 0; cz < chunks_z; ++cz) {
                // 计算块的边界框
                int start_x = cx * chunk_size;
                int start_y = cy * chunk_size;
                int start_z = cz * chunk_size;
                
                int end_x = std::min(start_x + chunk_size, grid_x);
                int end_y = std::min(start_y + chunk_size, grid_y);
                int end_z = std::min(start_z + chunk_size, grid_z);
                
                openvdb::CoordBBox bbox(openvdb::Coord(start_x, start_y, start_z),
                                       openvdb::Coord(end_x - 1, end_y - 1, end_z - 1));
                
                VoxelChunk chunk(chunk_id++, bbox);
                
                // 预计算块内的体素坐标
                for (int x = start_x; x < end_x; ++x) {
                    for (int y = start_y; y < end_y; ++y) {
                        for (int z = start_z; z < end_z; ++z) {
                            chunk.coords.emplace_back(x, y, z);
                        }
                    }
                }
                
                if (!chunk.coords.empty()) {
                    chunks.push_back(std::move(chunk));
                }
            }
        }
    }
    
    std::cout << "  分块完成: " << chunks.size() << " 块，每块最多 " << chunk_size << "³ 体素" << std::endl;
    return chunks;
}

bool OptimizedUDFBuilder::processChunksParallel(const std::vector<VoxelChunk>& chunks,
                                               const PointCloudT& cloud,
                                               GridT::Ptr& udf_grid,
                                               ConfidenceGridT::Ptr& confidence_grid) {
    std::atomic<int> processed_chunks(0);
    std::atomic<int> total_voxels(0);
    std::atomic<int> refined_voxels(0);
    
    #pragma omp parallel for if(config_.use_parallel_processing)
    for (int i = 0; i < static_cast<int>(chunks.size()); ++i) {
        const auto& chunk = chunks[i];
        
        int chunk_voxels = 0;
        int chunk_refined = 0;
        
        // 处理块内的每个体素
        for (const auto& coord : chunk.coords) {
            // 将体素坐标转换为世界坐标
            openvdb::Vec3f world_pos = udf_grid->transform().indexToWorld(coord);
            
            // 计算UDF值和置信度
            auto [udf_value, confidence] = computeOptimizedVoxelValues(world_pos, cloud);
            
            // 检查是否在截断距离内
            if (udf_value <= config_.truncation_distance) {
                // 线程安全地设置体素值
                {
                    auto udf_accessor = udf_grid->getAccessor();
                    auto conf_accessor = confidence_grid->getAccessor();
                    udf_accessor.setValue(coord, udf_value);
                    conf_accessor.setValue(coord, confidence);
                }
                
                chunk_voxels++;
                
                // 检查是否需要细化
                OptimizedVoxelRefinementInfo refine_info = 
                    computeOptimizedRefinementInfo(coord, cloud, *udf_grid);
                
                if (refine_info.needs_refinement) {
                    // 执行细化
                    chunk_refined += performRefinement(coord, world_pos, cloud, udf_grid, confidence_grid);
                }
            }
        }
        
        // 更新统计信息
        total_voxels.fetch_add(chunk_voxels);
        refined_voxels.fetch_add(chunk_refined);
        
        int current_chunks = processed_chunks.fetch_add(1) + 1;
        if (current_chunks % 10 == 0 || current_chunks == static_cast<int>(chunks.size())) {
            std::cout << "    已处理块: " << current_chunks << "/" << chunks.size() 
                     << " (体素: " << total_voxels.load() << ", 细化: " << refined_voxels.load() << ")" << std::endl;
        }
    }
    
    stats_.total_voxels = total_voxels.load();
    stats_.refined_voxels = refined_voxels.load();
    
    return true;
}

int OptimizedUDFBuilder::performRefinement(const openvdb::Coord& base_coord,
                                          const openvdb::Vec3f& base_world_pos,
                                          const PointCloudT& cloud,
                                          GridT::Ptr& udf_grid,
                                          ConfidenceGridT::Ptr& confidence_grid) {
    int refined_count = 0;
    float coarse_size = config_.coarse_voxel_size;
    float fine_size = config_.fine_voxel_size;
    
    // 在粗体素内进行细化
    int subdivisions = static_cast<int>(coarse_size / fine_size);
    float step = coarse_size / subdivisions;
    
    for (int dx = 0; dx < subdivisions; ++dx) {
        for (int dy = 0; dy < subdivisions; ++dy) {
            for (int dz = 0; dz < subdivisions; ++dz) {
                // 计算细化体素的世界坐标
                openvdb::Vec3f fine_pos(
                    base_world_pos.x() - coarse_size/2 + (dx + 0.5f) * step,
                    base_world_pos.y() - coarse_size/2 + (dy + 0.5f) * step,
                    base_world_pos.z() - coarse_size/2 + (dz + 0.5f) * step
                );
                
                // 计算细化体素的UDF值
                auto [fine_udf, fine_conf] = computeOptimizedVoxelValues(fine_pos, cloud);
                
                if (fine_udf <= config_.truncation_distance) {
                    // 计算细化体素的网格坐标
                    openvdb::Coord fine_coord = udf_grid->transform().worldToIndexCellCentered(fine_pos);
                    
                    // 设置细化体素值
                    auto udf_accessor = udf_grid->getAccessor();
                    auto conf_accessor = confidence_grid->getAccessor();
                    udf_accessor.setValue(fine_coord, fine_udf);
                    conf_accessor.setValue(fine_coord, fine_conf);
                    
                    refined_count++;
                }
            }
        }
    }
    
    return refined_count;
}

OptimizedVoxelRefinementInfo OptimizedUDFBuilder::computeOptimizedRefinementInfo(
    const openvdb::Coord& coord,
    const PointCloudT& cloud,
    const GridT& grid) {
    
    OptimizedVoxelRefinementInfo info;
    
    // 获取世界坐标
    openvdb::Vec3f world_pos = grid.transform().indexToWorld(coord);
    
    // 查找最近的点来获取缓存的特征
    PointT search_point;
    search_point.x = world_pos.x();
    search_point.y = world_pos.y();
    search_point.z = world_pos.z();
    
    std::vector<int> indices(1);
    std::vector<float> distances(1);
    
    if (kdtree_->nearestKSearch(search_point, 1, indices, distances) > 0) {
        int nearest_idx = indices[0];
        
        if (config_.use_feature_cache) {
            info.features = getCachedFeatures(nearest_idx);
        } else {
            // 实时计算特征
            info.features.curvature = computeGeometricCurvature(world_pos, cloud);
            info.features.color_variance = computeEnhancedColorGradient(world_pos, cloud);
            info.features.is_edge = detectEdge(world_pos, cloud);
            
            Eigen::Vector4f plane_coeffs;
            info.features.plane_distance = computePlaneDistanceRANSAC(world_pos, cloud, plane_coeffs);
            info.features.is_planar = (info.features.plane_distance < config_.plane_distance_threshold);
        }
    }
    
    // 创建自适应细化策略
    AdaptiveRefinementCriteria criteria;
    criteria.curvature_threshold = config_.curvature_threshold;
    criteria.edge_threshold = config_.edge_threshold;
    criteria.color_gradient_threshold = config_.color_gradient_threshold;
    criteria.density_variation_threshold = config_.density_threshold_multiplier;
    
    // 根据特征调整阈值
    criteria.adaptThresholds(info.features);
    
    // 计算细化评分
    info.refinement_score = criteria.computeRefinementScore(info.features);
    
    // 判断是否需要细化
    info.needs_refinement = needsAdaptiveRefinement(info, criteria);
    
    // 设置细化原因标记
    info.refine_for_curvature = (info.features.curvature > criteria.curvature_threshold);
    info.refine_for_edge = info.features.is_edge;
    info.refine_for_color = (info.features.color_variance > criteria.color_gradient_threshold);
    info.refine_for_density = (info.features.density > criteria.density_variation_threshold);
    
    return info;
}

bool OptimizedUDFBuilder::needsAdaptiveRefinement(const OptimizedVoxelRefinementInfo& info,
                                                 const AdaptiveRefinementCriteria& criteria) {
    // 基于综合评分的细化判断
    return info.refinement_score > 1.0f;
}

std::pair<float, float> OptimizedUDFBuilder::computeOptimizedVoxelValues(
    const openvdb::Vec3f& world_pos,
    const PointCloudT& cloud) {
    
    PointT search_point;
    search_point.x = world_pos.x();
    search_point.y = world_pos.y();
    search_point.z = world_pos.z();
    
    std::vector<int> indices(1);
    std::vector<float> distances(1);
    
    float udf_value = config_.truncation_distance;
    float confidence = 0.0f;
    
    if (kdtree_->nearestKSearch(search_point, 1, indices, distances) > 0) {
        udf_value = std::sqrt(distances[0]);
        udf_value = std::min(udf_value, config_.truncation_distance);
        
        // 计算增强的置信度
        LocalFeatures features;
        if (config_.use_feature_cache && indices[0] < static_cast<int>(feature_cache_.size())) {
            features = getCachedFeatures(indices[0]);
        }
        
        confidence = computeEnhancedConfidence(world_pos, cloud, udf_value, features);
    }
    
    return {udf_value, confidence};
}

float OptimizedUDFBuilder::computeEnhancedConfidence(const openvdb::Vec3f& world_pos,
                                                    const PointCloudT& cloud,
                                                    float distance,
                                                    const LocalFeatures& features) {
    // 距离因子（距离越近置信度越高）
    float distance_factor = 1.0f - (distance / config_.truncation_distance);
    
    // 特征置信度因子
    float feature_factor = features.feature_confidence;
    
    // 密度因子
    float density_factor = std::min(1.0f, features.density / 1000.0f);
    
    // 平面性因子
    float planarity_factor = features.is_planar ? 1.2f : 0.8f;
    
    // 边缘因子（边缘区域置信度稍低）
    float edge_factor = features.is_edge ? 0.9f : 1.0f;
    
    // 加权组合
    float confidence = config_.density_weight * density_factor +
                      config_.color_weight * (1.0f - features.color_variance) +
                      config_.planarity_weight * planarity_factor +
                      config_.curvature_weight * (1.0f / (1.0f + features.curvature));
    
    confidence *= distance_factor * feature_factor * edge_factor;
    
    return std::max(0.0f, std::min(1.0f, confidence));
}


float OptimizedUDFBuilder::computeGeometricCurvature(const openvdb::Vec3f& pos, 
                                                     const PointCloudT& cloud) {
    // 使用主成分分析计算几何曲率
    Eigen::Matrix3f covariance = computeCovarianceMatrix(pos, cloud, config_.curvature_search_radius);
    
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver(covariance);
    auto eigenvalues = solver.eigenvalues();
    
    // 确保特征值按升序排列
    std::sort(eigenvalues.data(), eigenvalues.data() + 3);
    
    float lambda1 = eigenvalues[0];  // 最小特征值
    float lambda2 = eigenvalues[1];  // 中间特征值
    float lambda3 = eigenvalues[2];  // 最大特征值
    
    // 避免除零
    if (lambda3 < 1e-6f) {
        return 0.0f;
    }
    
    // 计算曲率度量
    float curvature = lambda1 / (lambda1 + lambda2 + lambda3);
    
    return curvature;
}

float OptimizedUDFBuilder::computePlaneDistanceRANSAC(const openvdb::Vec3f& pos,
                                                     const PointCloudT& cloud,
                                                     Eigen::Vector4f& plane_coeffs) {
    PointT search_point;
    search_point.x = pos.x();
    search_point.y = pos.y();
    search_point.z = pos.z();
    
    std::vector<int> indices;
    std::vector<float> distances;
    
    int count = kdtree_->radiusSearch(search_point, config_.plane_search_radius, indices, distances);
    
    if (count < config_.ransac_min_inliers) {
        plane_coeffs = Eigen::Vector4f::Zero();
        return std::numeric_limits<float>::max();
    }
    
    // 执行RANSAC平面拟合
    std::vector<int> inliers;
    if (fitPlaneRANSAC(pos, cloud, plane_coeffs, inliers)) {
        // 计算点到平面的距离
        float distance = std::abs(plane_coeffs[0] * pos.x() + 
                                 plane_coeffs[1] * pos.y() + 
                                 plane_coeffs[2] * pos.z() + 
                                 plane_coeffs[3]);
        float norm = std::sqrt(plane_coeffs[0] * plane_coeffs[0] + 
                              plane_coeffs[1] * plane_coeffs[1] + 
                              plane_coeffs[2] * plane_coeffs[2]);
        
        return (norm > 1e-6f) ? (distance / norm) : std::numeric_limits<float>::max();
    }
    
    return std::numeric_limits<float>::max();
}

float OptimizedUDFBuilder::computeEnhancedColorGradient(const openvdb::Vec3f& pos, 
                                                       const PointCloudT& cloud) {
    PointT search_point;
    search_point.x = pos.x();
    search_point.y = pos.y();
    search_point.z = pos.z();
    
    std::vector<int> indices;
    std::vector<float> distances;
    
    int count = kdtree_->radiusSearch(search_point, config_.color_search_radius, indices, distances);
    
    if (count < 2) {
        return 0.0f;
    }
    
    // 计算颜色统计
    float sum_r = 0, sum_g = 0, sum_b = 0;
    for (int idx : indices) {
        const PointT& pt = cloud.points[idx];
        sum_r += pt.r;
        sum_g += pt.g;
        sum_b += pt.b;
    }
    
    float mean_r = sum_r / count;
    float mean_g = sum_g / count;
    float mean_b = sum_b / count;
    
    // 计算颜色方差
    float variance_r = 0, variance_g = 0, variance_b = 0;
    for (int idx : indices) {
        const PointT& pt = cloud.points[idx];
        float dr = pt.r - mean_r;
        float dg = pt.g - mean_g;
        float db = pt.b - mean_b;
        variance_r += dr * dr;
        variance_g += dg * dg;
        variance_b += db * db;
    }
    
    variance_r /= count;
    variance_g /= count;
    variance_b /= count;
    
    // 计算总颜色梯度
    float color_gradient = std::sqrt(variance_r + variance_g + variance_b) / 255.0f;
    
    return std::min(1.0f, color_gradient);
}

bool OptimizedUDFBuilder::detectEdge(const openvdb::Vec3f& pos, const PointCloudT& cloud) {
    // 使用法向量变化检测边缘
    PointT search_point;
    search_point.x = pos.x();
    search_point.y = pos.y();
    search_point.z = pos.z();
    
    std::vector<int> indices;
    std::vector<float> distances;
    
    int count = kdtree_->radiusSearch(search_point, config_.curvature_search_radius, indices, distances);
    
    if (count < 5) {
        return false;
    }
    
    // 计算法向量的方差
    Eigen::Vector3f mean_normal = Eigen::Vector3f::Zero();
    int valid_normals = 0;
    
    for (int idx : indices) {
        const PointT& pt = cloud.points[idx];
        if (!std::isnan(pt.normal_x) && !std::isnan(pt.normal_y) && !std::isnan(pt.normal_z)) {
            mean_normal += Eigen::Vector3f(pt.normal_x, pt.normal_y, pt.normal_z);
            valid_normals++;
        }
    }
    
    if (valid_normals < 3) {
        return false;
    }
    
    mean_normal /= valid_normals;
    mean_normal.normalize();
    
    // 计算法向量偏差
    float deviation_sum = 0.0f;
    for (int idx : indices) {
        const PointT& pt = cloud.points[idx];
        if (!std::isnan(pt.normal_x) && !std::isnan(pt.normal_y) && !std::isnan(pt.normal_z)) {
            Eigen::Vector3f normal(pt.normal_x, pt.normal_y, pt.normal_z);
            normal.normalize();
            
            float dot_product = std::abs(mean_normal.dot(normal));
            deviation_sum += (1.0f - dot_product);
        }
    }
    
    float average_deviation = deviation_sum / valid_normals;
    
    return average_deviation > config_.edge_threshold;
}

Eigen::Matrix3f OptimizedUDFBuilder::computeCovarianceMatrix(const openvdb::Vec3f& pos,
                                                            const PointCloudT& cloud,
                                                            float search_radius) {
    PointT search_point;
    search_point.x = pos.x();
    search_point.y = pos.y();
    search_point.z = pos.z();
    
    std::vector<int> indices;
    std::vector<float> distances;
    
    int count = kdtree_->radiusSearch(search_point, search_radius, indices, distances);
    
    if (count < 3) {
        return Eigen::Matrix3f::Identity();
    }
    
    // 计算质心
    Eigen::Vector3f centroid = Eigen::Vector3f::Zero();
    for (int idx : indices) {
        const PointT& pt = cloud.points[idx];
        centroid += Eigen::Vector3f(pt.x, pt.y, pt.z);
    }
    centroid /= count;
    
    // 计算协方差矩阵
    Eigen::Matrix3f covariance = Eigen::Matrix3f::Zero();
    for (int idx : indices) {
        const PointT& pt = cloud.points[idx];
        Eigen::Vector3f diff = Eigen::Vector3f(pt.x, pt.y, pt.z) - centroid;
        covariance += diff * diff.transpose();
    }
    covariance /= count;
    
    return covariance;
}

bool OptimizedUDFBuilder::fitPlaneRANSAC(const openvdb::Vec3f& center,
                                        const PointCloudT& cloud,
                                        Eigen::Vector4f& plane_coeffs,
                                        std::vector<int>& inliers) {
    PointT search_point;
    search_point.x = center.x();
    search_point.y = center.y();
    search_point.z = center.z();
    
    std::vector<int> indices;
    std::vector<float> distances;
    
    int count = kdtree_->radiusSearch(search_point, config_.plane_search_radius, indices, distances);
    
    if (count < config_.ransac_min_inliers) {
        return false;
    }
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, count - 1);
    
    int best_inliers = 0;
    Eigen::Vector4f best_plane = Eigen::Vector4f::Zero();
    
    for (int iter = 0; iter < config_.ransac_max_iterations; ++iter) {
        // 随机选择3个点
        std::vector<int> sample_indices(3);
        for (int i = 0; i < 3; ++i) {
            sample_indices[i] = indices[dis(gen)];
        }
        
        // 确保3个点不共线
        const PointT& p1 = cloud.points[sample_indices[0]];
        const PointT& p2 = cloud.points[sample_indices[1]];
        const PointT& p3 = cloud.points[sample_indices[2]];
        
        Eigen::Vector3f v1(p2.x - p1.x, p2.y - p1.y, p2.z - p1.z);
        Eigen::Vector3f v2(p3.x - p1.x, p3.y - p1.y, p3.z - p1.z);
        Eigen::Vector3f normal = v1.cross(v2);
        
        if (normal.norm() < 1e-6f) {
            continue;  // 共线，跳过
        }
        
        normal.normalize();
        
        // 计算平面方程 ax + by + cz + d = 0
        float d = -(normal.x() * p1.x + normal.y() * p1.y + normal.z() * p1.z);
        Eigen::Vector4f current_plane(normal.x(), normal.y(), normal.z(), d);
        
        // 计算内点数量
        int current_inliers = 0;
        for (int idx : indices) {
            const PointT& pt = cloud.points[idx];
            float distance = std::abs(current_plane[0] * pt.x + 
                                    current_plane[1] * pt.y + 
                                    current_plane[2] * pt.z + 
                                    current_plane[3]);
            
            if (distance < config_.ransac_distance_threshold) {
                current_inliers++;
            }
        }
        
        if (current_inliers > best_inliers) {
            best_inliers = current_inliers;
            best_plane = current_plane;
        }
    }
    
    if (best_inliers >= config_.ransac_min_inliers) {
        plane_coeffs = best_plane;
        
        // 收集最终的内点
        inliers.clear();
        for (int idx : indices) {
            const PointT& pt = cloud.points[idx];
            float distance = std::abs(best_plane[0] * pt.x + 
                                    best_plane[1] * pt.y + 
                                    best_plane[2] * pt.z + 
                                    best_plane[3]);
            
            if (distance < config_.ransac_distance_threshold) {
                inliers.push_back(idx);
            }
        }
        
        return true;
    }
    
    return false;
}

LocalFeatures OptimizedUDFBuilder::getCachedFeatures(int point_index) const {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    auto it = feature_cache_.find(point_index);
    if (it != feature_cache_.end()) {
        return it->second;
    }
    return LocalFeatures{};  // 返回默认特征
}

void OptimizedUDFBuilder::setCachedFeatures(int point_index, const LocalFeatures& features) {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    feature_cache_[point_index] = features;
}

void OptimizedUDFBuilder::updateStats(const std::string& operation, double time, int count) {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    // 这里可以添加更详细的统计信息收集
    // 目前主要统计信息在主函数中更新
}

void OptimizedUDFBuilder::applyOptimizedGaussianFilter(GridT::Ptr grid) {
    // 使用OpenVDB的高效滤波器
    openvdb::tools::LevelSetFilter<GridT> filter(*grid);
    filter.gaussian(config_.gaussian_width);
    
    std::cout << "  高斯滤波完成，滤波宽度: " << config_.gaussian_width << std::endl;
}

} // namespace recon

