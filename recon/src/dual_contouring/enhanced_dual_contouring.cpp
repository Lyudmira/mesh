/**
 * 增强版双重轮廓模块实现文件
 * 
 * 版本: 2.0 - 增强版本
 * 日期: 2025-08-12
 */

#include "enhanced_dual_contouring.h"
#include <openvdb/tools/Gradient.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/common.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <chrono>
#include <omp.h>

namespace recon {

// 立方体边的定义 (12条边)
static const std::array<std::pair<int, int>, 12> CUBE_EDGES = {{
    {0, 1}, {1, 3}, {3, 2}, {2, 0},  // 底面4条边
    {4, 5}, {5, 7}, {7, 6}, {6, 4},  // 顶面4条边
    {0, 4}, {1, 5}, {2, 6}, {3, 7}   // 垂直4条边
}};

// 立方体顶点偏移 (8个顶点)
static const std::array<openvdb::Coord, 8> CUBE_VERTICES = {{
    {0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 0},
    {0, 0, 1}, {1, 0, 1}, {0, 1, 1}, {1, 1, 1}
}};

// ============================================================================
// EnhancedQEFData 实现
// ============================================================================

void EnhancedQEFData::addConstraint(const Eigen::Vector3f& point,
                                   const Eigen::Vector3f& normal,
                                   float weight,
                                   bool is_feature) {
    Eigen::Vector3f n = normal.normalized() * weight;
    
    // 更新系数矩阵和右端向量
    A += n * n.transpose();
    b += n * (n.dot(point));
    c += weight * weight * (n.dot(point) * n.dot(point));
    ++num_constraints;
    
    // 更新质心和边界框
    if (num_constraints == 1) {
        centroid = point;
        min_bound = point;
        max_bound = point;
    } else {
        centroid = (centroid * (num_constraints - 1) + point) / num_constraints;
        min_bound = min_bound.cwiseMin(point);
        max_bound = max_bound.cwiseMax(point);
    }
    
    // 如果是特征约束，单独存储
    if (is_feature) {
        feature_points.push_back(point);
        feature_normals.push_back(normal);
        feature_weights.push_back(weight);
    }
}

void EnhancedQEFData::setAnisotropicWeights(float normal_w, float tangent_w) {
    normal_weight = normal_w;
    tangent_weight = tangent_w;
}

Eigen::Vector3f EnhancedQEFData::solve(float regularization) const {
    if (num_constraints == 0) {
        return centroid;
    }
    
    // 添加正则化项
    Eigen::Matrix3f regularized_A = A + Eigen::Matrix3f::Identity() * regularization;
    
    // 使用LDLT分解求解
    Eigen::LDLT<Eigen::Matrix3f> ldlt(regularized_A);
    if (ldlt.info() != Eigen::Success) {
        // 如果分解失败，返回质心
        return centroid;
    }
    
    Eigen::Vector3f solution = ldlt.solve(b);
    
    // 检查解的有效性
    if (!isValidSolution(solution)) {
        return centroid;
    }
    
    return solution;
}

Eigen::Vector3f EnhancedQEFData::solveSVD(float regularization, float svd_threshold) const {
    if (num_constraints == 0) {
        return centroid;
    }
    
    // 使用SVD分解进行更稳定的求解
    Eigen::JacobiSVD<Eigen::Matrix3f> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
    
    // 处理奇异值，提高数值稳定性
    Eigen::Vector3f singular_values = svd.singularValues();
    for (int i = 0; i < 3; ++i) {
        if (singular_values[i] < svd_threshold) {
            singular_values[i] = regularization;
        } else {
            singular_values[i] += regularization;
        }
    }
    
    // 重构并求解
    Eigen::Matrix3f regularized_A = svd.matrixU() * 
                                   singular_values.asDiagonal() * 
                                   svd.matrixV().transpose();
    
    Eigen::LDLT<Eigen::Matrix3f> ldlt(regularized_A);
    if (ldlt.info() != Eigen::Success) {
        return centroid;
    }
    
    Eigen::Vector3f solution = ldlt.solve(b);
    
    if (!isValidSolution(solution)) {
        return centroid;
    }
    
    return solution;
}

Eigen::Vector3f EnhancedQEFData::solveAnisotropic(float normal_weight, 
                                                  float tangent_weight, 
                                                  float regularization) const {
    if (num_constraints == 0) {
        return centroid;
    }
    
    // 构建各向异性权重矩阵
    Eigen::Matrix3f weight_matrix = Eigen::Matrix3f::Identity() * tangent_weight;
    
    // 对于每个约束，增强法向量方向的权重
    Eigen::Matrix3f anisotropic_A = A * tangent_weight;
    Eigen::Vector3f anisotropic_b = b * tangent_weight;
    
    // 添加法向量方向的额外权重
    for (size_t i = 0; i < feature_normals.size(); ++i) {
        const Eigen::Vector3f& normal = feature_normals[i];
        float extra_weight = (normal_weight - tangent_weight) * feature_weights[i];
        
        Eigen::Vector3f weighted_normal = normal * extra_weight;
        anisotropic_A += weighted_normal * weighted_normal.transpose();
        anisotropic_b += weighted_normal * (weighted_normal.dot(feature_points[i]));
    }
    
    // 添加正则化
    anisotropic_A += Eigen::Matrix3f::Identity() * regularization;
    
    // 求解
    Eigen::LDLT<Eigen::Matrix3f> ldlt(anisotropic_A);
    if (ldlt.info() != Eigen::Success) {
        return centroid;
    }
    
    Eigen::Vector3f solution = ldlt.solve(anisotropic_b);
    
    if (!isValidSolution(solution)) {
        return centroid;
    }
    
    return solution;
}

Eigen::Vector3f EnhancedQEFData::solveWithFeatures(float regularization) const {
    if (feature_points.empty()) {
        return solve(regularization);
    }
    
    // 使用各向异性求解，特征约束有更高权重
    return solveAnisotropic(normal_weight, tangent_weight, regularization);
}

float EnhancedQEFData::computeError(const Eigen::Vector3f& point) const {
    if (num_constraints == 0) {
        return 0.0f;
    }
    
    // 计算二次误差
    float error = point.transpose() * A * point - 2.0f * b.dot(point) + c;
    return std::max(0.0f, error);
}

bool EnhancedQEFData::isValidSolution(const Eigen::Vector3f& solution) const {
    // 检查解是否包含NaN或无穷大
    if (!solution.allFinite()) {
        return false;
    }
    
    // 检查解是否在合理的边界框内
    Eigen::Vector3f expanded_min = min_bound - Eigen::Vector3f::Constant(1.0f);
    Eigen::Vector3f expanded_max = max_bound + Eigen::Vector3f::Constant(1.0f);
    
    return (solution.array() >= expanded_min.array()).all() && 
           (solution.array() <= expanded_max.array()).all();
}

// ============================================================================
// EnhancedDualContouringExtractor 实现
// ============================================================================

EnhancedDualContouringExtractor::EnhancedDualContouringExtractor(
    const EnhancedDualContouringConfig& config) 
    : config_(config) {
    
    // 初始化OpenVDB
    openvdb::initialize();
    
    // 设置线程数
    if (config_.num_threads > 0) {
        omp_set_num_threads(config_.num_threads);
    }
    
    if (config_.enable_debug_output) {
        std::cout << "增强版双重轮廓提取器初始化完成" << std::endl;
        std::cout << "  并行处理: " << (config_.use_parallel_processing ? "启用" : "禁用") << std::endl;
        std::cout << "  SVD求解器: " << (config_.use_svd_solver ? "启用" : "禁用") << std::endl;
        std::cout << "  特征保持: " << (config_.preserve_sharp_edges ? "启用" : "禁用") << std::endl;
        std::cout << "  线程数: " << omp_get_max_threads() << std::endl;
    }
}

EnhancedDualContouringExtractor::~EnhancedDualContouringExtractor() = default;

bool EnhancedDualContouringExtractor::extractSurface(const UDFGridT::Ptr& udf_grid,
                                                     const PointCloudT::Ptr& cloud,
                                                     pcl::PolygonMesh& mesh) {
    if (!udf_grid) {
        std::cerr << "UDF网格为空" << std::endl;
        return false;
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    if (config_.enable_debug_output) {
        std::cout << "开始增强版双重轮廓表面提取" << std::endl;
    }
    
    // 清空之前的数据
    cells_.clear();
    vertices_.clear();
    normals_.clear();
    triangles_.clear();
    triangle_qualities_.clear();
    cutting_edges_.clear();
    edge_map_.clear();
    
    // 1. 查找切割边
    if (config_.enable_debug_output) {
        std::cout << "1. 查找切割边..." << std::endl;
    }
    auto step_start = std::chrono::high_resolution_clock::now();
    
    findCuttingEdges(udf_grid);
    
    auto step_end = std::chrono::high_resolution_clock::now();
    double edge_finding_time = std::chrono::duration<double>(step_end - step_start).count();
    
    if (config_.enable_debug_output) {
        std::cout << "  找到 " << cutting_edges_.size() << " 条切割边，耗时: " 
                 << edge_finding_time << "s" << std::endl;
    }
    
    // 2. 构建双重单元
    if (config_.enable_debug_output) {
        std::cout << "2. 构建双重单元..." << std::endl;
    }
    step_start = std::chrono::high_resolution_clock::now();
    
    buildDualCellsParallel(udf_grid, cloud);
    
    step_end = std::chrono::high_resolution_clock::now();
    stats_.cell_building_time = std::chrono::duration<double>(step_end - step_start).count();
    
    if (config_.enable_debug_output) {
        std::cout << "  构建了 " << cells_.size() << " 个双重单元，耗时: " 
                 << stats_.cell_building_time << "s" << std::endl;
    }
    
    // 3. 生成三角形
    if (config_.enable_debug_output) {
        std::cout << "3. 生成三角形..." << std::endl;
    }
    step_start = std::chrono::high_resolution_clock::now();
    
    generateCorrectTriangles(udf_grid);
    
    step_end = std::chrono::high_resolution_clock::now();
    stats_.triangulation_time = std::chrono::duration<double>(step_end - step_start).count();
    
    if (config_.enable_debug_output) {
        std::cout << "  生成了 " << triangles_.size() << " 个三角形，耗时: " 
                 << stats_.triangulation_time << "s" << std::endl;
    }
    
    // 4. 质量控制
    if (config_.remove_degenerate || config_.validate_topology) {
        if (config_.enable_debug_output) {
            std::cout << "4. 质量控制..." << std::endl;
        }
        step_start = std::chrono::high_resolution_clock::now();
        
        performQualityControl();
        
        step_end = std::chrono::high_resolution_clock::now();
        stats_.quality_control_time = std::chrono::duration<double>(step_end - step_start).count();
        
        if (config_.enable_debug_output) {
            std::cout << "  质量控制完成，耗时: " << stats_.quality_control_time << "s" << std::endl;
            std::cout << "  移除退化三角形: " << stats_.degenerate_removed << std::endl;
            std::cout << "  拓扑错误: " << stats_.topology_errors << std::endl;
        }
    }
    
    // 5. 构建PCL网格
    if (config_.enable_debug_output) {
        std::cout << "5. 构建PCL网格..." << std::endl;
    }
    
    buildPCLMesh(mesh);
    
    // 计算总时间和统计信息
    auto end_time = std::chrono::high_resolution_clock::now();
    stats_.total_time = std::chrono::duration<double>(end_time - start_time).count();
    
    updateStatistics();
    
    if (config_.enable_debug_output) {
        std::cout << "增强版双重轮廓表面提取完成!" << std::endl;
        std::cout << "性能统计:" << std::endl;
        std::cout << "  总时间: " << stats_.total_time << "s" << std::endl;
        std::cout << "  单元构建时间: " << stats_.cell_building_time << "s" << std::endl;
        std::cout << "  QEF求解时间: " << stats_.qef_solving_time << "s" << std::endl;
        std::cout << "  三角化时间: " << stats_.triangulation_time << "s" << std::endl;
        std::cout << "  质量控制时间: " << stats_.quality_control_time << "s" << std::endl;
        std::cout << "  生成顶点数: " << stats_.vertices_generated << std::endl;
        std::cout << "  生成三角形数: " << stats_.triangles_generated << std::endl;
        std::cout << "  特征单元数: " << stats_.feature_cells << std::endl;
        std::cout << "  平均QEF误差: " << stats_.average_qef_error << std::endl;
        std::cout << "  平均顶点置信度: " << stats_.average_vertex_confidence << std::endl;
        std::cout << "  网格质量评分: " << stats_.mesh_quality_score << std::endl;
    }
    
    return true;
}


void EnhancedDualContouringExtractor::findCuttingEdges(const UDFGridT::Ptr& udf_grid) {
    cutting_edges_.clear();
    edge_map_.clear();
    
    auto udf_accessor = udf_grid->getConstAccessor();
    
    // 获取活跃体素的边界框
    openvdb::CoordBBox bbox;
    udf_grid->evalActiveVoxelBoundingBox(bbox);
    
    int edge_count = 0;
    
    // 遍历所有可能的单元
    for (int z = bbox.min().z(); z < bbox.max().z(); ++z) {
        for (int y = bbox.min().y(); y < bbox.max().y(); ++y) {
            for (int x = bbox.min().x(); x < bbox.max().x(); ++x) {
                openvdb::Coord base_coord(x, y, z);
                
                // 检查这个单元的12条边
                for (int edge_idx = 0; edge_idx < 12; ++edge_idx) {
                    auto [coord1, coord2] = getEdgeCoords(base_coord, edge_idx);
                    
                    // 检查这条边是否被切割
                    if (isEdgeCut(coord1, coord2, udf_grid)) {
                        // 避免重复添加边
                        std::pair<openvdb::Coord, openvdb::Coord> edge_key = 
                            (coord1 < coord2) ? std::make_pair(coord1, coord2) : std::make_pair(coord2, coord1);
                        
                        if (edge_map_.find(edge_key) == edge_map_.end()) {
                            EdgeInfo edge_info;
                            edge_info.coord1 = coord1;
                            edge_info.coord2 = coord2;
                            edge_info.edge_index = edge_idx;
                            edge_info.is_cut = true;
                            
                            cutting_edges_.push_back(edge_info);
                            edge_map_[edge_key] = edge_count++;
                        }
                    }
                }
            }
        }
    }
    
    stats_.total_cells = (bbox.max().x() - bbox.min().x()) * 
                        (bbox.max().y() - bbox.min().y()) * 
                        (bbox.max().z() - bbox.min().z());
}

void EnhancedDualContouringExtractor::buildDualCellsParallel(const UDFGridT::Ptr& udf_grid,
                                                            const PointCloudT::Ptr& cloud) {
    // 收集所有需要处理的单元坐标
    std::vector<openvdb::Coord> active_coords;
    
    // 从切割边推断需要处理的单元
    std::unordered_set<openvdb::Coord> coord_set;
    for (const auto& edge : cutting_edges_) {
        // 每条切割边可能影响多个单元
        std::vector<openvdb::Coord> adjacent = getAdjacentCells(edge.coord1, edge.coord2, edge.edge_index);
        for (const auto& coord : adjacent) {
            coord_set.insert(coord);
        }
    }
    
    active_coords.assign(coord_set.begin(), coord_set.end());
    
    if (config_.enable_debug_output) {
        std::cout << "  需要处理 " << active_coords.size() << " 个单元" << std::endl;
    }
    
    // 并行处理单元
    std::atomic<int> processed_count(0);
    std::atomic<int> feature_count(0);
    
    auto qef_start = std::chrono::high_resolution_clock::now();
    
    #pragma omp parallel for if(config_.use_parallel_processing)
    for (int i = 0; i < static_cast<int>(active_coords.size()); ++i) {
        const openvdb::Coord& coord = active_coords[i];
        
        // 处理单个单元
        EnhancedDualCell cell = processSingleCell(coord, udf_grid, cloud);
        
        if (cell.has_vertex) {
            // 线程安全地添加顶点和单元
            int vertex_idx = addVertexThreadSafe(cell.vertex_position, cell.vertex_normal);
            cell.vertex_index = vertex_idx;
            
            // 存储单元
            {
                std::lock_guard<std::mutex> lock(cells_mutex_);
                cells_[coord] = cell;
            }
            
            if (cell.is_feature_cell) {
                feature_count.fetch_add(1);
            }
        }
        
        int current_count = processed_count.fetch_add(1) + 1;
        if (config_.enable_debug_output && (current_count % 1000 == 0 || current_count == static_cast<int>(active_coords.size()))) {
            std::cout << "    已处理单元: " << current_count << "/" << active_coords.size() << std::endl;
        }
    }
    
    auto qef_end = std::chrono::high_resolution_clock::now();
    stats_.qef_solving_time = std::chrono::duration<double>(qef_end - qef_start).count();
    
    stats_.active_cells = static_cast<int>(cells_.size());
    stats_.feature_cells = feature_count.load();
}

EnhancedDualCell EnhancedDualContouringExtractor::processSingleCell(
    const openvdb::Coord& coord,
    const UDFGridT::Ptr& udf_grid,
    const PointCloudT::Ptr& cloud) {
    
    EnhancedDualCell cell;
    cell.coord = coord;
    cell.has_vertex = false;
    cell.vertex_index = -1;
    
    // 检查这个单元是否与表面相交
    std::vector<float> corner_values(8);
    bool has_positive = false, has_negative = false;
    
    for (int i = 0; i < 8; ++i) {
        openvdb::Coord corner_coord = coord + CUBE_VERTICES[i];
        float value = getUDFValue(openvdb::Vec3f(corner_coord.x(), corner_coord.y(), corner_coord.z()), udf_grid);
        corner_values[i] = value;
        
        if (value > 0) has_positive = true;
        else if (value < 0) has_negative = true;
    }
    
    // 如果没有符号变化，这个单元不与表面相交
    if (!(has_positive && has_negative)) {
        return cell;
    }
    
    // 计算边交点并构建QEF
    for (int edge_idx = 0; edge_idx < 12; ++edge_idx) {
        const auto& edge = CUBE_EDGES[edge_idx];
        int v1_idx = edge.first;
        int v2_idx = edge.second;
        
        float val1 = corner_values[v1_idx];
        float val2 = corner_values[v2_idx];
        
        // 检查这条边是否被切割
        if ((val1 > 0 && val2 < 0) || (val1 < 0 && val2 > 0)) {
            openvdb::Coord coord1 = coord + CUBE_VERTICES[v1_idx];
            openvdb::Coord coord2 = coord + CUBE_VERTICES[v2_idx];
            
            // 计算边交点
            EnhancedEdgeIntersection intersection = 
                computeEnhancedEdgeIntersection(coord1, coord2, edge_idx, udf_grid, cloud);
            
            if (intersection.is_valid) {
                cell.intersections.push_back(intersection);
                
                // 添加到QEF约束
                cell.qef.addConstraint(
                    Eigen::Vector3f(intersection.position.x(), intersection.position.y(), intersection.position.z()),
                    Eigen::Vector3f(intersection.normal.x(), intersection.normal.y(), intersection.normal.z()),
                    intersection.confidence,
                    intersection.is_feature
                );
            }
        }
    }
    
    // 如果有足够的约束，求解QEF
    if (cell.qef.num_constraints > 0) {
        // 检测特征
        if (config_.preserve_sharp_edges) {
            detectSharpFeatures(cell, udf_grid, cloud);
        }
        
        // 设置各向异性权重
        cell.qef.setAnisotropicWeights(config_.anisotropic_normal_weight, 
                                      config_.anisotropic_tangent_weight);
        
        // 求解QEF获得顶点位置
        cell.vertex_position = solveEnhancedQEF(cell);
        
        // 计算QEF误差
        Eigen::Vector3f vertex_eigen(cell.vertex_position.x(), 
                                    cell.vertex_position.y(), 
                                    cell.vertex_position.z());
        cell.qef_error = cell.qef.computeError(vertex_eigen);
        
        // 估计顶点法向量
        cell.vertex_normal = estimateEnhancedNormal(cell.vertex_position, udf_grid, cloud);
        
        // 计算顶点置信度
        cell.vertex_confidence = 1.0f / (1.0f + cell.qef_error);
        
        cell.has_vertex = true;
    }
    
    return cell;
}

EnhancedEdgeIntersection EnhancedDualContouringExtractor::computeEnhancedEdgeIntersection(
    const openvdb::Coord& coord1,
    const openvdb::Coord& coord2,
    int edge_index,
    const UDFGridT::Ptr& udf_grid,
    const PointCloudT::Ptr& cloud) {
    
    EnhancedEdgeIntersection intersection;
    
    // 获取两个端点的UDF值
    openvdb::Vec3f pos1(coord1.x(), coord1.y(), coord1.z());
    openvdb::Vec3f pos2(coord2.x(), coord2.y(), coord2.z());
    
    float val1 = getUDFValue(pos1, udf_grid);
    float val2 = getUDFValue(pos2, udf_grid);
    
    // 检查是否真的被切割
    if ((val1 > 0 && val2 < 0) || (val1 < 0 && val2 > 0)) {
        // 线性插值计算交点位置
        float t = computeIntersectionParameter(val1, val2);
        
        // 将坐标转换为世界坐标
        openvdb::Vec3f world_pos1 = udf_grid->transform().indexToWorld(pos1);
        openvdb::Vec3f world_pos2 = udf_grid->transform().indexToWorld(pos2);
        
        intersection.position = world_pos1 + (world_pos2 - world_pos1) * t;
        
        // 估计交点法向量
        intersection.normal = estimateEnhancedNormal(intersection.position, udf_grid, cloud);
        
        // 计算置信度（基于插值参数和法向量质量）
        float interpolation_confidence = 1.0f - std::abs(t - 0.5f) * 2.0f;  // 中点附近置信度更高
        float normal_confidence = intersection.normal.length();
        intersection.confidence = interpolation_confidence * normal_confidence;
        
        // 计算局部曲率（简化版本）
        intersection.curvature = std::abs(val1 - val2) / computeDistance(world_pos1, world_pos2);
        
        intersection.is_valid = true;
        intersection.is_feature = false;  // 稍后在特征检测中设置
    }
    
    return intersection;
}

openvdb::Vec3f EnhancedDualContouringExtractor::solveEnhancedQEF(const EnhancedDualCell& cell) {
    if (cell.qef.num_constraints == 0) {
        // 如果没有约束，返回单元中心
        openvdb::Vec3f center(cell.coord.x() + 0.5f, 
                             cell.coord.y() + 0.5f, 
                             cell.coord.z() + 0.5f);
        return center;
    }
    
    openvdb::Vec3f solution;
    
    if (config_.use_svd_solver) {
        // 使用SVD求解器
        Eigen::Vector3f eigen_solution = cell.qef.solveSVD(config_.qef_regularization, config_.svd_threshold);
        solution = openvdb::Vec3f(eigen_solution.x(), eigen_solution.y(), eigen_solution.z());
    } else if (cell.is_feature_cell && config_.preserve_sharp_edges) {
        // 使用特征保持求解
        Eigen::Vector3f eigen_solution = cell.qef.solveWithFeatures(config_.qef_regularization);
        solution = openvdb::Vec3f(eigen_solution.x(), eigen_solution.y(), eigen_solution.z());
    } else {
        // 使用标准求解
        Eigen::Vector3f eigen_solution = cell.qef.solve(config_.qef_regularization);
        solution = openvdb::Vec3f(eigen_solution.x(), eigen_solution.y(), eigen_solution.z());
    }
    
    return solution;
}

openvdb::Vec3f EnhancedDualContouringExtractor::estimateEnhancedNormal(
    const openvdb::Vec3f& position,
    const UDFGridT::Ptr& udf_grid,
    const PointCloudT::Ptr& cloud) {
    
    openvdb::Vec3f gradient_normal(0.0f);
    openvdb::Vec3f cloud_normal(0.0f);
    
    // 计算梯度法向量
    if (config_.use_multiscale_gradient) {
        gradient_normal = computeMultiScaleGradient(position, udf_grid);
    } else {
        gradient_normal = computeGradientAtScale(position, udf_grid, 1.0f);
    }
    
    // 从点云估计法向量
    if (config_.use_adaptive_normals && cloud && !cloud->empty()) {
        cloud_normal = estimateNormalFromCloud(position, cloud);
    }
    
    // 融合两种法向量
    if (cloud_normal.lengthSqr() > 1e-6f && gradient_normal.lengthSqr() > 1e-6f) {
        return fuseNormals(gradient_normal, cloud_normal, position, cloud);
    } else if (gradient_normal.lengthSqr() > 1e-6f) {
        return gradient_normal.normalized();
    } else if (cloud_normal.lengthSqr() > 1e-6f) {
        return cloud_normal.normalized();
    } else {
        return openvdb::Vec3f(0.0f, 0.0f, 1.0f);  // 默认法向量
    }
}

openvdb::Vec3f EnhancedDualContouringExtractor::computeMultiScaleGradient(
    const openvdb::Vec3f& position,
    const UDFGridT::Ptr& udf_grid) {
    
    openvdb::Vec3f weighted_gradient(0.0f);
    float total_weight = 0.0f;
    
    for (float scale : config_.gradient_scales) {
        openvdb::Vec3f gradient = computeGradientAtScale(position, udf_grid, scale);
        float confidence = computeGradientConfidence(gradient, scale);
        
        if (confidence > config_.gradient_confidence_threshold) {
            weighted_gradient += gradient * confidence;
            total_weight += confidence;
        }
    }
    
    if (total_weight > 1e-6f) {
        weighted_gradient /= total_weight;
    }
    
    return weighted_gradient;
}

openvdb::Vec3f EnhancedDualContouringExtractor::computeGradientAtScale(
    const openvdb::Vec3f& position,
    const UDFGridT::Ptr& udf_grid,
    float scale) {
    
    // 计算采样点
    float h = scale * udf_grid->transform().voxelSize().x();
    
    openvdb::Vec3f grad;
    
    // 中心差分计算梯度
    grad.x() = (getUDFValue(position + openvdb::Vec3f(h, 0, 0), udf_grid) - 
                getUDFValue(position - openvdb::Vec3f(h, 0, 0), udf_grid)) / (2.0f * h);
    
    grad.y() = (getUDFValue(position + openvdb::Vec3f(0, h, 0), udf_grid) - 
                getUDFValue(position - openvdb::Vec3f(0, h, 0), udf_grid)) / (2.0f * h);
    
    grad.z() = (getUDFValue(position + openvdb::Vec3f(0, 0, h), udf_grid) - 
                getUDFValue(position - openvdb::Vec3f(0, 0, h), udf_grid)) / (2.0f * h);
    
    return grad;
}

float EnhancedDualContouringExtractor::computeGradientConfidence(const openvdb::Vec3f& gradient, 
                                                                float scale) {
    // 基于梯度幅值和尺度计算置信度
    float magnitude = gradient.length();
    float scale_factor = 1.0f / (1.0f + scale);  // 偏向小尺度
    
    return magnitude * scale_factor;
}


openvdb::Vec3f EnhancedDualContouringExtractor::fuseNormals(
    const openvdb::Vec3f& gradient_normal,
    const openvdb::Vec3f& cloud_normal,
    const openvdb::Vec3f& position,
    const PointCloudT::Ptr& cloud) {
    
    // 计算各自的置信度
    float gradient_confidence = gradient_normal.length();
    float cloud_confidence = computeCloudNormalConfidence(cloud_normal, position, cloud);
    
    // 检查一致性
    openvdb::Vec3f norm_grad = gradient_normal.normalized();
    openvdb::Vec3f norm_cloud = cloud_normal.normalized();
    float consistency = norm_grad.dot(norm_cloud);
    
    if (consistency < 0.3f) {
        // 不一致时，选择置信度更高的
        return (gradient_confidence > cloud_confidence) ? norm_grad : norm_cloud;
    }
    
    // 一致时，加权融合
    float total_weight = gradient_confidence + cloud_confidence;
    if (total_weight > 1e-6f) {
        openvdb::Vec3f fused = (norm_grad * gradient_confidence + norm_cloud * cloud_confidence) / total_weight;
        return fused.normalized();
    }
    
    return norm_grad;
}

openvdb::Vec3f EnhancedDualContouringExtractor::estimateNormalFromCloud(
    const openvdb::Vec3f& position,
    const PointCloudT::Ptr& cloud) {
    
    if (!cloud || cloud->empty()) {
        return openvdb::Vec3f(0.0f);
    }
    
    pcl::KdTreeFLANN<PointT> tree;
    tree.setInputCloud(cloud);
    
    PointT query_point;
    query_point.x = position.x();
    query_point.y = position.y();
    query_point.z = position.z();
    
    std::vector<int> indices;
    std::vector<float> distances;
    
    int k = tree.radiusSearch(query_point, config_.normal_estimation_radius, indices, distances);
    
    if (k < 3) {
        return openvdb::Vec3f(0.0f);
    }
    
    // 加权平均法向量
    openvdb::Vec3f weighted_normal(0.0f);
    float total_weight = 0.0f;
    
    for (int i = 0; i < k; ++i) {
        const PointT& point = cloud->points[indices[i]];
        
        // 检查法向量是否有效
        if (std::isnan(point.normal_x) || std::isnan(point.normal_y) || std::isnan(point.normal_z)) {
            continue;
        }
        
        openvdb::Vec3f normal(point.normal_x, point.normal_y, point.normal_z);
        
        // 基于距离的权重
        float weight = 1.0f / (1.0f + distances[i]);
        
        weighted_normal += normal * weight;
        total_weight += weight;
    }
    
    if (total_weight > 1e-6f) {
        weighted_normal /= total_weight;
    }
    
    return weighted_normal;
}

float EnhancedDualContouringExtractor::computeCloudNormalConfidence(
    const openvdb::Vec3f& cloud_normal,
    const openvdb::Vec3f& position,
    const PointCloudT::Ptr& cloud) {
    
    if (!cloud || cloud->empty()) {
        return 0.0f;
    }
    
    // 基于法向量长度和局部一致性计算置信度
    float magnitude_confidence = cloud_normal.length();
    
    // 计算局部法向量一致性
    pcl::KdTreeFLANN<PointT> tree;
    tree.setInputCloud(cloud);
    
    PointT query_point;
    query_point.x = position.x();
    query_point.y = position.y();
    query_point.z = position.z();
    
    std::vector<int> indices;
    std::vector<float> distances;
    
    int k = tree.radiusSearch(query_point, config_.normal_estimation_radius, indices, distances);
    
    if (k < 3) {
        return magnitude_confidence;
    }
    
    openvdb::Vec3f norm_cloud = cloud_normal.normalized();
    float consistency_sum = 0.0f;
    int valid_count = 0;
    
    for (int i = 0; i < k; ++i) {
        const PointT& point = cloud->points[indices[i]];
        
        if (std::isnan(point.normal_x) || std::isnan(point.normal_y) || std::isnan(point.normal_z)) {
            continue;
        }
        
        openvdb::Vec3f normal(point.normal_x, point.normal_y, point.normal_z);
        normal.normalize();
        
        consistency_sum += std::abs(norm_cloud.dot(normal));
        valid_count++;
    }
    
    float consistency_confidence = (valid_count > 0) ? (consistency_sum / valid_count) : 0.0f;
    
    return magnitude_confidence * consistency_confidence;
}

void EnhancedDualContouringExtractor::detectSharpFeatures(EnhancedDualCell& cell,
                                                          const UDFGridT::Ptr& udf_grid,
                                                          const PointCloudT::Ptr& cloud) {
    cell.is_feature_cell = false;
    cell.feature_edges.clear();
    cell.feature_strength = 0.0f;
    
    if (cell.intersections.size() < 2) {
        return;
    }
    
    // 检测尖锐边缘
    std::vector<bool> is_sharp_edge(cell.intersections.size(), false);
    int sharp_count = 0;
    
    for (size_t i = 0; i < cell.intersections.size(); ++i) {
        for (size_t j = i + 1; j < cell.intersections.size(); ++j) {
            const auto& int1 = cell.intersections[i];
            const auto& int2 = cell.intersections[j];
            
            if (isSharpEdge(int1.position, int2.position, int1.normal, int2.normal)) {
                is_sharp_edge[i] = true;
                is_sharp_edge[j] = true;
                sharp_count++;
            }
        }
    }
    
    // 如果有尖锐边缘，标记为特征单元
    if (sharp_count > 0) {
        cell.is_feature_cell = true;
        cell.feature_strength = static_cast<float>(sharp_count) / cell.intersections.size();
        
        // 收集特征边
        for (size_t i = 0; i < cell.intersections.size(); ++i) {
            if (is_sharp_edge[i]) {
                cell.feature_edges.push_back(static_cast<int>(i));
                cell.intersections[i].is_feature = true;
                
                // 添加特征约束到QEF
                cell.qef.addConstraint(
                    Eigen::Vector3f(cell.intersections[i].position.x(), 
                                   cell.intersections[i].position.y(), 
                                   cell.intersections[i].position.z()),
                    Eigen::Vector3f(cell.intersections[i].normal.x(), 
                                   cell.intersections[i].normal.y(), 
                                   cell.intersections[i].normal.z()),
                    cell.intersections[i].confidence * config_.feature_weight_multiplier,
                    true
                );
            }
        }
    }
    
    // 检测角点
    if (config_.detect_corners && cell.intersections.size() >= 3) {
        std::vector<openvdb::Vec3f> normals;
        for (const auto& intersection : cell.intersections) {
            normals.push_back(intersection.normal);
        }
        
        if (isCorner(normals)) {
            cell.is_feature_cell = true;
            cell.feature_strength = std::max(cell.feature_strength, 0.8f);
        }
    }
}

bool EnhancedDualContouringExtractor::isSharpEdge(const openvdb::Vec3f& p1, 
                                                  const openvdb::Vec3f& p2,
                                                  const openvdb::Vec3f& n1, 
                                                  const openvdb::Vec3f& n2) {
    // 计算法向量夹角
    openvdb::Vec3f norm_n1 = n1.normalized();
    openvdb::Vec3f norm_n2 = n2.normalized();
    
    float dot_product = norm_n1.dot(norm_n2);
    dot_product = std::max(-1.0f, std::min(1.0f, dot_product));  // 限制在[-1, 1]范围内
    
    float angle_radians = std::acos(dot_product);
    float angle_degrees = angle_radians * 180.0f / M_PI;
    
    return angle_degrees > config_.sharp_edge_threshold;
}

bool EnhancedDualContouringExtractor::isCorner(const std::vector<openvdb::Vec3f>& normals) {
    if (normals.size() < 3) {
        return false;
    }
    
    // 检查是否有三个或更多法向量形成大角度
    int large_angle_count = 0;
    
    for (size_t i = 0; i < normals.size(); ++i) {
        for (size_t j = i + 1; j < normals.size(); ++j) {
            openvdb::Vec3f n1 = normals[i].normalized();
            openvdb::Vec3f n2 = normals[j].normalized();
            
            float dot_product = n1.dot(n2);
            dot_product = std::max(-1.0f, std::min(1.0f, dot_product));
            
            float angle_degrees = std::acos(dot_product) * 180.0f / M_PI;
            
            if (angle_degrees > config_.corner_threshold) {
                large_angle_count++;
            }
        }
    }
    
    // 如果有足够多的大角度，认为是角点
    return large_angle_count >= 2;
}

void EnhancedDualContouringExtractor::generateCorrectTriangles(const UDFGridT::Ptr& udf_grid) {
    triangles_.clear();
    
    // 遍历所有切割边，为每条边生成对应的四边形（分解为两个三角形）
    for (const auto& edge : cutting_edges_) {
        // 找到这条边相邻的单元
        std::vector<openvdb::Coord> adjacent_cells = getAdjacentCells(edge.coord1, edge.coord2, edge.edge_index);
        
        // 为这条边生成三角形
        generateTrianglesForEdge(adjacent_cells, edge.edge_index);
    }
    
    if (config_.enable_debug_output) {
        std::cout << "  生成了 " << triangles_.size() << " 个原始三角形" << std::endl;
    }
}

std::vector<openvdb::Coord> EnhancedDualContouringExtractor::getAdjacentCells(
    const openvdb::Coord& c1,
    const openvdb::Coord& c2,
    int edge_index) {
    
    std::vector<openvdb::Coord> adjacent_cells;
    
    // 根据边的方向确定相邻的单元
    // 这是双重轮廓算法的核心：每条切割边对应一个四边形，由4个相邻单元的顶点组成
    
    openvdb::Vec3i edge_dir = c2 - c1;
    
    // 根据边的方向，找到相邻的4个单元
    if (edge_dir.x() != 0) {
        // X方向的边
        openvdb::Coord base = openvdb::Coord(std::min(c1.x(), c2.x()) - 1, 
                                            std::min(c1.y(), c2.y()) - 1, 
                                            std::min(c1.z(), c2.z()) - 1);
        adjacent_cells.push_back(base);
        adjacent_cells.push_back(base + openvdb::Coord(0, 1, 0));
        adjacent_cells.push_back(base + openvdb::Coord(0, 0, 1));
        adjacent_cells.push_back(base + openvdb::Coord(0, 1, 1));
    } else if (edge_dir.y() != 0) {
        // Y方向的边
        openvdb::Coord base = openvdb::Coord(std::min(c1.x(), c2.x()) - 1, 
                                            std::min(c1.y(), c2.y()) - 1, 
                                            std::min(c1.z(), c2.z()) - 1);
        adjacent_cells.push_back(base);
        adjacent_cells.push_back(base + openvdb::Coord(1, 0, 0));
        adjacent_cells.push_back(base + openvdb::Coord(0, 0, 1));
        adjacent_cells.push_back(base + openvdb::Coord(1, 0, 1));
    } else if (edge_dir.z() != 0) {
        // Z方向的边
        openvdb::Coord base = openvdb::Coord(std::min(c1.x(), c2.x()) - 1, 
                                            std::min(c1.y(), c2.y()) - 1, 
                                            std::min(c1.z(), c2.z()) - 1);
        adjacent_cells.push_back(base);
        adjacent_cells.push_back(base + openvdb::Coord(1, 0, 0));
        adjacent_cells.push_back(base + openvdb::Coord(0, 1, 0));
        adjacent_cells.push_back(base + openvdb::Coord(1, 1, 0));
    }
    
    return adjacent_cells;
}

void EnhancedDualContouringExtractor::generateTrianglesForEdge(
    const std::vector<openvdb::Coord>& adjacent_cells,
    int edge_index) {
    
    if (adjacent_cells.size() != 4) {
        return;
    }
    
    // 收集有效的顶点索引
    std::vector<int> vertex_indices;
    for (const auto& coord : adjacent_cells) {
        auto it = cells_.find(coord);
        if (it != cells_.end() && it->second.has_vertex) {
            vertex_indices.push_back(it->second.vertex_index);
        }
    }
    
    // 如果有足够的顶点，生成三角形
    if (vertex_indices.size() >= 3) {
        if (vertex_indices.size() == 4) {
            // 四边形，分解为两个三角形
            // 需要确保正确的顶点顺序以保证法向量方向
            addTriangleThreadSafe({vertex_indices[0], vertex_indices[1], vertex_indices[2]});
            addTriangleThreadSafe({vertex_indices[0], vertex_indices[2], vertex_indices[3]});
        } else if (vertex_indices.size() == 3) {
            // 三角形
            addTriangleThreadSafe({vertex_indices[0], vertex_indices[1], vertex_indices[2]});
        }
    }
}


void EnhancedDualContouringExtractor::performQualityControl() {
    if (config_.remove_degenerate) {
        removeDegenerateTriangles();
    }
    
    if (config_.validate_topology) {
        if (!validateMeshTopology()) {
            if (config_.enable_debug_output) {
                std::cout << "  检测到拓扑错误，尝试修复..." << std::endl;
            }
            fixTopologyErrors();
        }
    }
}

void EnhancedDualContouringExtractor::removeDegenerateTriangles() {
    std::vector<std::array<int, 3>> valid_triangles;
    std::vector<TriangleQuality> valid_qualities;
    
    valid_triangles.reserve(triangles_.size());
    valid_qualities.reserve(triangles_.size());
    
    int removed_count = 0;
    
    for (const auto& triangle : triangles_) {
        TriangleQuality quality = computeTriangleQuality(triangle);
        
        if (!quality.is_degenerate) {
            valid_triangles.push_back(triangle);
            valid_qualities.push_back(quality);
        } else {
            removed_count++;
        }
    }
    
    triangles_ = std::move(valid_triangles);
    triangle_qualities_ = std::move(valid_qualities);
    stats_.degenerate_removed = removed_count;
    
    if (config_.enable_debug_output) {
        std::cout << "  移除了 " << removed_count << " 个退化三角形" << std::endl;
    }
}

TriangleQuality EnhancedDualContouringExtractor::computeTriangleQuality(
    const std::array<int, 3>& triangle) {
    
    TriangleQuality quality;
    
    if (triangle[0] >= static_cast<int>(vertices_.size()) || 
        triangle[1] >= static_cast<int>(vertices_.size()) || 
        triangle[2] >= static_cast<int>(vertices_.size())) {
        quality.is_degenerate = true;
        return quality;
    }
    
    const openvdb::Vec3f& v1 = vertices_[triangle[0]];
    const openvdb::Vec3f& v2 = vertices_[triangle[1]];
    const openvdb::Vec3f& v3 = vertices_[triangle[2]];
    
    // 计算面积
    quality.area = enhanced_dual_contouring_utils::computeTriangleArea(v1, v2, v3);
    
    // 检查面积
    if (quality.area < config_.min_triangle_area) {
        quality.is_degenerate = true;
        return quality;
    }
    
    // 计算边长
    float edge1 = computeDistance(v1, v2);
    float edge2 = computeDistance(v2, v3);
    float edge3 = computeDistance(v3, v1);
    
    // 检查边长
    if (edge1 < config_.min_edge_length || edge2 < config_.min_edge_length || edge3 < config_.min_edge_length ||
        edge1 > config_.max_edge_length || edge2 > config_.max_edge_length || edge3 > config_.max_edge_length) {
        quality.is_degenerate = true;
        return quality;
    }
    
    // 计算长宽比
    quality.aspect_ratio = enhanced_dual_contouring_utils::computeTriangleAspectRatio(v1, v2, v3);
    
    // 计算角度
    auto angles = enhanced_dual_contouring_utils::computeTriangleAngles(v1, v2, v3);
    quality.min_angle = *std::min_element(angles.begin(), angles.end());
    quality.max_angle = *std::max_element(angles.begin(), angles.end());
    
    // 检查角度
    if (quality.min_angle < 5.0f || quality.max_angle > 175.0f) {
        quality.is_degenerate = true;
        return quality;
    }
    
    quality.is_degenerate = false;
    return quality;
}

bool EnhancedDualContouringExtractor::validateMeshTopology() {
    // 检查流形性质
    std::unordered_map<std::pair<int, int>, int> edge_count;
    
    for (const auto& triangle : triangles_) {
        for (int i = 0; i < 3; ++i) {
            int v1 = triangle[i];
            int v2 = triangle[(i + 1) % 3];
            
            if (v1 > v2) std::swap(v1, v2);  // 确保顺序一致
            
            edge_count[{v1, v2}]++;
        }
    }
    
    // 检查边的使用次数
    int non_manifold_edges = 0;
    for (const auto& [edge, count] : edge_count) {
        if (count > 2) {
            non_manifold_edges++;
        }
    }
    
    stats_.topology_errors = non_manifold_edges;
    
    return non_manifold_edges == 0;
}

void EnhancedDualContouringExtractor::fixTopologyErrors() {
    // 简单的拓扑修复：移除非流形边
    std::unordered_map<std::pair<int, int>, int> edge_count;
    std::unordered_set<std::pair<int, int>> problematic_edges;
    
    // 统计边的使用次数
    for (const auto& triangle : triangles_) {
        for (int i = 0; i < 3; ++i) {
            int v1 = triangle[i];
            int v2 = triangle[(i + 1) % 3];
            
            if (v1 > v2) std::swap(v1, v2);
            
            edge_count[{v1, v2}]++;
        }
    }
    
    // 找到问题边
    for (const auto& [edge, count] : edge_count) {
        if (count > 2) {
            problematic_edges.insert(edge);
        }
    }
    
    // 移除包含问题边的三角形
    std::vector<std::array<int, 3>> valid_triangles;
    valid_triangles.reserve(triangles_.size());
    
    for (const auto& triangle : triangles_) {
        bool has_problematic_edge = false;
        
        for (int i = 0; i < 3; ++i) {
            int v1 = triangle[i];
            int v2 = triangle[(i + 1) % 3];
            
            if (v1 > v2) std::swap(v1, v2);
            
            if (problematic_edges.find({v1, v2}) != problematic_edges.end()) {
                has_problematic_edge = true;
                break;
            }
        }
        
        if (!has_problematic_edge) {
            valid_triangles.push_back(triangle);
        }
    }
    
    int removed_count = static_cast<int>(triangles_.size() - valid_triangles.size());
    triangles_ = std::move(valid_triangles);
    
    if (config_.enable_debug_output) {
        std::cout << "  修复拓扑错误，移除了 " << removed_count << " 个三角形" << std::endl;
    }
}

void EnhancedDualContouringExtractor::buildPCLMesh(pcl::PolygonMesh& mesh) {
    // 构建点云
    pcl::PointCloud<pcl::PointXYZ> cloud;
    cloud.reserve(vertices_.size());
    
    for (const auto& vertex : vertices_) {
        cloud.emplace_back(vertex.x(), vertex.y(), vertex.z());
    }
    
    pcl::toPCLPointCloud2(cloud, mesh.cloud);
    
    // 构建多边形
    mesh.polygons.clear();
    mesh.polygons.reserve(triangles_.size());
    
    for (const auto& triangle : triangles_) {
        pcl::Vertices vertices;
        vertices.vertices = {static_cast<uint32_t>(triangle[0]), 
                           static_cast<uint32_t>(triangle[1]), 
                           static_cast<uint32_t>(triangle[2])};
        mesh.polygons.push_back(vertices);
    }
}

void EnhancedDualContouringExtractor::updateStatistics() {
    stats_.vertices_generated = static_cast<int>(vertices_.size());
    stats_.triangles_generated = static_cast<int>(triangles_.size());
    
    // 计算平均QEF误差
    if (!cells_.empty()) {
        float total_qef_error = 0.0f;
        float total_confidence = 0.0f;
        
        for (const auto& [coord, cell] : cells_) {
            total_qef_error += cell.qef_error;
            total_confidence += cell.vertex_confidence;
        }
        
        stats_.average_qef_error = total_qef_error / cells_.size();
        stats_.average_vertex_confidence = total_confidence / cells_.size();
    }
    
    // 计算网格质量评分
    if (!triangle_qualities_.empty()) {
        float total_quality = 0.0f;
        
        for (const auto& quality : triangle_qualities_) {
            // 基于面积、长宽比和角度的综合评分
            float area_score = std::min(1.0f, quality.area / config_.min_triangle_area);
            float aspect_score = 1.0f / (1.0f + quality.aspect_ratio);
            float angle_score = (quality.min_angle > 30.0f && quality.max_angle < 120.0f) ? 1.0f : 0.5f;
            
            total_quality += (area_score + aspect_score + angle_score) / 3.0f;
        }
        
        stats_.mesh_quality_score = total_quality / triangle_qualities_.size();
    }
}

// 工具函数实现
float EnhancedDualContouringExtractor::getUDFValue(const openvdb::Vec3f& position, 
                                                   const UDFGridT::Ptr& udf_grid) {
    openvdb::Vec3d index_pos = udf_grid->transform().worldToIndex(position);
    return udf_grid->tree().getValue(openvdb::Coord::round(index_pos));
}

bool EnhancedDualContouringExtractor::isEdgeCut(const openvdb::Coord& coord1, 
                                                const openvdb::Coord& coord2,
                                                const UDFGridT::Ptr& udf_grid) {
    openvdb::Vec3f pos1(coord1.x(), coord1.y(), coord1.z());
    openvdb::Vec3f pos2(coord2.x(), coord2.y(), coord2.z());
    
    float val1 = getUDFValue(pos1, udf_grid);
    float val2 = getUDFValue(pos2, udf_grid);
    
    return (val1 > 0 && val2 < 0) || (val1 < 0 && val2 > 0);
}

float EnhancedDualContouringExtractor::computeIntersectionParameter(float val1, float val2) {
    if (std::abs(val1 - val2) < 1e-6f) {
        return 0.5f;
    }
    return val1 / (val1 - val2);
}

std::pair<openvdb::Coord, openvdb::Coord> EnhancedDualContouringExtractor::getEdgeCoords(
    const openvdb::Coord& base_coord, int edge_index) {
    
    if (edge_index < 0 || edge_index >= 12) {
        return {base_coord, base_coord};
    }
    
    const auto& edge = CUBE_EDGES[edge_index];
    openvdb::Coord coord1 = base_coord + CUBE_VERTICES[edge.first];
    openvdb::Coord coord2 = base_coord + CUBE_VERTICES[edge.second];
    
    return {coord1, coord2};
}

float EnhancedDualContouringExtractor::computeDistance(const openvdb::Vec3f& p1, 
                                                      const openvdb::Vec3f& p2) {
    openvdb::Vec3f diff = p2 - p1;
    return diff.length();
}

int EnhancedDualContouringExtractor::addVertexThreadSafe(const openvdb::Vec3f& vertex, 
                                                         const openvdb::Vec3f& normal) {
    std::lock_guard<std::mutex> lock(vertices_mutex_);
    int index = static_cast<int>(vertices_.size());
    vertices_.push_back(vertex);
    normals_.push_back(normal);
    return index;
}

void EnhancedDualContouringExtractor::addTriangleThreadSafe(const std::array<int, 3>& triangle) {
    std::lock_guard<std::mutex> lock(triangles_mutex_);
    triangles_.push_back(triangle);
}

// ============================================================================
// 工具函数命名空间实现
// ============================================================================

namespace enhanced_dual_contouring_utils {

float computeTriangleArea(const openvdb::Vec3f& v1,
                         const openvdb::Vec3f& v2,
                         const openvdb::Vec3f& v3) {
    openvdb::Vec3f edge1 = v2 - v1;
    openvdb::Vec3f edge2 = v3 - v1;
    openvdb::Vec3f cross = crossProduct(edge1, edge2);
    return 0.5f * length(cross);
}

openvdb::Vec3f computeTriangleNormal(const openvdb::Vec3f& v1,
                                    const openvdb::Vec3f& v2,
                                    const openvdb::Vec3f& v3) {
    openvdb::Vec3f edge1 = v2 - v1;
    openvdb::Vec3f edge2 = v3 - v1;
    return normalize(crossProduct(edge1, edge2));
}

float computeTriangleAspectRatio(const openvdb::Vec3f& v1,
                                const openvdb::Vec3f& v2,
                                const openvdb::Vec3f& v3) {
    float edge1 = length(v2 - v1);
    float edge2 = length(v3 - v2);
    float edge3 = length(v1 - v3);
    
    float max_edge = std::max({edge1, edge2, edge3});
    float min_edge = std::min({edge1, edge2, edge3});
    
    return (min_edge > 1e-6f) ? (max_edge / min_edge) : 1e6f;
}

std::array<float, 3> computeTriangleAngles(const openvdb::Vec3f& v1,
                                          const openvdb::Vec3f& v2,
                                          const openvdb::Vec3f& v3) {
    openvdb::Vec3f e1 = normalize(v2 - v1);
    openvdb::Vec3f e2 = normalize(v3 - v1);
    openvdb::Vec3f e3 = normalize(v1 - v2);
    openvdb::Vec3f e4 = normalize(v3 - v2);
    openvdb::Vec3f e5 = normalize(v1 - v3);
    openvdb::Vec3f e6 = normalize(v2 - v3);
    
    float angle1 = computeAngle(e1, e2) * 180.0f / M_PI;
    float angle2 = computeAngle(e3, e4) * 180.0f / M_PI;
    float angle3 = computeAngle(e5, e6) * 180.0f / M_PI;
    
    return {angle1, angle2, angle3};
}

float computeAngle(const openvdb::Vec3f& v1, const openvdb::Vec3f& v2) {
    float dot_product = v1.dot(v2);
    dot_product = std::max(-1.0f, std::min(1.0f, dot_product));
    return std::acos(dot_product);
}

bool isPointInBounds(const openvdb::Vec3f& point,
                    const openvdb::Vec3f& min_bound,
                    const openvdb::Vec3f& max_bound) {
    return (point.x() >= min_bound.x() && point.x() <= max_bound.x() &&
            point.y() >= min_bound.y() && point.y() <= max_bound.y() &&
            point.z() >= min_bound.z() && point.z() <= max_bound.z());
}

float pointToPlaneDistance(const openvdb::Vec3f& point,
                          const openvdb::Vec3f& plane_point,
                          const openvdb::Vec3f& plane_normal) {
    openvdb::Vec3f norm_normal = normalize(plane_normal);
    return std::abs(norm_normal.dot(point - plane_point));
}

openvdb::Vec3f crossProduct(const openvdb::Vec3f& v1, const openvdb::Vec3f& v2) {
    return openvdb::Vec3f(
        v1.y() * v2.z() - v1.z() * v2.y(),
        v1.z() * v2.x() - v1.x() * v2.z(),
        v1.x() * v2.y() - v1.y() * v2.x()
    );
}

openvdb::Vec3f normalize(const openvdb::Vec3f& v) {
    float len = length(v);
    return (len > 1e-6f) ? (v / len) : openvdb::Vec3f(0.0f, 0.0f, 1.0f);
}

float length(const openvdb::Vec3f& v) {
    return std::sqrt(v.x() * v.x() + v.y() * v.y() + v.z() * v.z());
}

} // namespace enhanced_dual_contouring_utils

} // namespace recon

