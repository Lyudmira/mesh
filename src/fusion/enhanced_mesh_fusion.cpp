/**
 * 增强版网格融合与后处理模块实现
 * 实现细节保留优先级规则、颜色混合、顶点焊接、alpha包装与平面对齐
 */

#include "enhanced_mesh_fusion.h"
#include <pcl/common/common.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/surface/convex_hull.h>
#include <pcl/surface/concave_hull.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <chrono>
#include <queue>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace recon {

// ============================================================================
// DetailPreservationRule 实现
// ============================================================================

bool DetailPreservationRule::shouldPreserve(const FusionConfig& config) const {
    // 关键细节必须保留
    if (priority == Priority::CRITICAL) {
        return true;
    }
    
    // 基于距离阈值判断
    if (shell_distance > config.shell_distance_threshold) {
        return false;
    }
    
    // 基于重要性评分判断
    float threshold = 0.5f;
    switch (priority) {
        case Priority::HIGH:
            threshold = 0.3f;
            break;
        case Priority::MEDIUM:
            threshold = 0.5f;
            break;
        case Priority::LOW:
            threshold = 0.7f;
            break;
        case Priority::NEGLIGIBLE:
            threshold = 0.9f;
            break;
        default:
            break;
    }
    
    return importance_score > threshold;
}

// ============================================================================
// EnhancedMeshFusion 实现
// ============================================================================

EnhancedMeshFusion::EnhancedMeshFusion(const FusionConfig& config) 
    : config_(config) {
    
    spatial_hash_ = std::make_unique<SpatialHashGrid>(config_.vertex_welding_threshold);
    
    #ifdef _OPENMP
    if (config_.use_parallel_processing && config_.num_threads > 0) {
        omp_set_num_threads(config_.num_threads);
    }
    #endif
}

EnhancedMeshFusion::~EnhancedMeshFusion() = default;

bool EnhancedMeshFusion::fuseMeshes(const std::vector<pcl::PolygonMesh>& input_meshes,
                                   const PointCloudT::Ptr& original_cloud,
                                   pcl::PolygonMesh& output_mesh) {
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    std::cout << "开始增强网格融合..." << std::endl;
    std::cout << "输入网格数量: " << input_meshes.size() << std::endl;
    
    if (input_meshes.empty()) {
        std::cerr << "错误: 没有输入网格" << std::endl;
        return false;
    }
    
    // 重置统计信息
    stats_ = Statistics{};
    stats_.input_meshes = static_cast<int>(input_meshes.size());
    
    // 1. 预处理输入网格
    std::cout << "预处理输入网格..." << std::endl;
    if (!preprocessInputMeshes(input_meshes, original_cloud)) {
        std::cerr << "预处理失败" << std::endl;
        return false;
    }
    
    // 2. 应用细节保留规则
    std::cout << "应用细节保留规则..." << std::endl;
    applyDetailPreservationRules();
    
    // 3. 执行颜色混合
    if (config_.enable_color_blending) {
        std::cout << "执行颜色混合..." << std::endl;
        performColorBlending(original_cloud);
    }
    
    // 4. 执行顶点焊接
    if (config_.enable_vertex_welding) {
        std::cout << "执行顶点焊接..." << std::endl;
        performVertexWelding();
    }
    
    // 5. 执行第二次alpha包装
    if (config_.enable_second_alpha_wrapping) {
        std::cout << "执行第二次alpha包装..." << std::endl;
        if (!performSecondAlphaWrapping()) {
            std::cout << "Alpha包装失败，跳过此步骤" << std::endl;
        }
    }
    
    // 6. 执行平面对齐
    if (config_.enable_plane_alignment) {
        std::cout << "执行平面对齐..." << std::endl;
        performPlaneAlignment();
    }
    
    // 7. 修复网格拓扑
    if (config_.enable_topology_repair) {
        std::cout << "修复网格拓扑..." << std::endl;
        repairMeshTopology();
    }
    
    // 8. 构建最终网格
    std::cout << "构建最终网格..." << std::endl;
    buildFinalMesh(output_mesh);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    stats_.fusion_time_seconds = std::chrono::duration<double>(end_time - start_time).count();
    
    std::cout << "网格融合完成:" << std::endl;
    std::cout << "  输出顶点数: " << stats_.output_vertices << std::endl;
    std::cout << "  输出面片数: " << stats_.output_faces << std::endl;
    std::cout << "  焊接顶点数: " << stats_.vertices_welded << std::endl;
    std::cout << "  处理时间: " << stats_.fusion_time_seconds << " 秒" << std::endl;
    
    return true;
}

bool EnhancedMeshFusion::preprocessInputMeshes(const std::vector<pcl::PolygonMesh>& input_meshes,
                                              const PointCloudT::Ptr& original_cloud) {
    
    vertices_.clear();
    faces_.clear();
    spatial_hash_->clear();
    
    int vertex_offset = 0;
    
    // 处理每个输入网格
    for (size_t mesh_idx = 0; mesh_idx < input_meshes.size(); ++mesh_idx) {
        const auto& mesh = input_meshes[mesh_idx];
        
        // 转换顶点
        pcl::PointCloud<pcl::PointXYZ> cloud;
        pcl::fromPCLPointCloud2(mesh.cloud, cloud);
        
        for (size_t i = 0; i < cloud.size(); ++i) {
            VertexInfo vertex;
            vertex.position = Eigen::Vector3f(cloud.points[i].x, cloud.points[i].y, cloud.points[i].z);
            vertex.original_index = static_cast<int>(i);
            vertex.confidence = 1.0f;
            
            // 计算细节保留规则
            vertex.preservation_rule = computePreservationRule(vertex, *original_cloud);
            
            vertices_.push_back(vertex);
            spatial_hash_->addVertex(static_cast<int>(vertices_.size() - 1), vertex.position);
        }
        
        // 转换面片
        for (const auto& polygon : mesh.polygons) {
            if (polygon.vertices.size() == 3) {
                FaceInfo face;
                face.vertex_indices[0] = polygon.vertices[0] + vertex_offset;
                face.vertex_indices[1] = polygon.vertices[1] + vertex_offset;
                face.vertex_indices[2] = polygon.vertices[2] + vertex_offset;
                
                // 计算面片信息
                const auto& v1 = vertices_[face.vertex_indices[0]].position;
                const auto& v2 = vertices_[face.vertex_indices[1]].position;
                const auto& v3 = vertices_[face.vertex_indices[2]].position;
                
                face.normal = fusion_utils::computeTriangleNormal(v1, v2, v3);
                face.area = fusion_utils::computeTriangleArea(v1, v2, v3);
                face.quality_score = fusion_utils::computeTriangleQuality(v1, v2, v3);
                face.is_degenerate = fusion_utils::isTriangleDegenerate(v1, v2, v3);
                face.centroid = (v1 + v2 + v3) / 3.0f;
                
                if (!face.is_degenerate) {
                    faces_.push_back(face);
                }
            }
        }
        
        vertex_offset += static_cast<int>(cloud.size());
        stats_.total_input_vertices += static_cast<int>(cloud.size());
        stats_.total_input_faces += static_cast<int>(mesh.polygons.size());
    }
    
    std::cout << "预处理完成: " << vertices_.size() << " 顶点, " << faces_.size() << " 面片" << std::endl;
    
    return !vertices_.empty() && !faces_.empty();
}

void EnhancedMeshFusion::applyDetailPreservationRules() {
    int preserved_vertices = 0;
    int removed_vertices = 0;
    
    // 标记要保留的顶点
    for (auto& vertex : vertices_) {
        if (vertex.preservation_rule.shouldPreserve(config_)) {
            preserved_vertices++;
        } else {
            vertex.confidence = 0.0f;  // 标记为移除
            removed_vertices++;
        }
    }
    
    // 移除低置信度的面片
    auto face_it = faces_.begin();
    while (face_it != faces_.end()) {
        bool should_remove = false;
        
        for (int i = 0; i < 3; ++i) {
            if (vertices_[face_it->vertex_indices[i]].confidence < 0.1f) {
                should_remove = true;
                break;
            }
        }
        
        if (should_remove) {
            face_it = faces_.erase(face_it);
            stats_.faces_removed++;
        } else {
            ++face_it;
        }
    }
    
    std::cout << "细节保留规则应用完成:" << std::endl;
    std::cout << "  保留顶点: " << preserved_vertices << std::endl;
    std::cout << "  移除顶点: " << removed_vertices << std::endl;
    std::cout << "  移除面片: " << stats_.faces_removed << std::endl;
}

DetailPreservationRule EnhancedMeshFusion::computePreservationRule(const VertexInfo& vertex,
                                                                   const PointCloudT& original_cloud) {
    DetailPreservationRule rule;
    
    // 计算到外壳的距离（简化实现）
    rule.shell_distance = 0.05f;  // 默认5cm
    
    // 计算重要性评分
    rule.importance_score = 0.8f;  // 默认高重要性
    
    // 判断是否为特征
    rule.is_feature = false;
    rule.is_boundary = false;
    
    // 基于距离和重要性确定优先级
    if (rule.shell_distance < config_.detail_preservation_threshold) {
        if (rule.importance_score > 0.8f) {
            rule.priority = DetailPreservationRule::Priority::CRITICAL;
        } else if (rule.importance_score > 0.6f) {
            rule.priority = DetailPreservationRule::Priority::HIGH;
        } else {
            rule.priority = DetailPreservationRule::Priority::MEDIUM;
        }
    } else {
        rule.priority = DetailPreservationRule::Priority::LOW;
    }
    
    rule.description = "基于距离和重要性的保留规则";
    
    return rule;
}

void EnhancedMeshFusion::performColorBlending(const PointCloudT::Ptr& original_cloud) {
    if (!original_cloud || original_cloud->empty()) {
        std::cout << "没有原始点云，跳过颜色混合" << std::endl;
        return;
    }
    
    ColorBlender blender(config_);
    blender.blendMeshColors(vertices_, original_cloud);
    
    std::cout << "颜色混合完成" << std::endl;
}

void EnhancedMeshFusion::performVertexWelding() {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    std::vector<std::vector<int>> welding_groups;
    std::vector<bool> processed(vertices_.size(), false);
    
    // 查找可焊接的顶点组
    for (size_t i = 0; i < vertices_.size(); ++i) {
        if (processed[i] || vertices_[i].confidence < 0.1f) {
            continue;
        }
        
        std::vector<int> group = findWeldableVertices(static_cast<int>(i));
        
        if (group.size() > 1) {
            welding_groups.push_back(group);
            
            for (int idx : group) {
                processed[idx] = true;
            }
        }
    }
    
    // 执行焊接
    for (const auto& group : welding_groups) {
        weldVertexGroup(group);
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    stats_.welding_time_seconds = std::chrono::duration<double>(end_time - start_time).count();
    stats_.vertices_welded = static_cast<int>(welding_groups.size());
    
    std::cout << "顶点焊接完成: " << welding_groups.size() << " 个焊接组" << std::endl;
}

std::vector<int> EnhancedMeshFusion::findWeldableVertices(int vertex_index) {
    std::vector<int> weldable_vertices;
    weldable_vertices.push_back(vertex_index);
    
    const auto& vertex = vertices_[vertex_index];
    
    // 查找邻近顶点
    std::vector<int> nearby_vertices = spatial_hash_->findNearbyVertices(
        vertex.position, config_.vertex_welding_threshold);
    
    for (int nearby_idx : nearby_vertices) {
        if (nearby_idx == vertex_index) continue;
        
        const auto& nearby_vertex = vertices_[nearby_idx];
        
        // 检查距离
        float distance = fusion_utils::distance(vertex.position, nearby_vertex.position);
        if (distance > config_.vertex_welding_threshold) {
            continue;
        }
        
        // 检查法向量一致性
        float normal_dot = vertex.normal.dot(nearby_vertex.normal);
        if (normal_dot < config_.normal_welding_threshold) {
            continue;
        }
        
        weldable_vertices.push_back(nearby_idx);
    }
    
    return weldable_vertices;
}

void EnhancedMeshFusion::weldVertexGroup(const std::vector<int>& vertex_indices) {
    if (vertex_indices.size() < 2) {
        return;
    }
    
    // 计算焊接后的位置和法向量
    Eigen::Vector3f avg_position(0, 0, 0);
    Eigen::Vector3f avg_normal(0, 0, 0);
    Eigen::Vector3f avg_color(0, 0, 0);
    float total_confidence = 0.0f;
    
    for (int idx : vertex_indices) {
        const auto& vertex = vertices_[idx];
        float weight = vertex.confidence;
        
        avg_position += vertex.position * weight;
        avg_normal += vertex.normal * weight;
        avg_color += vertex.color * weight;
        total_confidence += weight;
    }
    
    if (total_confidence > 0.0f) {
        avg_position /= total_confidence;
        avg_normal /= total_confidence;
        avg_color /= total_confidence;
        avg_normal.normalize();
    }
    
    // 更新第一个顶点，标记其他顶点为已焊接
    int master_idx = vertex_indices[0];
    vertices_[master_idx].position = avg_position;
    vertices_[master_idx].normal = avg_normal;
    vertices_[master_idx].color = avg_color;
    vertices_[master_idx].confidence = total_confidence / vertex_indices.size();
    
    for (size_t i = 1; i < vertex_indices.size(); ++i) {
        int slave_idx = vertex_indices[i];
        vertices_[slave_idx].is_welded = true;
        vertices_[slave_idx].welded_group_id = master_idx;
    }
    
    // 更新面片的顶点索引
    for (auto& face : faces_) {
        for (int i = 0; i < 3; ++i) {
            int& vertex_idx = face.vertex_indices[i];
            
            // 如果顶点被焊接，替换为主顶点
            for (size_t j = 1; j < vertex_indices.size(); ++j) {
                if (vertex_idx == vertex_indices[j]) {
                    vertex_idx = master_idx;
                    break;
                }
            }
        }
    }
}

bool EnhancedMeshFusion::performSecondAlphaWrapping() {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // 构建临时网格
    pcl::PolygonMesh temp_mesh;
    buildFinalMesh(temp_mesh);
    
    // 使用Alpha包装器
    AlphaWrapper::AlphaConfig alpha_config;
    alpha_config.alpha_value = config_.alpha_value;
    alpha_config.use_adaptive_alpha = config_.use_adaptive_alpha;
    
    AlphaWrapper wrapper(alpha_config);
    
    pcl::PolygonMesh wrapped_mesh;
    bool success = wrapper.wrapMesh(temp_mesh, wrapped_mesh);
    
    if (success) {
        // 更新顶点和面片信息
        vertices_.clear();
        faces_.clear();
        
        pcl::PointCloud<pcl::PointXYZ> cloud;
        pcl::fromPCLPointCloud2(wrapped_mesh.cloud, cloud);
        
        for (size_t i = 0; i < cloud.size(); ++i) {
            VertexInfo vertex;
            vertex.position = Eigen::Vector3f(cloud.points[i].x, cloud.points[i].y, cloud.points[i].z);
            vertex.original_index = static_cast<int>(i);
            vertices_.push_back(vertex);
        }
        
        for (const auto& polygon : wrapped_mesh.polygons) {
            if (polygon.vertices.size() == 3) {
                FaceInfo face;
                face.vertex_indices[0] = polygon.vertices[0];
                face.vertex_indices[1] = polygon.vertices[1];
                face.vertex_indices[2] = polygon.vertices[2];
                
                const auto& v1 = vertices_[face.vertex_indices[0]].position;
                const auto& v2 = vertices_[face.vertex_indices[1]].position;
                const auto& v3 = vertices_[face.vertex_indices[2]].position;
                
                face.normal = fusion_utils::computeTriangleNormal(v1, v2, v3);
                face.area = fusion_utils::computeTriangleArea(v1, v2, v3);
                face.quality_score = fusion_utils::computeTriangleQuality(v1, v2, v3);
                face.is_degenerate = fusion_utils::isTriangleDegenerate(v1, v2, v3);
                
                if (!face.is_degenerate) {
                    faces_.push_back(face);
                }
            }
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    stats_.alpha_wrapping_time_seconds = std::chrono::duration<double>(end_time - start_time).count();
    
    return success;
}

void EnhancedMeshFusion::performPlaneAlignment() {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    PlaneAligner aligner(config_);
    
    // 检测主要平面
    detected_planes_ = aligner.detectMajorPlanes(vertices_);
    stats_.planes_detected = static_cast<int>(detected_planes_.size());
    
    std::cout << "检测到 " << detected_planes_.size() << " 个主要平面" << std::endl;
    
    // 对齐顶点到平面
    aligner.alignVertices(vertices_, detected_planes_);
    
    // 确保墙壁直线
    std::vector<PlaneInfo> wall_planes;
    for (const auto& plane : detected_planes_) {
        if (plane.type == PlaneInfo::Type::WALL) {
            wall_planes.push_back(plane);
        }
    }
    
    if (!wall_planes.empty()) {
        aligner.ensureWallStraightness(vertices_, wall_planes);
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    stats_.plane_alignment_time_seconds = std::chrono::duration<double>(end_time - start_time).count();
    
    std::cout << "平面对齐完成" << std::endl;
}

void EnhancedMeshFusion::repairMeshTopology() {
    TopologyRepairer repairer(config_);
    
    auto repair_result = repairer.repairTopology(vertices_, faces_);
    
    stats_.topology_fixes = static_cast<int>(repair_result.fixed_issues.size());
    
    std::cout << "拓扑修复完成:" << std::endl;
    std::cout << "  修复问题数: " << repair_result.fixed_issues.size() << std::endl;
    std::cout << "  移除顶点数: " << repair_result.vertices_removed << std::endl;
    std::cout << "  移除面片数: " << repair_result.faces_removed << std::endl;
}

void EnhancedMeshFusion::buildFinalMesh(pcl::PolygonMesh& output_mesh) {
    // 收集有效顶点
    std::vector<int> vertex_mapping(vertices_.size(), -1);
    pcl::PointCloud<pcl::PointXYZRGBNormal> final_cloud;
    
    int valid_vertex_count = 0;
    for (size_t i = 0; i < vertices_.size(); ++i) {
        const auto& vertex = vertices_[i];
        
        if (vertex.confidence > 0.1f && !vertex.is_welded) {
            pcl::PointXYZRGBNormal pt;
            pt.x = vertex.position.x();
            pt.y = vertex.position.y();
            pt.z = vertex.position.z();
            pt.normal_x = vertex.normal.x();
            pt.normal_y = vertex.normal.y();
            pt.normal_z = vertex.normal.z();
            
            // 设置颜色
            pt.r = static_cast<uint8_t>(vertex.color.x() * 255);
            pt.g = static_cast<uint8_t>(vertex.color.y() * 255);
            pt.b = static_cast<uint8_t>(vertex.color.z() * 255);
            
            final_cloud.push_back(pt);
            vertex_mapping[i] = valid_vertex_count++;
        }
    }
    
    // 收集有效面片
    std::vector<pcl::Vertices> final_polygons;
    for (const auto& face : faces_) {
        if (face.is_degenerate) continue;
        
        pcl::Vertices polygon;
        bool valid_face = true;
        
        for (int i = 0; i < 3; ++i) {
            int original_idx = face.vertex_indices[i];
            int mapped_idx = vertex_mapping[original_idx];
            
            if (mapped_idx < 0) {
                valid_face = false;
                break;
            }
            
            polygon.vertices.push_back(mapped_idx);
        }
        
        if (valid_face) {
            final_polygons.push_back(polygon);
        }
    }
    
    // 构建最终网格
    pcl::toPCLPointCloud2(final_cloud, output_mesh.cloud);
    output_mesh.polygons = final_polygons;
    
    stats_.output_vertices = static_cast<int>(final_cloud.size());
    stats_.output_faces = static_cast<int>(final_polygons.size());
    
    std::cout << "最终网格构建完成: " << stats_.output_vertices 
              << " 顶点, " << stats_.output_faces << " 面片" << std::endl;
}

// ============================================================================
// SpatialHashGrid 实现
// ============================================================================

SpatialHashGrid::SpatialHashGrid(float cell_size) : cell_size_(cell_size) {
}

void SpatialHashGrid::addVertex(int vertex_id, const Eigen::Vector3f& position) {
    int64_t hash_key = computeHashKey(position);
    hash_grid_[hash_key].push_back(vertex_id);
}

std::vector<int> SpatialHashGrid::findNearbyVertices(const Eigen::Vector3f& position, float radius) {
    std::vector<int> nearby_vertices;
    
    // 计算需要查询的单元
    int radius_cells = static_cast<int>(std::ceil(radius / cell_size_));
    
    for (int dx = -radius_cells; dx <= radius_cells; ++dx) {
        for (int dy = -radius_cells; dy <= radius_cells; ++dy) {
            for (int dz = -radius_cells; dz <= radius_cells; ++dz) {
                Eigen::Vector3f offset_pos = position + Eigen::Vector3f(dx, dy, dz) * cell_size_;
                int64_t hash_key = computeHashKey(offset_pos);
                
                auto it = hash_grid_.find(hash_key);
                if (it != hash_grid_.end()) {
                    nearby_vertices.insert(nearby_vertices.end(), 
                                         it->second.begin(), it->second.end());
                }
            }
        }
    }
    
    return nearby_vertices;
}

void SpatialHashGrid::clear() {
    hash_grid_.clear();
}

int64_t SpatialHashGrid::computeHashKey(const Eigen::Vector3f& position) {
    int x = static_cast<int>(std::floor(position.x() / cell_size_));
    int y = static_cast<int>(std::floor(position.y() / cell_size_));
    int z = static_cast<int>(std::floor(position.z() / cell_size_));
    
    // 简单的哈希函数
    return static_cast<int64_t>(x) * 73856093LL + 
           static_cast<int64_t>(y) * 19349663LL + 
           static_cast<int64_t>(z) * 83492791LL;
}

// ============================================================================
// AlphaWrapper 实现
// ============================================================================

AlphaWrapper::AlphaWrapper(const AlphaConfig& config) : config_(config) {
}

bool AlphaWrapper::wrapMesh(const pcl::PolygonMesh& input_mesh,
                           pcl::PolygonMesh& output_mesh) {
    
    if (config_.use_adaptive_alpha) {
        return adaptiveWrap(input_mesh, output_mesh);
    }
    
    // 简化的alpha包装实现
    // 实际应该使用CGAL的Alpha_wrap_3
    
    // 暂时直接复制输入网格
    output_mesh = input_mesh;
    
    return validateWrappingResult(output_mesh);
}

bool AlphaWrapper::adaptiveWrap(const pcl::PolygonMesh& input_mesh,
                               pcl::PolygonMesh& output_mesh) {
    
    pcl::PointCloud<pcl::PointXYZ> cloud;
    pcl::fromPCLPointCloud2(input_mesh.cloud, cloud);
    
    // 计算自适应alpha值
    float adaptive_alpha = computeAdaptiveAlpha(cloud);
    adaptive_alpha = std::max(config_.min_alpha, std::min(config_.max_alpha, adaptive_alpha));
    
    std::cout << "使用自适应alpha值: " << adaptive_alpha << std::endl;
    
    // 执行包装
    return wrapMesh(input_mesh, output_mesh);
}

float AlphaWrapper::computeAdaptiveAlpha(const pcl::PointCloud<pcl::PointXYZ>& cloud) {
    if (cloud.empty()) {
        return config_.alpha_value;
    }
    
    // 计算平均最近邻距离
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_ptr(new pcl::PointCloud<pcl::PointXYZ>(cloud));
    kdtree.setInputCloud(cloud_ptr);
    
    float total_distance = 0.0f;
    int valid_points = 0;
    
    for (const auto& point : cloud.points) {
        std::vector<int> indices(2);
        std::vector<float> distances(2);
        
        if (kdtree.nearestKSearch(point, 2, indices, distances) >= 2) {
            total_distance += std::sqrt(distances[1]);  // 第二近的点（第一个是自己）
            valid_points++;
        }
    }
    
    if (valid_points > 0) {
        float avg_distance = total_distance / valid_points;
        return avg_distance * config_.alpha_scale_factor;
    }
    
    return config_.alpha_value;
}

bool AlphaWrapper::validateWrappingResult(const pcl::PolygonMesh& mesh) {
    // 简单验证：检查是否有面片
    return !mesh.polygons.empty();
}

// ============================================================================
// PlaneAligner 实现
// ============================================================================

PlaneAligner::PlaneAligner(const FusionConfig& config) : config_(config) {
}

std::vector<PlaneInfo> PlaneAligner::detectMajorPlanes(const std::vector<VertexInfo>& vertices) {
    std::vector<PlaneInfo> planes;
    
    if (vertices.size() < 100) {
        return planes;
    }
    
    // 使用RANSAC检测平面
    std::vector<int> remaining_indices;
    for (size_t i = 0; i < vertices.size(); ++i) {
        if (vertices[i].confidence > 0.5f) {
            remaining_indices.push_back(static_cast<int>(i));
        }
    }
    
    // 检测多个平面
    for (int plane_idx = 0; plane_idx < 5 && remaining_indices.size() > 50; ++plane_idx) {
        PlaneInfo plane = detectPlaneRANSAC(vertices, remaining_indices);
        
        if (plane.confidence > 0.5f && plane.vertex_indices.size() > 20) {
            plane.type = classifyPlaneType(plane);
            plane.is_major_plane = plane.vertex_indices.size() > vertices.size() * 0.1f;
            planes.push_back(plane);
            
            // 移除已分配的点
            std::unordered_set<int> assigned_set(plane.vertex_indices.begin(), plane.vertex_indices.end());
            remaining_indices.erase(
                std::remove_if(remaining_indices.begin(), remaining_indices.end(),
                              [&assigned_set](int idx) { return assigned_set.count(idx) > 0; }),
                remaining_indices.end());
        } else {
            break;
        }
    }
    
    std::cout << "检测到 " << planes.size() << " 个主要平面" << std::endl;
    
    return planes;
}

PlaneInfo PlaneAligner::detectPlaneRANSAC(const std::vector<VertexInfo>& vertices,
                                          const std::vector<int>& candidate_indices) {
    PlaneInfo plane;
    
    if (candidate_indices.size() < 3) {
        return plane;
    }
    
    // 简化的RANSAC实现
    int max_iterations = 1000;
    int best_inlier_count = 0;
    float best_plane_coeffs[4] = {0, 0, 1, 0};  // 默认水平面
    
    for (int iter = 0; iter < max_iterations; ++iter) {
        // 随机选择3个点
        std::vector<int> sample_indices(3);
        for (int i = 0; i < 3; ++i) {
            sample_indices[i] = candidate_indices[rand() % candidate_indices.size()];
        }
        
        // 计算平面方程
        const auto& p1 = vertices[sample_indices[0]].position;
        const auto& p2 = vertices[sample_indices[1]].position;
        const auto& p3 = vertices[sample_indices[2]].position;
        
        Eigen::Vector3f v1 = p2 - p1;
        Eigen::Vector3f v2 = p3 - p1;
        Eigen::Vector3f normal = v1.cross(v2).normalized();
        
        if (normal.norm() < 0.1f) continue;  // 退化情况
        
        float d = -normal.dot(p1);
        
        // 计算内点数
        int inlier_count = 0;
        for (int idx : candidate_indices) {
            const auto& pos = vertices[idx].position;
            float distance = std::abs(normal.dot(pos) + d);
            
            if (distance < config_.plane_alignment_threshold) {
                inlier_count++;
            }
        }
        
        if (inlier_count > best_inlier_count) {
            best_inlier_count = inlier_count;
            best_plane_coeffs[0] = normal.x();
            best_plane_coeffs[1] = normal.y();
            best_plane_coeffs[2] = normal.z();
            best_plane_coeffs[3] = d;
        }
    }
    
    // 构建最佳平面信息
    if (best_inlier_count > 10) {
        plane.coefficients = Eigen::Vector4f(best_plane_coeffs[0], best_plane_coeffs[1], 
                                            best_plane_coeffs[2], best_plane_coeffs[3]);
        plane.normal = Eigen::Vector3f(best_plane_coeffs[0], best_plane_coeffs[1], best_plane_coeffs[2]);
        plane.confidence = static_cast<float>(best_inlier_count) / candidate_indices.size();
        
        // 收集内点
        for (int idx : candidate_indices) {
            const auto& pos = vertices[idx].position;
            float distance = std::abs(plane.normal.dot(pos) + best_plane_coeffs[3]);
            
            if (distance < config_.plane_alignment_threshold) {
                plane.vertex_indices.push_back(idx);
            }
        }
        
        // 计算质心
        Eigen::Vector3f centroid(0, 0, 0);
        for (int idx : plane.vertex_indices) {
            centroid += vertices[idx].position;
        }
        plane.centroid = centroid / plane.vertex_indices.size();
    }
    
    return plane;
}

PlaneInfo::Type PlaneAligner::classifyPlaneType(const PlaneInfo& plane) {
    // 基于法向量方向分类平面
    Eigen::Vector3f up(0, 0, 1);
    float dot_up = std::abs(plane.normal.dot(up));
    
    if (dot_up > 0.8f) {
        // 接近水平
        if (plane.centroid.z() > 1.0f) {
            return PlaneInfo::Type::CEILING;
        } else {
            return PlaneInfo::Type::FLOOR;
        }
    } else if (dot_up < 0.3f) {
        // 接近垂直
        return PlaneInfo::Type::WALL;
    }
    
    return PlaneInfo::Type::OTHER;
}

void PlaneAligner::alignVertices(std::vector<VertexInfo>& vertices,
                                const std::vector<PlaneInfo>& planes) {
    
    int aligned_vertices = 0;
    
    for (const auto& plane : planes) {
        if (!plane.is_major_plane) continue;
        
        for (int vertex_idx : plane.vertex_indices) {
            auto& vertex = vertices[vertex_idx];
            
            // 计算到平面的距离
            float distance = distanceToPlane(vertex.position, plane);
            
            if (distance < config_.plane_alignment_threshold) {
                // 投影到平面
                Eigen::Vector3f projected = fusion_utils::projectToPlane(vertex.position, plane.coefficients);
                
                // 加权平均（保留一些原始位置）
                float alignment_weight = config_.wall_straightness_weight;
                vertex.position = (1.0f - alignment_weight) * vertex.position + alignment_weight * projected;
                
                aligned_vertices++;
            }
        }
    }
    
    std::cout << "对齐了 " << aligned_vertices << " 个顶点到主要平面" << std::endl;
}

void PlaneAligner::ensureWallStraightness(std::vector<VertexInfo>& vertices,
                                         const std::vector<PlaneInfo>& wall_planes) {
    
    for (const auto& wall : wall_planes) {
        // 对墙面顶点进行额外的直线对齐
        for (int vertex_idx : wall.vertex_indices) {
            auto& vertex = vertices[vertex_idx];
            
            // 投影到墙面平面
            Eigen::Vector3f projected = fusion_utils::projectToPlane(vertex.position, wall.coefficients);
            
            // 强制对齐到墙面
            vertex.position = projected;
        }
    }
    
    std::cout << "墙面直线对齐完成" << std::endl;
}

float PlaneAligner::distanceToPlane(const Eigen::Vector3f& point, const PlaneInfo& plane) {
    return std::abs(plane.normal.dot(point) + plane.coefficients.w()) / plane.normal.norm();
}

// ============================================================================
// TopologyRepairer 实现
// ============================================================================

TopologyRepairer::TopologyRepairer(const FusionConfig& config) : config_(config) {
}

TopologyRepairer::RepairResult TopologyRepairer::repairTopology(std::vector<VertexInfo>& vertices,
                                                               std::vector<FaceInfo>& faces) {
    RepairResult result;
    
    // 1. 移除孤立顶点
    if (config_.remove_isolated_vertices) {
        result.vertices_removed += removeIsolatedVertices(vertices, faces);
    }
    
    // 2. 移除退化面
    if (config_.remove_degenerate_faces) {
        result.faces_removed += removeDegenerateFaces(faces, vertices);
    }
    
    // 3. 修复非流形边
    if (config_.fix_non_manifold_edges) {
        int fixed = fixNonManifoldEdges(vertices, faces);
        if (fixed > 0) {
            result.fixed_issues.push_back(TopologyIssue::NON_MANIFOLD_EDGE);
        }
    }
    
    result.success = true;
    return result;
}

int TopologyRepairer::removeIsolatedVertices(std::vector<VertexInfo>& vertices,
                                            std::vector<FaceInfo>& faces) {
    
    std::vector<bool> vertex_used(vertices.size(), false);
    
    // 标记被面片使用的顶点
    for (const auto& face : faces) {
        for (int i = 0; i < 3; ++i) {
            if (face.vertex_indices[i] >= 0 && 
                face.vertex_indices[i] < static_cast<int>(vertices.size())) {
                vertex_used[face.vertex_indices[i]] = true;
            }
        }
    }
    
    // 标记孤立顶点为低置信度
    int removed_count = 0;
    for (size_t i = 0; i < vertices.size(); ++i) {
        if (!vertex_used[i]) {
            vertices[i].confidence = 0.0f;
            removed_count++;
        }
    }
    
    return removed_count;
}

int TopologyRepairer::removeDegenerateFaces(std::vector<FaceInfo>& faces,
                                           const std::vector<VertexInfo>& vertices) {
    
    int removed_count = 0;
    auto face_it = faces.begin();
    
    while (face_it != faces.end()) {
        bool is_degenerate = false;
        
        // 检查顶点索引有效性
        for (int i = 0; i < 3; ++i) {
            int idx = face_it->vertex_indices[i];
            if (idx < 0 || idx >= static_cast<int>(vertices.size()) || 
                vertices[idx].confidence < 0.1f) {
                is_degenerate = true;
                break;
            }
        }
        
        // 检查几何退化
        if (!is_degenerate) {
            const auto& v1 = vertices[face_it->vertex_indices[0]].position;
            const auto& v2 = vertices[face_it->vertex_indices[1]].position;
            const auto& v3 = vertices[face_it->vertex_indices[2]].position;
            
            is_degenerate = fusion_utils::isTriangleDegenerate(v1, v2, v3);
        }
        
        if (is_degenerate) {
            face_it = faces.erase(face_it);
            removed_count++;
        } else {
            ++face_it;
        }
    }
    
    return removed_count;
}

int TopologyRepairer::fixNonManifoldEdges(std::vector<VertexInfo>& vertices,
                                         std::vector<FaceInfo>& faces) {
    
    // 简化实现：检测并移除非流形边对应的面片
    std::unordered_map<std::pair<int, int>, int> edge_count;
    
    // 统计边的使用次数
    for (const auto& face : faces) {
        for (int i = 0; i < 3; ++i) {
            int v1 = face.vertex_indices[i];
            int v2 = face.vertex_indices[(i + 1) % 3];
            
            if (v1 > v2) std::swap(v1, v2);
            edge_count[{v1, v2}]++;
        }
    }
    
    // 标记非流形边
    std::unordered_set<std::pair<int, int>> non_manifold_edges;
    for (const auto& edge_pair : edge_count) {
        if (edge_pair.second > 2) {  // 非流形边
            non_manifold_edges.insert(edge_pair.first);
        }
    }
    
    // 移除包含非流形边的面片
    int fixed_count = 0;
    auto face_it = faces.begin();
    
    while (face_it != faces.end()) {
        bool has_non_manifold_edge = false;
        
        for (int i = 0; i < 3; ++i) {
            int v1 = face_it->vertex_indices[i];
            int v2 = face_it->vertex_indices[(i + 1) % 3];
            
            if (v1 > v2) std::swap(v1, v2);
            
            if (non_manifold_edges.count({v1, v2}) > 0) {
                has_non_manifold_edge = true;
                break;
            }
        }
        
        if (has_non_manifold_edge) {
            face_it = faces.erase(face_it);
            fixed_count++;
        } else {
            ++face_it;
        }
    }
    
    return fixed_count;
}

// ============================================================================
// ColorBlender 实现
// ============================================================================

ColorBlender::ColorBlender(const FusionConfig& config) : config_(config) {
    kdtree_ = std::make_unique<pcl::KdTreeFLANN<PointT>>();
}

void ColorBlender::blendMeshColors(std::vector<VertexInfo>& vertices,
                                  const PointCloudT::Ptr& original_cloud) {
    
    if (!original_cloud || original_cloud->empty()) {
        // 设置默认颜色
        for (auto& vertex : vertices) {
            vertex.color = Eigen::Vector3f(0.7f, 0.7f, 0.7f);  // 浅灰色
        }
        return;
    }
    
    kdtree_->setInputCloud(original_cloud);
    
    #pragma omp parallel for if(config_.use_parallel_processing)
    for (size_t i = 0; i < vertices.size(); ++i) {
        auto& vertex = vertices[i];
        vertex.color = computeVertexColor(vertex, *original_cloud);
    }
    
    std::cout << "颜色混合完成" << std::endl;
}

Eigen::Vector3f ColorBlender::computeVertexColor(const VertexInfo& vertex,
                                                const PointCloudT& original_cloud) {
    
    PointT search_point;
    search_point.x = vertex.position.x();
    search_point.y = vertex.position.y();
    search_point.z = vertex.position.z();
    
    std::vector<int> indices;
    std::vector<float> distances;
    
    if (kdtree_->radiusSearch(search_point, config_.color_blending_radius, indices, distances) == 0) {
        return Eigen::Vector3f(0.5f, 0.5f, 0.5f);  // 默认灰色
    }
    
    // 计算加权平均颜色
    Eigen::Vector3f blended_color(0, 0, 0);
    float total_weight = 0.0f;
    
    for (size_t i = 0; i < indices.size(); ++i) {
        const auto& pt = original_cloud.points[indices[i]];
        
        float weight = computeColorWeight(std::sqrt(distances[i]), config_.color_blending_radius);
        
        Eigen::Vector3f color(pt.r / 255.0f, pt.g / 255.0f, pt.b / 255.0f);
        blended_color += color * weight;
        total_weight += weight;
    }
    
    if (total_weight > 0.0f) {
        blended_color /= total_weight;
    }
    
    return fusion_utils::normalizeColor(blended_color);
}

float ColorBlender::computeColorWeight(float distance, float max_distance) {
    if (distance >= max_distance) {
        return 0.0f;
    }
    
    // 使用指数衰减
    float normalized_distance = distance / max_distance;
    return std::exp(-config_.color_weight_distance_decay * normalized_distance);
}

// ============================================================================
// 工具函数实现
// ============================================================================

namespace fusion_utils {

float computeTriangleArea(const Eigen::Vector3f& v1,
                         const Eigen::Vector3f& v2,
                         const Eigen::Vector3f& v3) {
    Eigen::Vector3f edge1 = v2 - v1;
    Eigen::Vector3f edge2 = v3 - v1;
    return 0.5f * edge1.cross(edge2).norm();
}

float computeTriangleQuality(const Eigen::Vector3f& v1,
                            const Eigen::Vector3f& v2,
                            const Eigen::Vector3f& v3) {
    
    float a = (v2 - v1).norm();
    float b = (v3 - v2).norm();
    float c = (v1 - v3).norm();
    
    if (a < 1e-8f || b < 1e-8f || c < 1e-8f) {
        return 0.0f;
    }
    
    float s = (a + b + c) * 0.5f;
    float area = std::sqrt(s * (s - a) * (s - b) * (s - c));
    
    // 质量 = 4 * sqrt(3) * area / (a² + b² + c²)
    float quality = 4.0f * std::sqrt(3.0f) * area / (a * a + b * b + c * c);
    
    return std::max(0.0f, std::min(1.0f, quality));
}

bool isTriangleDegenerate(const Eigen::Vector3f& v1,
                         const Eigen::Vector3f& v2,
                         const Eigen::Vector3f& v3,
                         float threshold) {
    
    float area = computeTriangleArea(v1, v2, v3);
    return area < threshold;
}

Eigen::Vector3f computeTriangleNormal(const Eigen::Vector3f& v1,
                                     const Eigen::Vector3f& v2,
                                     const Eigen::Vector3f& v3) {
    
    Eigen::Vector3f edge1 = v2 - v1;
    Eigen::Vector3f edge2 = v3 - v1;
    Eigen::Vector3f normal = edge1.cross(edge2);
    
    if (normal.norm() > 1e-8f) {
        return normal.normalized();
    }
    
    return Eigen::Vector3f(0, 0, 1);  // 默认向上
}

float distance(const Eigen::Vector3f& p1, const Eigen::Vector3f& p2) {
    return (p1 - p2).norm();
}

Eigen::Vector3f projectToPlane(const Eigen::Vector3f& point,
                              const Eigen::Vector4f& plane_coefficients) {
    
    Eigen::Vector3f normal(plane_coefficients.x(), plane_coefficients.y(), plane_coefficients.z());
    float d = plane_coefficients.w();
    
    float distance_to_plane = normal.dot(point) + d;
    return point - distance_to_plane * normal;
}

Eigen::Vector3f normalizeColor(const Eigen::Vector3f& color) {
    return Eigen::Vector3f(
        std::max(0.0f, std::min(1.0f, color.x())),
        std::max(0.0f, std::min(1.0f, color.y())),
        std::max(0.0f, std::min(1.0f, color.z()))
    );
}

} // namespace fusion_utils

} // namespace recon

