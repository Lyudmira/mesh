/**
 * 混合重建器实现
 */

#include "hybrid_reconstructor.h"
#include <pcl/common/common.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/surface/mls.h>
#include <pcl/surface/marching_cubes_hoppe.h>
#include <pcl/surface/grid_projection.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <chrono>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace recon {

// ============================================================================
// HybridReconstructor实现
// ============================================================================

HybridReconstructor::HybridReconstructor(const HybridReconstructorConfig& config)
    : config_(config) {
}

HybridReconstructor::~HybridReconstructor() = default;

bool HybridReconstructor::reconstruct(
    const std::vector<ReconstructionCluster>& clusters,
    const GridT& shell_grid,
    HybridReconstructionResult& result) {
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    std::cout << "开始混合重建，簇数量: " << clusters.size() << std::endl;
    
    // 1. 初始化
    if (!initialize()) {
        std::cerr << "混合重建器初始化失败" << std::endl;
        result.success = false;
        result.error_message = "初始化失败";
        return false;
    }
    
    // 2. 复制簇信息用于处理
    std::vector<ReconstructionCluster> working_clusters = clusters;
    result.clusters.clear();
    result.clusters.reserve(working_clusters.size());
    
    // 3. 并行重建各个簇
    std::cout << "开始并行重建各个簇..." << std::endl;
    
    #pragma omp parallel for if(config_.use_parallel_processing && working_clusters.size() > 1)
    for (size_t i = 0; i < working_clusters.size(); ++i) {
        auto cluster_start = std::chrono::high_resolution_clock::now();
        
        bool cluster_success = reconstructCluster(working_clusters[i], shell_grid);
        
        auto cluster_end = std::chrono::high_resolution_clock::now();
        working_clusters[i].reconstruction_time = 
            std::chrono::duration<double>(cluster_end - cluster_start).count();
        
        if (!cluster_success) {
            #pragma omp critical
            {
                std::cerr << "簇 " << working_clusters[i].cluster_id << " 重建失败" << std::endl;
            }
        } else {
            #pragma omp critical
            {
                std::cout << "簇 " << working_clusters[i].cluster_id << " 重建完成，方法: " 
                         << static_cast<int>(working_clusters[i].method) << std::endl;
            }
        }
    }
    
    // 4. 计算簇边界和邻接关系
    std::cout << "计算簇边界和邻接关系..." << std::endl;
    for (auto& cluster : working_clusters) {
        computeClusterBoundaries(cluster);
    }
    computeClusterAdjacency(working_clusters);
    
    // 5. 合并重建结果
    std::cout << "合并重建结果..." << std::endl;
    auto merge_start = std::chrono::high_resolution_clock::now();
    
    if (!mergeReconstructionResults(working_clusters, result.final_mesh)) {
        std::cerr << "重建结果合并失败" << std::endl;
        result.success = false;
        result.error_message = "合并失败";
        return false;
    }
    
    auto merge_end = std::chrono::high_resolution_clock::now();
    result.merging_time = std::chrono::duration<double>(merge_end - merge_start).count();
    
    // 6. 后处理
    std::cout << "执行后处理..." << std::endl;
    auto post_start = std::chrono::high_resolution_clock::now();
    
    if (!postProcessMesh(result.final_mesh)) {
        std::cerr << "后处理失败" << std::endl;
        result.success = false;
        result.error_message = "后处理失败";
        return false;
    }
    
    auto post_end = std::chrono::high_resolution_clock::now();
    result.post_processing_time = std::chrono::duration<double>(post_end - post_start).count();
    
    // 7. 更新结果统计
    result.clusters = working_clusters;
    updateStatistics(result);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    result.total_reconstruction_time = std::chrono::duration<double>(end_time - start_time).count();
    
    result.success = true;
    last_result_ = result;
    
    std::cout << "混合重建完成" << std::endl;
    std::cout << "统计信息:" << std::endl;
    std::cout << "  总簇数: " << result.total_clusters << std::endl;
    std::cout << "  GP3簇数: " << result.gp3_clusters << std::endl;
    std::cout << "  Poisson簇数: " << result.poisson_clusters << std::endl;
    std::cout << "  RIMLS簇数: " << result.rimls_clusters << std::endl;
    std::cout << "  混合簇数: " << result.hybrid_clusters << std::endl;
    std::cout << "  最终顶点数: " << result.final_vertex_count << std::endl;
    std::cout << "  最终面数: " << result.final_face_count << std::endl;
    std::cout << "  总重建时间: " << result.total_reconstruction_time << " 秒" << std::endl;
    std::cout << "  合并时间: " << result.merging_time << " 秒" << std::endl;
    std::cout << "  后处理时间: " << result.post_processing_time << " 秒" << std::endl;
    std::cout << "  整体质量分数: " << result.overall_quality_score << std::endl;
    
    return true;
}

bool HybridReconstructor::initialize() {
    // 初始化KD树
    kdtree_ = std::make_unique<pcl::KdTreeFLANN<PointT>>();
    
    // 清空缓存
    mesh_cache_.clear();
    
    std::cout << "混合重建器初始化完成" << std::endl;
    return true;
}

bool HybridReconstructor::reconstructCluster(
    ReconstructionCluster& cluster,
    const GridT& shell_grid) {
    
    if (!cluster.points || cluster.points->empty()) {
        std::cerr << "簇 " << cluster.cluster_id << " 点云为空" << std::endl;
        return false;
    }
    
    std::cout << "重建簇 " << cluster.cluster_id << "，点数: " << cluster.points->size() 
              << "，方法: " << static_cast<int>(cluster.method) << std::endl;
    
    // 预处理点云
    PointCloudT::Ptr processed_cloud = preprocessPointCloud(*cluster.points);
    if (!processed_cloud || processed_cloud->empty()) {
        std::cerr << "簇 " << cluster.cluster_id << " 预处理失败" << std::endl;
        return false;
    }
    
    // 根据选择的方法进行重建
    bool success = false;
    cluster.mesh = std::make_shared<pcl::PolygonMesh>();
    
    switch (cluster.method) {
        case ReconstructionMethod::GP3:
            success = reconstructWithGP3(*processed_cloud, cluster.mesh);
            break;
            
        case ReconstructionMethod::POISSON:
            success = reconstructWithPoisson(*processed_cloud, cluster.mesh);
            break;
            
        case ReconstructionMethod::RIMLS:
            success = reconstructWithRIMLS(*processed_cloud, cluster.mesh);
            break;
            
        case ReconstructionMethod::HYBRID:
            success = reconstructWithHybrid(*processed_cloud, cluster.quality, cluster.mesh);
            break;
            
        default:
            std::cerr << "不支持的重建方法: " << static_cast<int>(cluster.method) << std::endl;
            return false;
    }
    
    if (success && cluster.mesh && !cluster.mesh->polygons.empty()) {
        // 更新统计信息
        cluster.vertex_count = static_cast<int>(cluster.mesh->cloud.width * cluster.mesh->cloud.height);
        cluster.face_count = static_cast<int>(cluster.mesh->polygons.size());
        
        // 评估重建质量
        cluster.mesh_quality_score = evaluateReconstructionQuality(*cluster.mesh, *processed_cloud);
        
        std::cout << "簇 " << cluster.cluster_id << " 重建成功，顶点数: " << cluster.vertex_count 
                  << "，面数: " << cluster.face_count << "，质量分数: " << cluster.mesh_quality_score << std::endl;
        
        return true;
    } else {
        std::cerr << "簇 " << cluster.cluster_id << " 重建失败" << std::endl;
        return false;
    }
}

// ============================================================================
// 重建算法实现
// ============================================================================

bool HybridReconstructor::reconstructWithGP3(
    const PointCloudT& cloud,
    pcl::PolygonMesh::Ptr& mesh) {
    
    try {
        auto gp3 = createGP3Reconstructor();
        
        PointCloudT::Ptr cloud_ptr(new PointCloudT(cloud));
        gp3->setInputCloud(cloud_ptr);
        
        // 设置KD树
        pcl::KdTreeFLANN<PointT>::Ptr tree(new pcl::KdTreeFLANN<PointT>);
        gp3->setSearchMethod(tree);
        
        // 执行重建
        gp3->reconstruct(*mesh);
        
        return !mesh->polygons.empty();
        
    } catch (const std::exception& e) {
        std::cerr << "GP3重建异常: " << e.what() << std::endl;
        return false;
    }
}

bool HybridReconstructor::reconstructWithPoisson(
    const PointCloudT& cloud,
    pcl::PolygonMesh::Ptr& mesh) {
    
    try {
        auto poisson = createPoissonReconstructor();
        
        PointCloudT::Ptr cloud_ptr(new PointCloudT(cloud));
        poisson->setInputCloud(cloud_ptr);
        
        // 执行重建
        poisson->reconstruct(*mesh);
        
        return !mesh->polygons.empty();
        
    } catch (const std::exception& e) {
        std::cerr << "Poisson重建异常: " << e.what() << std::endl;
        return false;
    }
}

bool HybridReconstructor::reconstructWithRIMLS(
    const PointCloudT& cloud,
    pcl::PolygonMesh::Ptr& mesh) {
    
    try {
        // 使用MLS进行表面重建
        pcl::MovingLeastSquares<PointT, PointT> mls;
        PointCloudT::Ptr cloud_ptr(new PointCloudT(cloud));
        PointCloudT::Ptr mls_points(new PointCloudT);
        
        mls.setInputCloud(cloud_ptr);
        mls.setSearchRadius(config_.rimls.search_radius);
        mls.setPolynomialOrder(config_.rimls.polynomial_order);
        mls.setUpsamplingMethod(pcl::MovingLeastSquares<PointT, PointT>::SAMPLE_LOCAL_PLANE);
        mls.setUpsamplingRadius(config_.rimls.search_radius * 0.5f);
        mls.setUpsamplingStepSize(config_.rimls.step_size);
        
        // 计算MLS表面
        mls.process(*mls_points);
        
        if (mls_points->empty()) {
            std::cerr << "MLS处理后点云为空" << std::endl;
            return false;
        }
        
        // 使用GP3对MLS结果进行三角化
        return reconstructWithGP3(*mls_points, mesh);
        
    } catch (const std::exception& e) {
        std::cerr << "RIMLS重建异常: " << e.what() << std::endl;
        return false;
    }
}

bool HybridReconstructor::reconstructWithHybrid(
    const PointCloudT& cloud,
    const LocalQualityAssessment& quality,
    pcl::PolygonMesh::Ptr& mesh) {
    
    // 基于质量评估选择最佳方法组合
    std::vector<pcl::PolygonMesh::Ptr> candidate_meshes;
    std::vector<float> quality_scores;
    
    // 尝试多种方法
    pcl::PolygonMesh::Ptr gp3_mesh(new pcl::PolygonMesh);
    if (reconstructWithGP3(cloud, gp3_mesh)) {
        candidate_meshes.push_back(gp3_mesh);
        quality_scores.push_back(evaluateReconstructionQuality(*gp3_mesh, cloud));
    }
    
    pcl::PolygonMesh::Ptr poisson_mesh(new pcl::PolygonMesh);
    if (reconstructWithPoisson(cloud, poisson_mesh)) {
        candidate_meshes.push_back(poisson_mesh);
        quality_scores.push_back(evaluateReconstructionQuality(*poisson_mesh, cloud));
    }
    
    // 选择最佳结果
    if (!candidate_meshes.empty()) {
        auto best_it = std::max_element(quality_scores.begin(), quality_scores.end());
        size_t best_idx = std::distance(quality_scores.begin(), best_it);
        mesh = candidate_meshes[best_idx];
        return true;
    }
    
    return false;
}

// ============================================================================
// 重建器创建方法
// ============================================================================

std::unique_ptr<pcl::GreedyProjectionTriangulation<PointT>> 
HybridReconstructor::createGP3Reconstructor() {
    
    auto gp3 = std::make_unique<pcl::GreedyProjectionTriangulation<PointT>>();
    
    gp3->setSearchRadius(config_.gp3.search_radius);
    gp3->setMu(config_.gp3.mu);
    gp3->setMaximumNearestNeighbors(config_.gp3.maximum_nearest_neighbors);
    gp3->setMinimumAngle(config_.gp3.minimum_angle);
    gp3->setMaximumAngle(config_.gp3.maximum_angle);
    gp3->setMaximumSurfaceAngle(config_.gp3.maximum_surface_angle);
    gp3->setNormalConsistency(config_.gp3.normal_consistency);
    
    return gp3;
}

std::unique_ptr<pcl::Poisson<PointT>> HybridReconstructor::createPoissonReconstructor() {
    auto poisson = std::make_unique<pcl::Poisson<PointT>>();
    
    poisson->setDepth(config_.poisson.depth);
    poisson->setPointWeight(config_.poisson.point_weight);
    poisson->setSamplesPerNode(config_.poisson.samples_per_node);
    poisson->setScale(config_.poisson.scale);
    poisson->setIsoDivide(config_.poisson.iso_divide);
    poisson->setSolverDivide(config_.poisson.solver_divide);
    poisson->setConfidence(config_.poisson.confidence);
    poisson->setManifold(config_.poisson.manifold);
    poisson->setOutputPolygons(config_.poisson.output_polygons);
    
    return poisson;
}

// ============================================================================
// 合并和后处理方法
// ============================================================================

bool HybridReconstructor::mergeReconstructionResults(
    const std::vector<ReconstructionCluster>& clusters,
    pcl::PolygonMesh::Ptr& merged_mesh) {
    
    if (clusters.empty()) {
        std::cerr << "没有簇需要合并" << std::endl;
        return false;
    }
    
    merged_mesh = std::make_shared<pcl::PolygonMesh>();
    
    // 简单合并：直接连接所有网格
    pcl::PointCloud<pcl::PointXYZ> merged_vertices;
    std::vector<pcl::Vertices> merged_faces;
    
    int vertex_offset = 0;
    
    for (const auto& cluster : clusters) {
        if (!cluster.mesh || cluster.mesh->polygons.empty()) {
            continue;
        }
        
        // 转换点云
        pcl::PointCloud<pcl::PointXYZ> cluster_vertices;
        pcl::fromPCLPointCloud2(cluster.mesh->cloud, cluster_vertices);
        
        // 添加顶点
        for (const auto& vertex : cluster_vertices) {
            merged_vertices.push_back(vertex);
        }
        
        // 添加面，调整顶点索引
        for (const auto& polygon : cluster.mesh->polygons) {
            pcl::Vertices adjusted_polygon;
            for (uint32_t vertex_idx : polygon.vertices) {
                adjusted_polygon.vertices.push_back(vertex_idx + vertex_offset);
            }
            merged_faces.push_back(adjusted_polygon);
        }
        
        vertex_offset += static_cast<int>(cluster_vertices.size());
    }
    
    if (merged_vertices.empty() || merged_faces.empty()) {
        std::cerr << "合并后网格为空" << std::endl;
        return false;
    }
    
    // 转换为PCL网格格式
    pcl::toPCLPointCloud2(merged_vertices, merged_mesh->cloud);
    merged_mesh->polygons = merged_faces;
    
    std::cout << "网格合并完成，顶点数: " << merged_vertices.size() 
              << "，面数: " << merged_faces.size() << std::endl;
    
    // 如果启用无缝混合，进行边界处理
    if (config_.enable_seamless_blending) {
        // 这里可以实现更复杂的边界混合算法
        // 目前使用简单的后处理
    }
    
    return true;
}

bool HybridReconstructor::postProcessMesh(pcl::PolygonMesh::Ptr& mesh) {
    if (!mesh || mesh->polygons.empty()) {
        return false;
    }
    
    bool success = true;
    
    // 移除重复顶点
    if (config_.remove_duplicates) {
        success &= removeDuplicateVertices(mesh);
    }
    
    // 平滑边界
    if (config_.smooth_boundaries) {
        success &= smoothMeshBoundaries(mesh);
    }
    
    // 填充孔洞
    if (config_.fill_holes) {
        success &= fillSmallHoles(mesh);
    }
    
    // 优化拓扑
    success &= optimizeMeshTopology(mesh);
    
    // 验证质量
    success &= validateMeshQuality(*mesh);
    
    return success;
}

// ============================================================================
// 辅助方法实现
// ============================================================================

PointCloudT::Ptr HybridReconstructor::preprocessPointCloud(const PointCloudT& cloud) {
    PointCloudT::Ptr processed(new PointCloudT(cloud));
    
    // 统计离群点移除
    pcl::StatisticalOutlierRemoval<PointT> sor;
    sor.setInputCloud(processed);
    sor.setMeanK(50);
    sor.setStddevMulThresh(1.0);
    sor.filter(*processed);
    
    // 估计法向量（如果需要）
    if (!estimateNormals(processed)) {
        std::cerr << "法向量估计失败" << std::endl;
    }
    
    return processed;
}

bool HybridReconstructor::estimateNormals(PointCloudT::Ptr& cloud, float radius) {
    if (cloud->empty()) {
        return false;
    }
    
    // 检查是否已有法向量
    bool has_normals = true;
    for (const auto& point : *cloud) {
        if (std::isnan(point.normal_x) || std::isnan(point.normal_y) || std::isnan(point.normal_z)) {
            has_normals = false;
            break;
        }
    }
    
    if (has_normals) {
        return true; // 已有法向量
    }
    
    // 计算法向量
    pcl::NormalEstimation<PointT, PointT> ne;
    ne.setInputCloud(cloud);
    
    pcl::KdTreeFLANN<PointT>::Ptr tree(new pcl::KdTreeFLANN<PointT>);
    ne.setSearchMethod(tree);
    ne.setRadiusSearch(radius);
    
    ne.compute(*cloud);
    
    return true;
}

void HybridReconstructor::computeClusterBoundaries(ReconstructionCluster& cluster) {
    // 简化的边界计算
    // 实际应用中可以实现更复杂的边界检测算法
    
    if (!cluster.points || cluster.points->empty()) {
        return;
    }
    
    // 使用凸包或边界检测算法
    // 这里使用简单的距离阈值方法
    
    cluster.boundary_indices.clear();
    cluster.boundary_normals.clear();
    
    // 假设边界点是距离簇中心较远的点
    Eigen::Vector4f centroid;
    pcl::compute3DCentroid(*cluster.points, centroid);
    
    float max_distance = 0.0f;
    for (size_t i = 0; i < cluster.points->size(); ++i) {
        const auto& point = cluster.points->at(i);
        float distance = (Eigen::Vector3f(point.x, point.y, point.z) - 
                         centroid.head<3>()).norm();
        max_distance = std::max(max_distance, distance);
    }
    
    float boundary_threshold = max_distance * 0.8f;
    
    for (size_t i = 0; i < cluster.points->size(); ++i) {
        const auto& point = cluster.points->at(i);
        float distance = (Eigen::Vector3f(point.x, point.y, point.z) - 
                         centroid.head<3>()).norm();
        
        if (distance > boundary_threshold) {
            cluster.boundary_indices.push_back(static_cast<int>(i));
            cluster.boundary_normals.push_back(
                Eigen::Vector3f(point.normal_x, point.normal_y, point.normal_z));
        }
    }
}

void HybridReconstructor::computeClusterAdjacency(std::vector<ReconstructionCluster>& clusters) {
    // 计算簇之间的邻接关系
    
    for (size_t i = 0; i < clusters.size(); ++i) {
        for (size_t j = i + 1; j < clusters.size(); ++j) {
            // 检查两个簇是否邻接
            bool are_adjacent = false;
            float min_distance = std::numeric_limits<float>::max();
            
            // 简化的邻接检测：检查边界点之间的距离
            for (int idx1 : clusters[i].boundary_indices) {
                const auto& point1 = clusters[i].points->at(idx1);
                
                for (int idx2 : clusters[j].boundary_indices) {
                    const auto& point2 = clusters[j].points->at(idx2);
                    
                    float distance = std::sqrt(
                        std::pow(point1.x - point2.x, 2) +
                        std::pow(point1.y - point2.y, 2) +
                        std::pow(point1.z - point2.z, 2)
                    );
                    
                    min_distance = std::min(min_distance, distance);
                    
                    if (distance < config_.transition_bandwidth) {
                        are_adjacent = true;
                    }
                }
            }
            
            if (are_adjacent) {
                clusters[i].neighbor_clusters.push_back(clusters[j].cluster_id);
                clusters[j].neighbor_clusters.push_back(clusters[i].cluster_id);
                
                // 计算过渡权重
                float weight = std::exp(-min_distance / config_.transition_bandwidth);
                clusters[i].transition_weights[clusters[j].cluster_id] = weight;
                clusters[j].transition_weights[clusters[i].cluster_id] = weight;
            }
        }
    }
}

float HybridReconstructor::evaluateReconstructionQuality(
    const pcl::PolygonMesh& mesh,
    const PointCloudT& original_cloud) {
    
    if (mesh.polygons.empty() || original_cloud.empty()) {
        return 0.0f;
    }
    
    // 简化的质量评估
    float quality_score = 0.0f;
    
    // 1. 顶点数量合理性 (0.3权重)
    size_t vertex_count = mesh.cloud.width * mesh.cloud.height;
    float vertex_ratio = static_cast<float>(vertex_count) / original_cloud.size();
    float vertex_score = std::exp(-std::abs(vertex_ratio - 1.0f));
    quality_score += 0.3f * vertex_score;
    
    // 2. 面数量合理性 (0.3权重)
    size_t face_count = mesh.polygons.size();
    float expected_faces = vertex_count * 2.0f; // 经验值
    float face_ratio = static_cast<float>(face_count) / expected_faces;
    float face_score = std::exp(-std::abs(face_ratio - 1.0f));
    quality_score += 0.3f * face_score;
    
    // 3. 网格完整性 (0.4权重)
    float completeness_score = (face_count > 0 && vertex_count > 0) ? 1.0f : 0.0f;
    quality_score += 0.4f * completeness_score;
    
    return std::min(quality_score, 1.0f);
}

// ============================================================================
// 后处理方法实现
// ============================================================================

bool HybridReconstructor::removeDuplicateVertices(pcl::PolygonMesh::Ptr& mesh) {
    // 简化实现：这里应该实现真正的重复顶点移除算法
    return true;
}

bool HybridReconstructor::smoothMeshBoundaries(pcl::PolygonMesh::Ptr& mesh) {
    // 简化实现：这里应该实现边界平滑算法
    return true;
}

bool HybridReconstructor::fillSmallHoles(pcl::PolygonMesh::Ptr& mesh) {
    // 简化实现：这里应该实现孔洞填充算法
    return true;
}

bool HybridReconstructor::optimizeMeshTopology(pcl::PolygonMesh::Ptr& mesh) {
    // 简化实现：这里应该实现拓扑优化算法
    return true;
}

bool HybridReconstructor::validateMeshQuality(const pcl::PolygonMesh& mesh) {
    return !mesh.polygons.empty() && mesh.cloud.width * mesh.cloud.height > 0;
}

void HybridReconstructor::updateStatistics(HybridReconstructionResult& result) {
    result.total_clusters = static_cast<int>(result.clusters.size());
    result.gp3_clusters = 0;
    result.poisson_clusters = 0;
    result.rimls_clusters = 0;
    result.hybrid_clusters = 0;
    
    for (const auto& cluster : result.clusters) {
        switch (cluster.method) {
            case ReconstructionMethod::GP3:
                result.gp3_clusters++;
                break;
            case ReconstructionMethod::POISSON:
                result.poisson_clusters++;
                break;
            case ReconstructionMethod::RIMLS:
                result.rimls_clusters++;
                break;
            case ReconstructionMethod::HYBRID:
                result.hybrid_clusters++;
                break;
            default:
                break;
        }
    }
    
    if (result.final_mesh) {
        result.final_vertex_count = static_cast<int>(
            result.final_mesh->cloud.width * result.final_mesh->cloud.height);
        result.final_face_count = static_cast<int>(result.final_mesh->polygons.size());
    }
    
    // 计算整体质量分数
    float total_quality = 0.0f;
    int valid_clusters = 0;
    
    for (const auto& cluster : result.clusters) {
        if (cluster.mesh_quality_score > 0.0f) {
            total_quality += cluster.mesh_quality_score;
            valid_clusters++;
        }
    }
    
    result.overall_quality_score = (valid_clusters > 0) ? (total_quality / valid_clusters) : 0.0f;
}

// ============================================================================
// HybridReconstructorFactory实现
// ============================================================================

std::unique_ptr<HybridReconstructor> HybridReconstructorFactory::createStandardReconstructor() {
    HybridReconstructorConfig config;
    // 使用默认配置
    return std::make_unique<HybridReconstructor>(config);
}

std::unique_ptr<HybridReconstructor> HybridReconstructorFactory::createHighQualityReconstructor() {
    HybridReconstructorConfig config;
    
    // 高质量配置
    config.gp3.search_radius = 0.05f;
    config.gp3.maximum_nearest_neighbors = 150;
    config.poisson.depth = 10;
    config.poisson.point_weight = 6.0f;
    config.rimls.search_radius = 0.06f;
    config.rimls.polynomial_order = 3;
    
    config.smoothing_iterations = 5;
    config.enable_seamless_blending = true;
    
    return std::make_unique<HybridReconstructor>(config);
}

std::unique_ptr<HybridReconstructor> HybridReconstructorFactory::createFastReconstructor() {
    HybridReconstructorConfig config;
    
    // 快速配置
    config.gp3.search_radius = 0.15f;
    config.gp3.maximum_nearest_neighbors = 50;
    config.poisson.depth = 6;
    config.poisson.point_weight = 2.0f;
    
    config.smoothing_iterations = 1;
    config.enable_seamless_blending = false;
    config.remove_duplicates = false;
    config.fill_holes = false;
    
    return std::make_unique<HybridReconstructor>(config);
}

std::unique_ptr<HybridReconstructor> HybridReconstructorFactory::createMemoryOptimizedReconstructor() {
    HybridReconstructorConfig config;
    
    // 内存优化配置
    config.use_memory_optimization = true;
    config.max_memory_mb = 2048;
    config.gp3.maximum_nearest_neighbors = 75;
    config.poisson.depth = 7;
    
    return std::make_unique<HybridReconstructor>(config);
}

} // namespace recon

