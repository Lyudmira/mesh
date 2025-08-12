/**
 * 自适应细节层选择器实现
 * 根据局部法向质量和密度自动选择重建算法
 */

#include "adaptive_detail_selector.h"
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/common.h>
#include <pcl/features/normal_3d.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/surface/gp3.h>
#include <pcl/surface/poisson.h>
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
// LocalQualityAssessment 实现
// ============================================================================

void LocalQualityAssessment::computeOverallQuality() {
    // 加权计算总体质量
    overall_quality = 0.25f * normal_consistency +
                     0.25f * density_uniformity +
                     0.20f * surface_smoothness +
                     0.15f * coverage_completeness +
                     0.15f * sampling_adequacy;
    
    // 应用噪声惩罚
    overall_quality *= (1.0f - noise_level * 0.5f);
    
    // 确保在[0,1]范围内
    overall_quality = std::max(0.0f, std::min(1.0f, overall_quality));
}

// ============================================================================
// AdaptiveDetailSelector 实现
// ============================================================================

AdaptiveDetailSelector::AdaptiveDetailSelector(const DetailSelectorConfig& config) 
    : config_(config) {
    
    initializeComponents();
    
    // 设置OpenMP线程数
    #ifdef _OPENMP
    if (config_.use_parallel_processing && config_.num_threads > 0) {
        omp_set_num_threads(config_.num_threads);
    }
    #endif
}

AdaptiveDetailSelector::~AdaptiveDetailSelector() = default;

void AdaptiveDetailSelector::initializeComponents() {
    quality_evaluator_ = std::make_unique<LocalQualityEvaluator>(config_);
    clusterer_ = std::make_unique<AdaptiveClusterer>(config_);
    reconstructor_ = std::make_unique<HybridReconstructor>(config_);
}

std::vector<ReconstructionRecommendation> AdaptiveDetailSelector::analyzeAndRecommend(
    const PointCloudT::Ptr& cloud,
    const GridT::Ptr& udf_grid) {
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    std::cout << "开始自适应细节层分析..." << std::endl;
    std::cout << "输入点云大小: " << cloud->size() << std::endl;
    
    stats_.total_points = static_cast<int>(cloud->size());
    
    // 1. 执行自适应聚类
    std::cout << "执行自适应聚类..." << std::endl;
    auto clustering_start = std::chrono::high_resolution_clock::now();
    
    std::vector<PointCluster> clusters = performAdaptiveClustering(cloud);
    
    auto clustering_end = std::chrono::high_resolution_clock::now();
    stats_.clustering_time_seconds = std::chrono::duration<double>(clustering_end - clustering_start).count();
    stats_.total_clusters = static_cast<int>(clusters.size());
    
    std::cout << "聚类完成，共 " << clusters.size() << " 个聚类" << std::endl;
    
    // 2. 为每个聚类选择最优方法
    std::cout << "为聚类选择重建方法..." << std::endl;
    selectOptimalMethods(clusters);
    
    // 3. 生成推荐结果
    std::vector<ReconstructionRecommendation> recommendations;
    
    for (const auto& cluster : clusters) {
        ReconstructionRecommendation rec;
        rec.primary_method = cluster.recommended_method;
        rec.confidence = cluster.quality.overall_quality;
        rec.mixing_weight = 1.0f;
        
        // 设置算法特定参数
        switch (cluster.recommended_method) {
            case ReconstructionMethod::GP3:
                rec.parameters["search_radius"] = 0.1f;
                rec.parameters["mu"] = 2.5f;
                rec.parameters["maximum_nearest_neighbors"] = 100.0f;
                rec.reasoning = "高密度均匀区域，适合GP3快速三角化";
                stats_.gp3_clusters++;
                break;
                
            case ReconstructionMethod::POISSON:
                rec.parameters["depth"] = 8.0f;
                rec.parameters["point_weight"] = 4.0f;
                rec.parameters["scale"] = 1.1f;
                rec.reasoning = "平滑区域，适合Poisson全局重建";
                stats_.poisson_clusters++;
                break;
                
            case ReconstructionMethod::RIMLS:
                rec.parameters["search_radius"] = 0.05f;
                rec.parameters["polynomial_order"] = 2.0f;
                rec.reasoning = "噪声区域，适合RIMLS鲁棒重建";
                stats_.rimls_clusters++;
                break;
                
            case ReconstructionMethod::HYBRID:
                rec.secondary_method = ReconstructionMethod::POISSON;
                rec.mixing_weight = 0.6f;
                rec.reasoning = "复杂区域，需要混合方法";
                stats_.hybrid_clusters++;
                break;
                
            default:
                rec.reasoning = "默认方法";
                break;
        }
        
        recommendations.push_back(rec);
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    stats_.analysis_time_seconds = std::chrono::duration<double>(end_time - start_time).count();
    
    // 计算平均质量指标
    float total_quality = 0.0f;
    float total_normal_quality = 0.0f;
    float total_density_uniformity = 0.0f;
    
    for (const auto& cluster : clusters) {
        total_quality += cluster.quality.overall_quality;
        total_normal_quality += cluster.quality.normal_consistency;
        total_density_uniformity += cluster.quality.density_uniformity;
    }
    
    if (!clusters.empty()) {
        stats_.avg_cluster_quality = total_quality / clusters.size();
        stats_.avg_normal_quality = total_normal_quality / clusters.size();
        stats_.avg_density_uniformity = total_density_uniformity / clusters.size();
    }
    
    std::cout << "分析完成:" << std::endl;
    std::cout << "  GP3聚类: " << stats_.gp3_clusters << std::endl;
    std::cout << "  Poisson聚类: " << stats_.poisson_clusters << std::endl;
    std::cout << "  RIMLS聚类: " << stats_.rimls_clusters << std::endl;
    std::cout << "  混合聚类: " << stats_.hybrid_clusters << std::endl;
    std::cout << "  平均质量: " << stats_.avg_cluster_quality << std::endl;
    std::cout << "  分析时间: " << stats_.analysis_time_seconds << " 秒" << std::endl;
    
    return recommendations;
}

std::vector<PointCluster> AdaptiveDetailSelector::performAdaptiveClustering(const PointCloudT::Ptr& cloud) {
    return clusterer_->cluster(cloud);
}

void AdaptiveDetailSelector::selectOptimalMethods(std::vector<PointCluster>& clusters) {
    MethodSelectionStrategy strategy(MethodSelectionStrategy::StrategyType::HYBRID_ADAPTIVE, config_);
    
    #pragma omp parallel for if(config_.use_parallel_processing)
    for (size_t i = 0; i < clusters.size(); ++i) {
        auto& cluster = clusters[i];
        
        // 为聚类选择最优方法
        ReconstructionRecommendation rec = strategy.selectMethod(cluster.quality, cluster);
        cluster.recommended_method = rec.primary_method;
        
        if (config_.enable_debug_output) {
            #pragma omp critical
            {
                std::cout << "聚类 " << cluster.cluster_id << ": " 
                         << "方法=" << static_cast<int>(cluster.recommended_method)
                         << ", 质量=" << cluster.quality.overall_quality
                         << ", 点数=" << cluster.point_indices.size() << std::endl;
            }
        }
    }
}

bool AdaptiveDetailSelector::performHybridReconstruction(const std::vector<PointCluster>& clusters,
                                                        const PointCloudT::Ptr& cloud,
                                                        pcl::PolygonMesh& output_mesh) {
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    bool success = reconstructor_->reconstruct(clusters, cloud, output_mesh);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    stats_.reconstruction_time_seconds = std::chrono::duration<double>(end_time - start_time).count();
    
    return success;
}

// ============================================================================
// LocalQualityEvaluator 实现
// ============================================================================

LocalQualityEvaluator::LocalQualityEvaluator(const DetailSelectorConfig& config) 
    : config_(config) {
    kdtree_ = std::make_unique<pcl::KdTreeFLANN<PointT>>();
}

LocalQualityAssessment LocalQualityEvaluator::evaluate(const openvdb::Vec3f& center,
                                                       const PointCloudT& cloud,
                                                       float radius) {
    LocalQualityAssessment assessment;
    
    // 设置KD树
    PointCloudT::Ptr cloud_ptr(new PointCloudT(cloud));
    kdtree_->setInputCloud(cloud_ptr);
    
    // 查找邻域点
    PointT search_point;
    search_point.x = center.x();
    search_point.y = center.y();
    search_point.z = center.z();
    
    std::vector<int> indices;
    std::vector<float> distances;
    
    if (kdtree_->radiusSearch(search_point, radius, indices, distances) < 5) {
        // 点太少，质量很低
        assessment.overall_quality = 0.1f;
        return assessment;
    }
    
    // 评估各项质量指标
    evaluateNormalQuality(indices, cloud, assessment);
    evaluateDensityQuality(indices, distances, cloud, assessment);
    evaluateGeometricQuality(indices, cloud, assessment);
    evaluateCompleteness(center, indices, cloud, radius, assessment);
    
    // 计算综合质量
    assessment.computeOverallQuality();
    
    return assessment;
}

void LocalQualityEvaluator::evaluateNormalQuality(const std::vector<int>& indices,
                                                  const PointCloudT& cloud,
                                                  LocalQualityAssessment& assessment) {
    
    if (indices.size() < 3) {
        assessment.normal_consistency = 0.0f;
        assessment.normal_confidence = 0.0f;
        assessment.normal_smoothness = 0.0f;
        return;
    }
    
    // 计算法向量一致性
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
        assessment.normal_consistency = 0.0f;
        assessment.normal_confidence = 0.0f;
        assessment.normal_smoothness = 0.0f;
        return;
    }
    
    mean_normal /= valid_normals;
    mean_normal.normalize();
    
    // 计算一致性（角度方差的倒数）
    float angle_variance = 0.0f;
    for (int idx : indices) {
        const auto& pt = cloud.points[idx];
        if (std::isfinite(pt.normal_x) && std::isfinite(pt.normal_y) && std::isfinite(pt.normal_z)) {
            Eigen::Vector3f normal(pt.normal_x, pt.normal_y, pt.normal_z);
            normal.normalize();
            
            float dot_product = std::abs(mean_normal.dot(normal));
            dot_product = std::max(0.0f, std::min(1.0f, dot_product));
            float angle = std::acos(dot_product);
            angle_variance += angle * angle;
        }
    }
    
    angle_variance /= valid_normals;
    assessment.normal_consistency = 1.0f / (1.0f + angle_variance * 10.0f);
    
    // 置信度基于有效法向量的比例
    assessment.normal_confidence = static_cast<float>(valid_normals) / indices.size();
    
    // 平滑度基于法向量变化的平缓程度
    assessment.normal_smoothness = assessment.normal_consistency;
}

void LocalQualityEvaluator::evaluateDensityQuality(const std::vector<int>& indices,
                                                   const std::vector<float>& distances,
                                                   const PointCloudT& cloud,
                                                   LocalQualityAssessment& assessment) {
    
    if (indices.empty()) {
        assessment.point_density = 0.0f;
        assessment.density_uniformity = 0.0f;
        assessment.density_stability = 0.0f;
        return;
    }
    
    // 计算点密度
    float max_distance = std::sqrt(*std::max_element(distances.begin(), distances.end()));
    float volume = (4.0f / 3.0f) * M_PI * max_distance * max_distance * max_distance;
    assessment.point_density = static_cast<float>(indices.size()) / volume;
    
    // 计算密度均匀性（距离方差的倒数）
    float mean_distance = 0.0f;
    for (float dist : distances) {
        mean_distance += std::sqrt(dist);
    }
    mean_distance /= distances.size();
    
    float distance_variance = 0.0f;
    for (float dist : distances) {
        float d = std::sqrt(dist) - mean_distance;
        distance_variance += d * d;
    }
    distance_variance /= distances.size();
    
    assessment.density_uniformity = 1.0f / (1.0f + distance_variance * 100.0f);
    
    // 密度稳定性（基于局部密度变化）
    assessment.density_stability = assessment.density_uniformity;
}

void LocalQualityEvaluator::evaluateGeometricQuality(const std::vector<int>& indices,
                                                     const PointCloudT& cloud,
                                                     LocalQualityAssessment& assessment) {
    
    if (indices.size() < 10) {
        assessment.surface_smoothness = 0.0f;
        assessment.feature_sharpness = 0.0f;
        assessment.noise_level = 1.0f;
        return;
    }
    
    // 计算局部协方差矩阵
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
    
    // 特征值分解
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver(covariance);
    Eigen::Vector3f eigenvalues = solver.eigenvalues();
    
    // 表面平滑度（基于特征值比例）
    float lambda1 = eigenvalues(2);  // 最大特征值
    float lambda2 = eigenvalues(1);  // 中等特征值
    float lambda3 = eigenvalues(0);  // 最小特征值
    
    if (lambda1 > 1e-8f) {
        assessment.surface_smoothness = 1.0f - (lambda3 / lambda1);
        assessment.feature_sharpness = (lambda2 - lambda3) / lambda1;
    }
    
    // 噪声水平（基于最小特征值）
    assessment.noise_level = std::min(1.0f, lambda3 * 1000.0f);
}

void LocalQualityEvaluator::evaluateCompleteness(const openvdb::Vec3f& center,
                                                 const std::vector<int>& indices,
                                                 const PointCloudT& cloud,
                                                 float radius,
                                                 LocalQualityAssessment& assessment) {
    
    // 覆盖完整性：检查球面采样的均匀性
    int num_sectors = 26;  // 26个方向（6面+12边+8角）
    std::vector<bool> sector_covered(num_sectors, false);
    
    for (int idx : indices) {
        const auto& pt = cloud.points[idx];
        Eigen::Vector3f direction(pt.x - center.x(), pt.y - center.y(), pt.z - center.z());
        direction.normalize();
        
        // 简化的扇区映射
        int sector = 0;
        if (std::abs(direction.x()) > 0.7f) sector = direction.x() > 0 ? 0 : 1;
        else if (std::abs(direction.y()) > 0.7f) sector = direction.y() > 0 ? 2 : 3;
        else if (std::abs(direction.z()) > 0.7f) sector = direction.z() > 0 ? 4 : 5;
        else sector = 6 + (static_cast<int>(direction.x() * 10) % 20);
        
        sector = std::max(0, std::min(num_sectors - 1, sector));
        sector_covered[sector] = true;
    }
    
    int covered_sectors = std::count(sector_covered.begin(), sector_covered.end(), true);
    assessment.coverage_completeness = static_cast<float>(covered_sectors) / num_sectors;
    
    // 采样充分性：基于点密度
    float expected_points = assessment.point_density * (4.0f / 3.0f) * M_PI * radius * radius * radius;
    assessment.sampling_adequacy = std::min(1.0f, static_cast<float>(indices.size()) / expected_points);
}

// ============================================================================
// AdaptiveClusterer 实现
// ============================================================================

AdaptiveClusterer::AdaptiveClusterer(const DetailSelectorConfig& config) 
    : config_(config) {
    kdtree_ = std::make_unique<pcl::KdTreeFLANN<PointT>>();
}

std::vector<PointCluster> AdaptiveClusterer::cluster(const PointCloudT::Ptr& cloud) {
    std::cout << "开始自适应聚类..." << std::endl;
    
    // 设置KD树
    kdtree_->setInputCloud(cloud);
    
    // 1. 基于几何的初始聚类
    std::vector<PointCluster> geometric_clusters = clusterByGeometry(cloud);
    
    std::cout << "几何聚类完成，共 " << geometric_clusters.size() << " 个聚类" << std::endl;
    
    // 2. 评估每个聚类的质量
    LocalQualityEvaluator evaluator(config_);
    
    for (auto& cluster : geometric_clusters) {
        // 计算聚类质心
        Eigen::Vector3f centroid(0, 0, 0);
        for (int idx : cluster.point_indices) {
            const auto& pt = cloud->points[idx];
            centroid += Eigen::Vector3f(pt.x, pt.y, pt.z);
        }
        centroid /= cluster.point_indices.size();
        cluster.centroid = centroid;
        
        // 评估质量
        openvdb::Vec3f center(centroid.x(), centroid.y(), centroid.z());
        cluster.quality = evaluator.evaluate(center, *cloud, config_.clustering_radius);
        
        // 计算一致性评分
        cluster.coherence_score = computeClusterCoherence(cluster, *cloud);
    }
    
    // 3. 优化聚类结果
    optimizeClusters(geometric_clusters, *cloud);
    
    std::cout << "聚类优化完成，最终 " << geometric_clusters.size() << " 个聚类" << std::endl;
    
    return geometric_clusters;
}

std::vector<PointCluster> AdaptiveClusterer::clusterByGeometry(const PointCloudT::Ptr& cloud) {
    std::vector<PointCluster> clusters;
    std::vector<bool> processed(cloud->size(), false);
    int cluster_id = 0;
    
    for (size_t i = 0; i < cloud->size(); ++i) {
        if (processed[i]) continue;
        
        // 开始新聚类
        PointCluster cluster;
        cluster.cluster_id = cluster_id++;
        
        std::queue<int> to_process;
        to_process.push(static_cast<int>(i));
        processed[i] = true;
        
        // 区域增长
        while (!to_process.empty()) {
            int current_idx = to_process.front();
            to_process.pop();
            cluster.point_indices.push_back(current_idx);
            
            const auto& current_pt = cloud->points[current_idx];
            
            // 查找邻居
            std::vector<int> neighbor_indices;
            std::vector<float> neighbor_distances;
            
            if (kdtree_->radiusSearch(current_pt, config_.clustering_radius, 
                                     neighbor_indices, neighbor_distances) > 0) {
                
                for (int neighbor_idx : neighbor_indices) {
                    if (processed[neighbor_idx]) continue;
                    
                    const auto& neighbor_pt = cloud->points[neighbor_idx];
                    
                    // 检查法向量一致性
                    if (std::isfinite(current_pt.normal_x) && std::isfinite(neighbor_pt.normal_x)) {
                        Eigen::Vector3f n1(current_pt.normal_x, current_pt.normal_y, current_pt.normal_z);
                        Eigen::Vector3f n2(neighbor_pt.normal_x, neighbor_pt.normal_y, neighbor_pt.normal_z);
                        
                        float dot_product = std::abs(n1.normalized().dot(n2.normalized()));
                        
                        if (dot_product > config_.cluster_normal_threshold) {
                            processed[neighbor_idx] = true;
                            to_process.push(neighbor_idx);
                        }
                    } else {
                        // 如果没有法向量，仅基于距离
                        processed[neighbor_idx] = true;
                        to_process.push(neighbor_idx);
                    }
                }
            }
        }
        
        // 只保留足够大的聚类
        if (cluster.point_indices.size() >= static_cast<size_t>(config_.min_points_per_cluster)) {
            clusters.push_back(cluster);
        }
    }
    
    return clusters;
}

float AdaptiveClusterer::computeClusterCoherence(const PointCluster& cluster,
                                                const PointCloudT& cloud) {
    
    if (cluster.point_indices.size() < 3) {
        return 0.0f;
    }
    
    // 计算法向量一致性
    Eigen::Vector3f mean_normal(0, 0, 0);
    int valid_normals = 0;
    
    for (int idx : cluster.point_indices) {
        const auto& pt = cloud.points[idx];
        if (std::isfinite(pt.normal_x) && std::isfinite(pt.normal_y) && std::isfinite(pt.normal_z)) {
            mean_normal += Eigen::Vector3f(pt.normal_x, pt.normal_y, pt.normal_z);
            valid_normals++;
        }
    }
    
    if (valid_normals == 0) {
        return 0.5f;  // 中等一致性
    }
    
    mean_normal /= valid_normals;
    mean_normal.normalize();
    
    // 计算一致性评分
    float consistency_sum = 0.0f;
    for (int idx : cluster.point_indices) {
        const auto& pt = cloud.points[idx];
        if (std::isfinite(pt.normal_x) && std::isfinite(pt.normal_y) && std::isfinite(pt.normal_z)) {
            Eigen::Vector3f normal(pt.normal_x, pt.normal_y, pt.normal_z);
            normal.normalize();
            
            float dot_product = std::abs(mean_normal.dot(normal));
            consistency_sum += dot_product;
        }
    }
    
    return consistency_sum / valid_normals;
}

void AdaptiveClusterer::optimizeClusters(std::vector<PointCluster>& clusters,
                                        const PointCloudT& cloud) {
    
    std::cout << "优化聚类结果..." << std::endl;
    
    // 1. 合并相似聚类
    mergeSimilarClusters(clusters, cloud);
    
    // 2. 分割过大聚类
    splitLargeClusters(clusters, cloud);
    
    // 3. 检测边界点
    for (auto& cluster : clusters) {
        cluster.boundary_points = detectBoundaryPoints(cluster, cloud);
        cluster.is_boundary_cluster = !cluster.boundary_points.empty();
        cluster.needs_shell_guidance = cluster.is_boundary_cluster;
    }
    
    std::cout << "聚类优化完成" << std::endl;
}

void AdaptiveClusterer::mergeSimilarClusters(std::vector<PointCluster>& clusters,
                                            const PointCloudT& cloud) {
    
    bool merged = true;
    while (merged) {
        merged = false;
        
        for (size_t i = 0; i < clusters.size() && !merged; ++i) {
            for (size_t j = i + 1; j < clusters.size() && !merged; ++j) {
                auto& cluster1 = clusters[i];
                auto& cluster2 = clusters[j];
                
                // 检查聚类相似性
                float distance = (cluster1.centroid - cluster2.centroid).norm();
                float normal_similarity = cluster1.dominant_normal.dot(cluster2.dominant_normal);
                
                if (distance < config_.clustering_radius * 2.0f && 
                    normal_similarity > config_.cluster_normal_threshold) {
                    
                    // 合并聚类
                    cluster1.point_indices.insert(cluster1.point_indices.end(),
                                                  cluster2.point_indices.begin(),
                                                  cluster2.point_indices.end());
                    
                    // 重新计算质心
                    Eigen::Vector3f new_centroid(0, 0, 0);
                    for (int idx : cluster1.point_indices) {
                        const auto& pt = cloud.points[idx];
                        new_centroid += Eigen::Vector3f(pt.x, pt.y, pt.z);
                    }
                    cluster1.centroid = new_centroid / cluster1.point_indices.size();
                    
                    // 删除第二个聚类
                    clusters.erase(clusters.begin() + j);
                    merged = true;
                }
            }
        }
    }
}

void AdaptiveClusterer::splitLargeClusters(std::vector<PointCluster>& clusters,
                                          const PointCloudT& cloud) {
    
    std::vector<PointCluster> new_clusters;
    
    for (auto& cluster : clusters) {
        if (cluster.point_indices.size() > 10000) {  // 大于10k点的聚类
            // 使用K-means分割
            // 简化实现：随机分割
            size_t half_size = cluster.point_indices.size() / 2;
            
            PointCluster cluster1 = cluster;
            PointCluster cluster2;
            cluster2.cluster_id = cluster.cluster_id + 1000;  // 避免ID冲突
            
            cluster1.point_indices.resize(half_size);
            cluster2.point_indices.assign(cluster.point_indices.begin() + half_size,
                                         cluster.point_indices.end());
            
            new_clusters.push_back(cluster1);
            new_clusters.push_back(cluster2);
        } else {
            new_clusters.push_back(cluster);
        }
    }
    
    clusters = std::move(new_clusters);
}

std::vector<int> AdaptiveClusterer::detectBoundaryPoints(const PointCluster& cluster,
                                                        const PointCloudT& cloud) {
    std::vector<int> boundary_points;
    
    // 简化的边界检测：检查每个点的邻域密度
    for (int idx : cluster.point_indices) {
        const auto& pt = cloud.points[idx];
        
        std::vector<int> neighbor_indices;
        std::vector<float> neighbor_distances;
        
        int neighbor_count = kdtree_->radiusSearch(pt, config_.clustering_radius * 0.5f,
                                                  neighbor_indices, neighbor_distances);
        
        // 如果邻居数量少于阈值，认为是边界点
        if (neighbor_count < config_.min_points_per_cluster / 10) {
            boundary_points.push_back(idx);
        }
    }
    
    return boundary_points;
}

// ============================================================================
// MethodSelectionStrategy 实现
// ============================================================================

MethodSelectionStrategy::MethodSelectionStrategy(StrategyType type, const DetailSelectorConfig& config)
    : strategy_type_(type), config_(config) {
}

ReconstructionRecommendation MethodSelectionStrategy::selectMethod(const LocalQualityAssessment& quality,
                                                                   const PointCluster& cluster) {
    
    switch (strategy_type_) {
        case StrategyType::QUALITY_BASED:
            return selectByQuality(quality);
        case StrategyType::DENSITY_BASED:
            return selectByDensity(quality);
        case StrategyType::FEATURE_BASED:
            return selectByFeature(quality);
        case StrategyType::HYBRID_ADAPTIVE:
        default:
            return selectHybridAdaptive(quality, cluster);
    }
}

ReconstructionRecommendation MethodSelectionStrategy::selectByQuality(const LocalQualityAssessment& quality) {
    ReconstructionRecommendation rec;
    rec.confidence = quality.overall_quality;
    
    if (quality.normal_consistency > config_.normal_quality_threshold &&
        quality.density_uniformity > config_.density_uniformity_threshold) {
        // 高质量区域，使用GP3
        rec.primary_method = ReconstructionMethod::GP3;
        rec.reasoning = "高质量区域，法向量一致且密度均匀";
    } else if (quality.surface_smoothness > 0.7f && quality.noise_level < 0.3f) {
        // 平滑区域，使用Poisson
        rec.primary_method = ReconstructionMethod::POISSON;
        rec.reasoning = "平滑区域，适合全局重建";
    } else if (quality.noise_level > config_.noise_level_threshold) {
        // 噪声区域，使用RIMLS
        rec.primary_method = ReconstructionMethod::RIMLS;
        rec.reasoning = "噪声区域，需要鲁棒重建";
    } else {
        // 混合方法
        rec.primary_method = ReconstructionMethod::HYBRID;
        rec.secondary_method = ReconstructionMethod::POISSON;
        rec.mixing_weight = 0.6f;
        rec.reasoning = "复杂区域，使用混合方法";
    }
    
    return rec;
}

ReconstructionRecommendation MethodSelectionStrategy::selectByDensity(const LocalQualityAssessment& quality) {
    ReconstructionRecommendation rec;
    rec.confidence = quality.density_uniformity;
    
    if (quality.point_density > config_.gp3_density_threshold) {
        rec.primary_method = ReconstructionMethod::GP3;
        rec.reasoning = "高密度区域，适合GP3";
    } else if (quality.density_uniformity > 0.6f) {
        rec.primary_method = ReconstructionMethod::POISSON;
        rec.reasoning = "密度均匀，适合Poisson";
    } else {
        rec.primary_method = ReconstructionMethod::RIMLS;
        rec.reasoning = "密度不均匀，使用RIMLS";
    }
    
    return rec;
}

ReconstructionRecommendation MethodSelectionStrategy::selectByFeature(const LocalQualityAssessment& quality) {
    ReconstructionRecommendation rec;
    rec.confidence = quality.feature_sharpness;
    
    if (quality.feature_sharpness > 0.5f) {
        rec.primary_method = ReconstructionMethod::GP3;
        rec.reasoning = "特征明显，使用GP3保持细节";
    } else if (quality.surface_smoothness > 0.8f) {
        rec.primary_method = ReconstructionMethod::POISSON;
        rec.reasoning = "表面平滑，使用Poisson";
    } else {
        rec.primary_method = ReconstructionMethod::RIMLS;
        rec.reasoning = "特征模糊，使用RIMLS";
    }
    
    return rec;
}

ReconstructionRecommendation MethodSelectionStrategy::selectHybridAdaptive(const LocalQualityAssessment& quality,
                                                                           const PointCluster& cluster) {
    ReconstructionRecommendation rec;
    
    // 计算各方法的适用性评分
    float gp3_score = 0.0f;
    float poisson_score = 0.0f;
    float rimls_score = 0.0f;
    
    // GP3适用性：高密度 + 好法向量 + 低噪声
    gp3_score = quality.density_uniformity * 0.4f +
                quality.normal_consistency * 0.4f +
                (1.0f - quality.noise_level) * 0.2f;
    
    // Poisson适用性：平滑表面 + 完整覆盖
    poisson_score = quality.surface_smoothness * 0.5f +
                    quality.coverage_completeness * 0.3f +
                    quality.normal_consistency * 0.2f;
    
    // RIMLS适用性：高噪声 + 不完整采样
    rimls_score = quality.noise_level * 0.4f +
                  (1.0f - quality.sampling_adequacy) * 0.3f +
                  (1.0f - quality.density_uniformity) * 0.3f;
    
    // 选择最高评分的方法
    if (gp3_score > poisson_score && gp3_score > rimls_score) {
        rec.primary_method = ReconstructionMethod::GP3;
        rec.confidence = gp3_score;
        rec.reasoning = "GP3评分最高: 密度=" + std::to_string(quality.density_uniformity) +
                       ", 法向量=" + std::to_string(quality.normal_consistency);
    } else if (poisson_score > rimls_score) {
        rec.primary_method = ReconstructionMethod::POISSON;
        rec.confidence = poisson_score;
        rec.reasoning = "Poisson评分最高: 平滑度=" + std::to_string(quality.surface_smoothness) +
                       ", 完整性=" + std::to_string(quality.coverage_completeness);
    } else {
        rec.primary_method = ReconstructionMethod::RIMLS;
        rec.confidence = rimls_score;
        rec.reasoning = "RIMLS评分最高: 噪声=" + std::to_string(quality.noise_level) +
                       ", 采样=" + std::to_string(quality.sampling_adequacy);
    }
    
    // 如果评分接近，使用混合方法
    float max_score = std::max({gp3_score, poisson_score, rimls_score});
    float second_max = 0.0f;
    ReconstructionMethod second_method = ReconstructionMethod::POISSON;
    
    if (gp3_score != max_score && gp3_score > second_max) {
        second_max = gp3_score;
        second_method = ReconstructionMethod::GP3;
    }
    if (poisson_score != max_score && poisson_score > second_max) {
        second_max = poisson_score;
        second_method = ReconstructionMethod::POISSON;
    }
    if (rimls_score != max_score && rimls_score > second_max) {
        second_max = rimls_score;
        second_method = ReconstructionMethod::RIMLS;
    }
    
    // 如果第二高评分接近最高评分，使用混合方法
    if (max_score - second_max < 0.2f && config_.enable_hybrid_reconstruction) {
        rec.primary_method = ReconstructionMethod::HYBRID;
        rec.secondary_method = second_method;
        rec.mixing_weight = second_max / (max_score + second_max);
        rec.reasoning += " + 混合方法(评分接近)";
    }
    
    return rec;
}

// ============================================================================
// HybridReconstructor 实现
// ============================================================================

HybridReconstructor::HybridReconstructor(const DetailSelectorConfig& config) 
    : config_(config) {
}

bool HybridReconstructor::reconstruct(const std::vector<PointCluster>& clusters,
                                     const PointCloudT::Ptr& cloud,
                                     pcl::PolygonMesh& output_mesh) {
    
    std::cout << "开始混合重建..." << std::endl;
    
    std::vector<pcl::PolygonMesh> cluster_meshes;
    cluster_meshes.reserve(clusters.size());
    
    // 并行重建各个聚类
    #pragma omp parallel for if(config_.use_parallel_processing)
    for (size_t i = 0; i < clusters.size(); ++i) {
        const auto& cluster = clusters[i];
        
        pcl::PolygonMesh cluster_mesh;
        bool success = false;
        
        switch (cluster.recommended_method) {
            case ReconstructionMethod::GP3:
                success = reconstructGP3(cluster, cloud, cluster_mesh);
                break;
            case ReconstructionMethod::POISSON:
                success = reconstructPoisson(cluster, cloud, cluster_mesh);
                break;
            case ReconstructionMethod::RIMLS:
                success = reconstructRIMLS(cluster, cloud, cluster_mesh);
                break;
            case ReconstructionMethod::HYBRID:
                // 混合重建：先用主方法，再用次方法
                success = reconstructPoisson(cluster, cloud, cluster_mesh);
                break;
            default:
                success = reconstructGP3(cluster, cloud, cluster_mesh);
                break;
        }
        
        if (success) {
            #pragma omp critical
            {
                cluster_meshes.push_back(cluster_mesh);
            }
        }
    }
    
    std::cout << "聚类重建完成，成功 " << cluster_meshes.size() << "/" << clusters.size() << " 个" << std::endl;
    
    // 融合所有网格
    if (!cluster_meshes.empty()) {
        return fuseMeshes(cluster_meshes, output_mesh);
    }
    
    return false;
}

bool HybridReconstructor::reconstructGP3(const PointCluster& cluster,
                                         const PointCloudT::Ptr& cloud,
                                         pcl::PolygonMesh& mesh) {
    
    // 提取聚类点云
    PointCloudT::Ptr cluster_cloud(new PointCloudT);
    for (int idx : cluster.point_indices) {
        cluster_cloud->push_back(cloud->points[idx]);
    }
    
    if (cluster_cloud->size() < 3) {
        return false;
    }
    
    // GP3重建
    pcl::GreedyProjectionTriangulation<PointT> gp3;
    pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>);
    
    gp3.setInputCloud(cluster_cloud);
    gp3.setSearchMethod(tree);
    gp3.setSearchRadius(0.1f);
    gp3.setMu(2.5);
    gp3.setMaximumNearestNeighbors(100);
    gp3.setMaximumSurfaceAngle(M_PI/4);
    gp3.setMinimumAngle(M_PI/18);
    gp3.setMaximumAngle(2*M_PI/3);
    gp3.setNormalConsistency(false);
    
    try {
        gp3.reconstruct(mesh);
        return mesh.polygons.size() > 0;
    } catch (const std::exception& e) {
        std::cerr << "GP3重建失败: " << e.what() << std::endl;
        return false;
    }
}

bool HybridReconstructor::reconstructPoisson(const PointCluster& cluster,
                                            const PointCloudT::Ptr& cloud,
                                            pcl::PolygonMesh& mesh) {
    
    // 提取聚类点云
    PointCloudT::Ptr cluster_cloud(new PointCloudT);
    for (int idx : cluster.point_indices) {
        cluster_cloud->push_back(cloud->points[idx]);
    }
    
    if (cluster_cloud->size() < 10) {
        return false;
    }
    
    // Poisson重建
    pcl::Poisson<PointT> poisson;
    poisson.setInputCloud(cluster_cloud);
    poisson.setDepth(8);
    poisson.setPointWeight(4.0f);
    poisson.setScale(1.1f);
    poisson.setIsoDivide(8);
    poisson.setSolverDivide(8);
    
    try {
        poisson.reconstruct(mesh);
        return mesh.polygons.size() > 0;
    } catch (const std::exception& e) {
        std::cerr << "Poisson重建失败: " << e.what() << std::endl;
        return false;
    }
}

bool HybridReconstructor::reconstructRIMLS(const PointCluster& cluster,
                                          const PointCloudT::Ptr& cloud,
                                          pcl::PolygonMesh& mesh) {
    
    // RIMLS重建的简化实现
    // 实际应该使用PCL的MLS实现
    
    // 暂时使用GP3作为替代
    return reconstructGP3(cluster, cloud, mesh);
}

bool HybridReconstructor::fuseMeshes(const std::vector<pcl::PolygonMesh>& meshes,
                                    pcl::PolygonMesh& output_mesh) {
    
    if (meshes.empty()) {
        return false;
    }
    
    if (meshes.size() == 1) {
        output_mesh = meshes[0];
        return true;
    }
    
    // 简化的网格融合：直接合并顶点和面
    output_mesh.cloud = meshes[0].cloud;
    output_mesh.polygons = meshes[0].polygons;
    
    for (size_t i = 1; i < meshes.size(); ++i) {
        const auto& mesh = meshes[i];
        
        // 合并点云
        pcl::PointCloud<pcl::PointXYZ> current_cloud;
        pcl::fromPCLPointCloud2(output_mesh.cloud, current_cloud);
        
        pcl::PointCloud<pcl::PointXYZ> new_cloud;
        pcl::fromPCLPointCloud2(mesh.cloud, new_cloud);
        
        int vertex_offset = static_cast<int>(current_cloud.size());
        current_cloud += new_cloud;
        
        pcl::toPCLPointCloud2(current_cloud, output_mesh.cloud);
        
        // 合并面片，调整顶点索引
        for (const auto& polygon : mesh.polygons) {
            pcl::Vertices adjusted_polygon = polygon;
            for (auto& vertex_idx : adjusted_polygon.vertices) {
                vertex_idx += vertex_offset;
            }
            output_mesh.polygons.push_back(adjusted_polygon);
        }
    }
    
    std::cout << "网格融合完成，总顶点数: " << output_mesh.cloud.width 
              << ", 总面片数: " << output_mesh.polygons.size() << std::endl;
    
    return true;
}

// ============================================================================
// OffsetBandProcessor 实现
// ============================================================================

OffsetBandProcessor::OffsetBandProcessor(const DetailSelectorConfig& config) 
    : config_(config) {
}

OffsetBandProcessor::OffsetBandInfo OffsetBandProcessor::extractOffsetBand(
    const PointCloudT::Ptr& cloud,
    const openvdb::FloatGrid::Ptr& udf_grid) {
    
    OffsetBandInfo band_info;
    band_info.band_width = config_.offset_band_width;
    
    // 遍历所有点，根据UDF值分类
    for (size_t i = 0; i < cloud->size(); ++i) {
        const auto& pt = cloud->points[i];
        openvdb::Vec3f world_pos(pt.x, pt.y, pt.z);
        
        // 获取UDF值
        auto accessor = udf_grid->getConstAccessor();
        openvdb::Coord coord = udf_grid->transform().worldToIndexCellCentered(world_pos);
        float udf_value = accessor.getValue(coord);
        
        if (udf_value <= config_.offset_band_width) {
            if (udf_value <= config_.offset_band_width * 0.3f) {
                band_info.inner_points.push_back(static_cast<int>(i));
            } else if (udf_value >= config_.offset_band_width * 0.7f) {
                band_info.outer_points.push_back(static_cast<int>(i));
            } else {
                band_info.boundary_points.push_back(static_cast<int>(i));
            }
        }
    }
    
    // 估计外壳法向量
    if (config_.use_shell_normal_guidance && !band_info.boundary_points.empty()) {
        band_info.shell_normal = estimateShellNormal(*cloud, band_info.boundary_points);
        
        // 使用外壳法向量重新分类点
        classifyPointsByShellNormal(band_info, *cloud);
    }
    
    std::cout << "偏移带提取完成:" << std::endl;
    std::cout << "  内侧点: " << band_info.inner_points.size() << std::endl;
    std::cout << "  外侧点: " << band_info.outer_points.size() << std::endl;
    std::cout << "  边界点: " << band_info.boundary_points.size() << std::endl;
    
    return band_info;
}

void OffsetBandProcessor::classifyPointsByShellNormal(OffsetBandInfo& band_info,
                                                     const PointCloudT& cloud) {
    
    if (band_info.boundary_points.empty()) {
        return;
    }
    
    // 计算边界质心作为参考点
    Eigen::Vector3f boundary_centroid(0, 0, 0);
    for (int idx : band_info.boundary_points) {
        const auto& pt = cloud.points[idx];
        boundary_centroid += Eigen::Vector3f(pt.x, pt.y, pt.z);
    }
    boundary_centroid /= band_info.boundary_points.size();
    
    // 重新分类内侧和外侧点
    std::vector<int> new_inner_points;
    std::vector<int> new_outer_points;
    
    for (int idx : band_info.inner_points) {
        if (isPointInside(cloud.points[idx], band_info.shell_normal, boundary_centroid)) {
            new_inner_points.push_back(idx);
        } else {
            new_outer_points.push_back(idx);
        }
    }
    
    for (int idx : band_info.outer_points) {
        if (isPointInside(cloud.points[idx], band_info.shell_normal, boundary_centroid)) {
            new_inner_points.push_back(idx);
        } else {
            new_outer_points.push_back(idx);
        }
    }
    
    band_info.inner_points = std::move(new_inner_points);
    band_info.outer_points = std::move(new_outer_points);
}

Eigen::Vector3f OffsetBandProcessor::estimateShellNormal(const PointCloudT& cloud,
                                                        const std::vector<int>& boundary_points) {
    
    if (boundary_points.size() < 3) {
        return Eigen::Vector3f(0, 0, 1);  // 默认向上
    }
    
    // 使用PCA估计主要方向
    Eigen::Matrix3f covariance = Eigen::Matrix3f::Zero();
    Eigen::Vector3f centroid(0, 0, 0);
    
    // 计算质心
    for (int idx : boundary_points) {
        const auto& pt = cloud.points[idx];
        centroid += Eigen::Vector3f(pt.x, pt.y, pt.z);
    }
    centroid /= boundary_points.size();
    
    // 计算协方差矩阵
    for (int idx : boundary_points) {
        const auto& pt = cloud.points[idx];
        Eigen::Vector3f diff(pt.x - centroid.x(), pt.y - centroid.y(), pt.z - centroid.z());
        covariance += diff * diff.transpose();
    }
    covariance /= boundary_points.size();
    
    // 特征值分解
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver(covariance);
    Eigen::Matrix3f eigenvectors = solver.eigenvectors();
    
    // 最小特征值对应的特征向量作为法向量
    Eigen::Vector3f shell_normal = eigenvectors.col(0);
    shell_normal.normalize();
    
    return shell_normal;
}

bool OffsetBandProcessor::isPointInside(const PointT& point,
                                       const Eigen::Vector3f& shell_normal,
                                       const Eigen::Vector3f& reference_point) {
    
    Eigen::Vector3f point_vec(point.x, point.y, point.z);
    Eigen::Vector3f to_point = point_vec - reference_point;
    
    // 如果点在法向量指向的一侧，认为是外侧
    float dot_product = to_point.dot(shell_normal);
    return dot_product < 0.0f;  // 负值表示内侧
}

// ============================================================================
// 工厂实现
// ============================================================================

std::unique_ptr<AdaptiveDetailSelector> DetailSelectorFactory::create(
    SelectorType type,
    const DetailSelectorConfig& config) {
    
    // 目前所有类型都返回相同的实现
    // 未来可以根据类型返回不同的实现
    return std::make_unique<AdaptiveDetailSelector>(config);
}

} // namespace recon

