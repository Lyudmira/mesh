/**
 * 细节重建集成器实现
 */

#include "detail_reconstruction_integrator.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <chrono>
#include <fstream>
#include <filesystem>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace recon {

// ============================================================================
// DetailReconstructionIntegrator实现
// ============================================================================

DetailReconstructionIntegrator::DetailReconstructionIntegrator(
    const DetailIntegrationConfig& config)
    : config_(config), current_memory_usage_mb_(0) {
    
    if (!initializeComponents()) {
        throw std::runtime_error("细节重建集成器初始化失败");
    }
}

DetailReconstructionIntegrator::~DetailReconstructionIntegrator() {
    cleanup();
}

bool DetailReconstructionIntegrator::initializeComponents() {
    try {
        // 创建自适应选择器
        selector_ = std::make_unique<AdaptiveDetailSelector>(config_.selector_config);
        
        // 创建混合重建器
        reconstructor_ = std::make_unique<HybridReconstructor>(config_.reconstructor_config);
        
        // 初始化缓存
        if (config_.enable_caching) {
            selection_cache_.reserve(config_.max_cache_size);
            reconstruction_cache_.reserve(config_.max_cache_size);
        }
        
        // 创建调试输出目录
        if (config_.enable_debug_output) {
            std::filesystem::create_directories(config_.debug_output_dir);
        }
        
        std::cout << "细节重建集成器初始化完成" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "集成器初始化异常: " << e.what() << std::endl;
        return false;
    }
}

bool DetailReconstructionIntegrator::integrateDetailReconstruction(
    const PointCloudT::Ptr& cloud,
    const GridT::Ptr& shell_grid,
    DetailIntegrationResult& result) {
    
    start_time_ = std::chrono::high_resolution_clock::now();
    
    std::cout << "开始细节重建集成..." << std::endl;
    std::cout << "输入点云大小: " << cloud->size() << std::endl;
    
    try {
        // 1. 执行自适应选择
        std::cout << "执行自适应选择..." << std::endl;
        auto selection_start = std::chrono::high_resolution_clock::now();
        
        if (!performAdaptiveSelection(cloud, shell_grid, result.recommendations)) {
            result.success = false;
            result.error_message = "自适应选择失败";
            return false;
        }
        
        auto selection_end = std::chrono::high_resolution_clock::now();
        stats_.selection_time_seconds = 
            std::chrono::duration<double>(selection_end - selection_start).count();
        
        std::cout << "自适应选择完成，推荐数: " << result.recommendations.size() << std::endl;
        
        // 2. 验证推荐质量
        if (!validateRecommendations(result.recommendations)) {
            result.success = false;
            result.error_message = "推荐验证失败";
            return false;
        }
        
        // 3. 执行混合重建
        std::cout << "执行混合重建..." << std::endl;
        auto reconstruction_start = std::chrono::high_resolution_clock::now();
        
        if (!performHybridReconstruction(result.recommendations, shell_grid, 
                                       result.reconstruction_result)) {
            result.success = false;
            result.error_message = "混合重建失败";
            return false;
        }
        
        auto reconstruction_end = std::chrono::high_resolution_clock::now();
        stats_.reconstruction_time_seconds = 
            std::chrono::duration<double>(reconstruction_end - reconstruction_start).count();
        
        std::cout << "混合重建完成" << std::endl;
        
        // 4. 执行跨簇优化（如果启用）
        if (config_.enable_cross_cluster_optimization) {
            std::cout << "执行跨簇优化..." << std::endl;
            auto optimization_start = std::chrono::high_resolution_clock::now();
            
            result.optimized_clusters = result.reconstruction_result.clusters;
            performCrossClusterOptimization(result.optimized_clusters, shell_grid);
            
            auto optimization_end = std::chrono::high_resolution_clock::now();
            stats_.optimization_time_seconds = 
                std::chrono::duration<double>(optimization_end - optimization_start).count();
        }
        
        // 5. 质量反馈细化（如果启用）
        if (config_.enable_quality_feedback) {
            std::cout << "执行质量反馈细化..." << std::endl;
            
            auto& clusters_to_refine = config_.enable_cross_cluster_optimization ? 
                                     result.optimized_clusters : result.reconstruction_result.clusters;
            
            refineWithQualityFeedback(clusters_to_refine, shell_grid);
        }
        
        // 6. 评估整体质量
        float integration_quality = evaluateIntegrationQuality(
            result.reconstruction_result, *cloud);
        
        // 7. 更新统计信息
        updateStatistics(result);
        
        // 8. 保存调试信息
        if (config_.enable_debug_output) {
            saveDebugInfo(result);
        }
        
        // 9. 验证最终结果
        if (!validateResult(result)) {
            result.success = false;
            result.error_message = "结果验证失败";
            return false;
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        stats_.total_time_seconds = 
            std::chrono::duration<double>(end_time - start_time_).count();
        
        result.success = true;
        last_result_ = result;
        
        std::cout << "细节重建集成完成" << std::endl;
        std::cout << "统计信息:" << std::endl;
        std::cout << "  总时间: " << stats_.total_time_seconds << " 秒" << std::endl;
        std::cout << "  选择时间: " << stats_.selection_time_seconds << " 秒" << std::endl;
        std::cout << "  重建时间: " << stats_.reconstruction_time_seconds << " 秒" << std::endl;
        std::cout << "  优化时间: " << stats_.optimization_time_seconds << " 秒" << std::endl;
        std::cout << "  总簇数: " << stats_.total_clusters << std::endl;
        std::cout << "  成功簇数: " << stats_.successful_clusters << std::endl;
        std::cout << "  失败簇数: " << stats_.failed_clusters << std::endl;
        std::cout << "  平均质量: " << stats_.avg_cluster_quality << std::endl;
        std::cout << "  集成质量: " << integration_quality << std::endl;
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "细节重建集成异常: " << e.what() << std::endl;
        result.success = false;
        result.error_message = std::string("异常: ") + e.what();
        return false;
    }
}

bool DetailReconstructionIntegrator::performAdaptiveSelection(
    const PointCloudT::Ptr& cloud,
    const GridT::Ptr& shell_grid,
    std::vector<ReconstructionRecommendation>& recommendations) {
    
    // 检查缓存
    if (config_.enable_caching) {
        size_t cache_key = computeCacheKey(*cloud, *shell_grid);
        auto cache_it = selection_cache_.find(cache_key);
        
        if (cache_it != selection_cache_.end()) {
            recommendations = cache_it->second;
            stats_.cache_hit_count++;
            std::cout << "使用缓存的选择结果" << std::endl;
            return true;
        }
        stats_.cache_miss_count++;
    }
    
    // 执行自适应选择
    try {
        recommendations = selector_->analyzeAndRecommend(cloud, shell_grid);
        
        // 更新缓存
        if (config_.enable_caching && !recommendations.empty()) {
            size_t cache_key = computeCacheKey(*cloud, *shell_grid);
            selection_cache_[cache_key] = recommendations;
            
            // 清理过大的缓存
            if (selection_cache_.size() > config_.max_cache_size) {
                selection_cache_.clear();
            }
        }
        
        return !recommendations.empty();
        
    } catch (const std::exception& e) {
        std::cerr << "自适应选择异常: " << e.what() << std::endl;
        return false;
    }
}

bool DetailReconstructionIntegrator::performHybridReconstruction(
    const std::vector<ReconstructionRecommendation>& recommendations,
    const GridT::Ptr& shell_grid,
    HybridReconstructionResult& result) {
    
    // 转换推荐为重建簇
    std::vector<ReconstructionCluster> clusters = 
        convertRecommendationsToClusters(recommendations);
    
    if (clusters.empty()) {
        std::cerr << "没有有效的重建簇" << std::endl;
        return false;
    }
    
    // 优化簇分配
    if (config_.enable_cross_cluster_optimization) {
        optimizeClusterAssignment(clusters);
        balanceWorkload(clusters);
    }
    
    // 执行混合重建
    try {
        bool success = reconstructor_->reconstruct(clusters, *shell_grid, result);
        
        if (!success) {
            // 处理失败的簇
            handleFailedClusters(clusters, shell_grid);
            
            // 重新尝试重建
            success = reconstructor_->reconstruct(clusters, *shell_grid, result);
        }
        
        return success;
        
    } catch (const std::exception& e) {
        std::cerr << "混合重建异常: " << e.what() << std::endl;
        return false;
    }
}

bool DetailReconstructionIntegrator::performCrossClusterOptimization(
    std::vector<ReconstructionCluster>& clusters,
    const GridT::Ptr& shell_grid) {
    
    std::cout << "执行跨簇优化，簇数: " << clusters.size() << std::endl;
    
    // 1. 分析簇间关系
    for (size_t i = 0; i < clusters.size(); ++i) {
        for (size_t j = i + 1; j < clusters.size(); ++j) {
            // 检查簇间距离和相似性
            // 这里可以实现更复杂的簇间优化算法
        }
    }
    
    // 2. 优化边界连接
    for (auto& cluster : clusters) {
        // 优化与邻接簇的边界连接
        for (int neighbor_id : cluster.neighbor_clusters) {
            // 实现边界优化逻辑
        }
    }
    
    // 3. 全局一致性检查
    bool consistency_improved = false;
    
    // 简化的一致性检查
    for (auto& cluster : clusters) {
        if (cluster.mesh_quality_score < config_.min_cluster_quality) {
            // 尝试改进低质量簇
            // 这里可以实现质量改进算法
            consistency_improved = true;
        }
    }
    
    return true;
}

bool DetailReconstructionIntegrator::refineWithQualityFeedback(
    std::vector<ReconstructionCluster>& clusters,
    const GridT::Ptr& shell_grid) {
    
    std::cout << "执行质量反馈细化..." << std::endl;
    
    int refinement_count = 0;
    
    for (int iteration = 0; iteration < config_.max_refinement_iterations; ++iteration) {
        bool any_improvement = false;
        
        for (auto& cluster : clusters) {
            float initial_quality = cluster.mesh_quality_score;
            
            // 检查是否需要细化
            if (initial_quality < config_.min_cluster_quality) {
                std::cout << "细化簇 " << cluster.cluster_id 
                         << "，当前质量: " << initial_quality << std::endl;
                
                // 尝试不同的重建方法
                ReconstructionMethod original_method = cluster.method;
                
                // 尝试其他方法
                std::vector<ReconstructionMethod> alternative_methods = {
                    ReconstructionMethod::GP3,
                    ReconstructionMethod::POISSON,
                    ReconstructionMethod::RIMLS,
                    ReconstructionMethod::HYBRID
                };
                
                float best_quality = initial_quality;
                ReconstructionMethod best_method = original_method;
                pcl::PolygonMesh::Ptr best_mesh = cluster.mesh;
                
                for (auto method : alternative_methods) {
                    if (method == original_method) continue;
                    
                    cluster.method = method;
                    
                    // 重新重建
                    if (reconstructor_->reconstructCluster(cluster, *shell_grid)) {
                        if (cluster.mesh_quality_score > best_quality + 
                            config_.quality_improvement_threshold) {
                            
                            best_quality = cluster.mesh_quality_score;
                            best_method = method;
                            best_mesh = cluster.mesh;
                            any_improvement = true;
                        }
                    }
                }
                
                // 应用最佳结果
                cluster.method = best_method;
                cluster.mesh = best_mesh;
                cluster.mesh_quality_score = best_quality;
                
                if (best_quality > initial_quality) {
                    refinement_count++;
                    std::cout << "簇 " << cluster.cluster_id 
                             << " 质量改进: " << initial_quality 
                             << " -> " << best_quality << std::endl;
                }
            }
        }
        
        if (!any_improvement) {
            std::cout << "质量反馈细化收敛，迭代次数: " << iteration + 1 << std::endl;
            break;
        }
    }
    
    stats_.refined_clusters = refinement_count;
    std::cout << "质量反馈细化完成，改进簇数: " << refinement_count << std::endl;
    
    return true;
}

// ============================================================================
// 辅助方法实现
// ============================================================================

std::vector<ReconstructionCluster> 
DetailReconstructionIntegrator::convertRecommendationsToClusters(
    const std::vector<ReconstructionRecommendation>& recommendations) {
    
    std::vector<ReconstructionCluster> clusters;
    clusters.reserve(recommendations.size());
    
    for (size_t i = 0; i < recommendations.size(); ++i) {
        const auto& rec = recommendations[i];
        
        ReconstructionCluster cluster;
        cluster.cluster_id = static_cast<int>(i);
        cluster.method = rec.recommended_method;
        cluster.points = rec.cluster_points;
        cluster.quality = rec.quality_assessment;
        
        clusters.push_back(cluster);
    }
    
    return clusters;
}

bool DetailReconstructionIntegrator::validateRecommendations(
    const std::vector<ReconstructionRecommendation>& recommendations) {
    
    if (recommendations.empty()) {
        std::cerr << "推荐列表为空" << std::endl;
        return false;
    }
    
    for (const auto& rec : recommendations) {
        // 检查点云
        if (!rec.cluster_points || rec.cluster_points->empty()) {
            std::cerr << "推荐中包含空点云" << std::endl;
            return false;
        }
        
        // 检查质量评估
        if (rec.quality_assessment.overall_quality < 0.0f || 
            rec.quality_assessment.overall_quality > 1.0f) {
            std::cerr << "质量评估超出范围: " << rec.quality_assessment.overall_quality << std::endl;
            return false;
        }
        
        // 检查推荐方法
        if (rec.recommended_method == ReconstructionMethod::AUTO) {
            std::cerr << "推荐方法未确定" << std::endl;
            return false;
        }
    }
    
    return true;
}

bool DetailReconstructionIntegrator::optimizeClusterAssignment(
    std::vector<ReconstructionCluster>& clusters) {
    
    // 简化的簇分配优化
    // 基于点云大小和质量进行负载均衡
    
    std::sort(clusters.begin(), clusters.end(), 
        [](const ReconstructionCluster& a, const ReconstructionCluster& b) {
            return a.points->size() > b.points->size();
        });
    
    return true;
}

bool DetailReconstructionIntegrator::balanceWorkload(
    std::vector<ReconstructionCluster>& clusters) {
    
    // 简化的负载均衡
    // 根据点云大小和复杂度分配计算资源
    
    size_t total_points = 0;
    for (const auto& cluster : clusters) {
        total_points += cluster.points->size();
    }
    
    // 设置优先级
    for (auto& cluster : clusters) {
        float complexity_factor = static_cast<float>(cluster.points->size()) / total_points;
        // 可以根据复杂度调整重建参数
    }
    
    return true;
}

bool DetailReconstructionIntegrator::handleFailedClusters(
    std::vector<ReconstructionCluster>& clusters,
    const GridT::Ptr& shell_grid) {
    
    std::cout << "处理失败的簇..." << std::endl;
    
    int failed_count = 0;
    
    for (auto& cluster : clusters) {
        if (!cluster.mesh || cluster.mesh->polygons.empty()) {
            failed_count++;
            
            std::cout << "处理失败簇 " << cluster.cluster_id << std::endl;
            
            // 尝试简化的重建方法
            cluster.method = ReconstructionMethod::GP3;
            
            // 简化点云
            if (cluster.points->size() > 1000) {
                // 可以实现点云简化逻辑
            }
            
            // 重新尝试重建
            reconstructor_->reconstructCluster(cluster, *shell_grid);
        }
    }
    
    std::cout << "处理了 " << failed_count << " 个失败簇" << std::endl;
    return true;
}

float DetailReconstructionIntegrator::evaluateIntegrationQuality(
    const HybridReconstructionResult& result,
    const PointCloudT& original_cloud) {
    
    if (!result.success || !result.final_mesh) {
        return 0.0f;
    }
    
    // 综合质量评估
    float quality_score = 0.0f;
    
    // 1. 重建完整性 (30%)
    float completeness = static_cast<float>(result.successful_clusters) / 
                        std::max(1, result.total_clusters);
    quality_score += 0.3f * completeness;
    
    // 2. 平均簇质量 (40%)
    quality_score += 0.4f * result.overall_quality_score;
    
    // 3. 网格质量 (30%)
    float mesh_quality = 0.0f;
    if (result.final_vertex_count > 0 && result.final_face_count > 0) {
        // 简化的网格质量评估
        float vertex_density = static_cast<float>(result.final_vertex_count) / original_cloud.size();
        mesh_quality = std::exp(-std::abs(vertex_density - 1.0f));
    }
    quality_score += 0.3f * mesh_quality;
    
    return std::min(quality_score, 1.0f);
}

// ============================================================================
// 缓存和优化方法
// ============================================================================

size_t DetailReconstructionIntegrator::computeCacheKey(
    const PointCloudT& cloud, const GridT& grid) {
    
    // 简化的缓存键计算
    size_t key = 0;
    
    // 基于点云大小和网格信息
    key ^= std::hash<size_t>{}(cloud.size());
    key ^= std::hash<std::string>{}(grid.getName());
    
    return key;
}

bool DetailReconstructionIntegrator::checkCache(
    size_t key, HybridReconstructionResult& result) {
    
    auto it = reconstruction_cache_.find(key);
    if (it != reconstruction_cache_.end()) {
        result = it->second;
        return true;
    }
    
    return false;
}

void DetailReconstructionIntegrator::updateCache(
    size_t key, const HybridReconstructionResult& result) {
    
    if (reconstruction_cache_.size() >= config_.max_cache_size) {
        reconstruction_cache_.clear();
    }
    
    reconstruction_cache_[key] = result;
}

void DetailReconstructionIntegrator::updateStatistics(
    const DetailIntegrationResult& result) {
    
    stats_.total_clusters = result.reconstruction_result.total_clusters;
    stats_.successful_clusters = 0;
    stats_.failed_clusters = 0;
    
    stats_.gp3_clusters = result.reconstruction_result.gp3_clusters;
    stats_.poisson_clusters = result.reconstruction_result.poisson_clusters;
    stats_.rimls_clusters = result.reconstruction_result.rimls_clusters;
    stats_.hybrid_clusters = result.reconstruction_result.hybrid_clusters;
    
    // 计算质量统计
    if (!result.reconstruction_result.clusters.empty()) {
        float total_quality = 0.0f;
        stats_.min_cluster_quality = 1.0f;
        stats_.max_cluster_quality = 0.0f;
        
        for (const auto& cluster : result.reconstruction_result.clusters) {
            if (cluster.mesh && !cluster.mesh->polygons.empty()) {
                stats_.successful_clusters++;
            } else {
                stats_.failed_clusters++;
            }
            
            total_quality += cluster.mesh_quality_score;
            stats_.min_cluster_quality = std::min(stats_.min_cluster_quality, 
                                                 cluster.mesh_quality_score);
            stats_.max_cluster_quality = std::max(stats_.max_cluster_quality, 
                                                 cluster.mesh_quality_score);
        }
        
        stats_.avg_cluster_quality = total_quality / result.reconstruction_result.clusters.size();
    }
    
    // 网格统计
    stats_.total_vertices = result.reconstruction_result.final_vertex_count;
    stats_.total_faces = result.reconstruction_result.final_face_count;
    
    if (stats_.total_vertices > 0) {
        stats_.mesh_density = static_cast<float>(stats_.total_faces) / stats_.total_vertices;
    }
}

bool DetailReconstructionIntegrator::validateResult(
    const DetailIntegrationResult& result) {
    
    if (!result.success) {
        return false;
    }
    
    if (result.recommendations.empty()) {
        std::cerr << "结果中没有推荐" << std::endl;
        return false;
    }
    
    if (!result.reconstruction_result.final_mesh) {
        std::cerr << "结果中没有最终网格" << std::endl;
        return false;
    }
    
    if (result.reconstruction_result.final_mesh->polygons.empty()) {
        std::cerr << "最终网格为空" << std::endl;
        return false;
    }
    
    return true;
}

void DetailReconstructionIntegrator::saveDebugInfo(
    const DetailIntegrationResult& result) {
    
    if (!config_.enable_debug_output) {
        return;
    }
    
    try {
        std::string debug_file = config_.debug_output_dir + "/integration_debug.txt";
        std::ofstream file(debug_file);
        
        if (file.is_open()) {
            file << "细节重建集成调试信息\n";
            file << "========================\n\n";
            
            file << "统计信息:\n";
            file << "  总时间: " << stats_.total_time_seconds << " 秒\n";
            file << "  总簇数: " << stats_.total_clusters << "\n";
            file << "  成功簇数: " << stats_.successful_clusters << "\n";
            file << "  失败簇数: " << stats_.failed_clusters << "\n";
            file << "  平均质量: " << stats_.avg_cluster_quality << "\n";
            
            file << "\n推荐信息:\n";
            for (size_t i = 0; i < result.recommendations.size(); ++i) {
                const auto& rec = result.recommendations[i];
                file << "  推荐 " << i << ":\n";
                file << "    方法: " << static_cast<int>(rec.recommended_method) << "\n";
                file << "    质量: " << rec.quality_assessment.overall_quality << "\n";
                file << "    点数: " << (rec.cluster_points ? rec.cluster_points->size() : 0) << "\n";
            }
            
            file.close();
            std::cout << "调试信息已保存到: " << debug_file << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "保存调试信息失败: " << e.what() << std::endl;
    }
}

void DetailReconstructionIntegrator::setConfig(const DetailIntegrationConfig& config) {
    config_ = config;
    
    // 更新子组件配置
    if (selector_) {
        // selector_->setConfig(config_.selector_config);
    }
    
    if (reconstructor_) {
        reconstructor_->setConfig(config_.reconstructor_config);
    }
}

void DetailReconstructionIntegrator::clearCache() {
    selection_cache_.clear();
    reconstruction_cache_.clear();
    std::cout << "缓存已清理" << std::endl;
}

void DetailReconstructionIntegrator::cleanup() {
    clearCache();
    
    selector_.reset();
    reconstructor_.reset();
}

// ============================================================================
// DetailIntegratorFactory实现
// ============================================================================

std::unique_ptr<DetailReconstructionIntegrator> 
DetailIntegratorFactory::createStandardIntegrator() {
    DetailIntegrationConfig config;
    // 使用默认配置
    return std::make_unique<DetailReconstructionIntegrator>(config);
}

std::unique_ptr<DetailReconstructionIntegrator> 
DetailIntegratorFactory::createHighQualityIntegrator() {
    DetailIntegrationConfig config;
    
    // 高质量配置
    config.enable_quality_feedback = true;
    config.enable_adaptive_refinement = true;
    config.enable_cross_cluster_optimization = true;
    
    config.min_cluster_quality = 0.5f;
    config.quality_improvement_threshold = 0.05f;
    config.max_refinement_iterations = 5;
    
    // 高质量重建器配置
    config.reconstructor_config = *HybridReconstructorFactory::createHighQualityReconstructor()->getConfig();
    
    return std::make_unique<DetailReconstructionIntegrator>(config);
}

std::unique_ptr<DetailReconstructionIntegrator> 
DetailIntegratorFactory::createFastIntegrator() {
    DetailIntegrationConfig config;
    
    // 快速配置
    config.enable_quality_feedback = false;
    config.enable_adaptive_refinement = false;
    config.enable_cross_cluster_optimization = false;
    
    config.max_refinement_iterations = 1;
    
    // 快速重建器配置
    config.reconstructor_config = *HybridReconstructorFactory::createFastReconstructor()->getConfig();
    
    return std::make_unique<DetailReconstructionIntegrator>(config);
}

std::unique_ptr<DetailReconstructionIntegrator> 
DetailIntegratorFactory::createMemoryOptimizedIntegrator() {
    DetailIntegrationConfig config;
    
    // 内存优化配置
    config.enable_caching = false;
    config.max_cache_size = 100;
    
    // 内存优化重建器配置
    config.reconstructor_config = *HybridReconstructorFactory::createMemoryOptimizedReconstructor()->getConfig();
    
    return std::make_unique<DetailReconstructionIntegrator>(config);
}

std::unique_ptr<DetailReconstructionIntegrator> 
DetailIntegratorFactory::createDebugIntegrator() {
    DetailIntegrationConfig config;
    
    // 调试配置
    config.enable_debug_output = true;
    config.save_intermediate_results = true;
    config.debug_output_dir = "./debug_integration";
    
    return std::make_unique<DetailReconstructionIntegrator>(config);
}

} // namespace recon

