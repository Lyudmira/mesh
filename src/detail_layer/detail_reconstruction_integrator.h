/**
 * 细节重建集成器
 * 驱动自适应选择器和混合重建器协同工作
 * 
 * 版本: 1.0
 * 日期: 2025-08-12
 */

#ifndef DETAIL_RECONSTRUCTION_INTEGRATOR_H
#define DETAIL_RECONSTRUCTION_INTEGRATOR_H

#include "adaptive_detail_selector.h"
#include "hybrid_reconstructor.h"
#include <memory>
#include <vector>
#include <unordered_map>
#include <openvdb/openvdb.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/PolygonMesh.h>

namespace recon {

/**
 * 集成配置参数
 */
struct DetailIntegrationConfig {
    // 选择器配置
    DetailSelectorConfig selector_config;
    
    // 重建器配置
    HybridReconstructorConfig reconstructor_config;
    
    // 集成参数
    bool enable_quality_feedback = true;       // 启用质量反馈
    bool enable_adaptive_refinement = true;    // 启用自适应细化
    bool enable_cross_cluster_optimization = true; // 启用跨簇优化
    
    // 质量控制参数
    float min_cluster_quality = 0.3f;          // 最小簇质量
    float quality_improvement_threshold = 0.1f; // 质量改进阈值
    int max_refinement_iterations = 3;          // 最大细化迭代次数
    
    // 性能参数
    bool use_parallel_processing = true;       // 并行处理
    int num_threads = 0;                       // 线程数
    bool enable_caching = true;                // 启用缓存
    size_t max_cache_size = 1000;              // 最大缓存大小
    
    // 调试参数
    bool enable_debug_output = false;          // 调试输出
    bool save_intermediate_results = false;    // 保存中间结果
    std::string debug_output_dir = "./debug";  // 调试输出目录
};

/**
 * 集成统计信息
 */
struct DetailIntegrationStats {
    // 时间统计
    double total_time_seconds = 0.0;
    double selection_time_seconds = 0.0;
    double reconstruction_time_seconds = 0.0;
    double optimization_time_seconds = 0.0;
    
    // 簇统计
    int total_clusters = 0;
    int successful_clusters = 0;
    int failed_clusters = 0;
    int refined_clusters = 0;
    
    // 方法统计
    int gp3_clusters = 0;
    int poisson_clusters = 0;
    int rimls_clusters = 0;
    int hybrid_clusters = 0;
    
    // 质量统计
    float avg_cluster_quality = 0.0f;
    float min_cluster_quality = 1.0f;
    float max_cluster_quality = 0.0f;
    float overall_quality_improvement = 0.0f;
    
    // 网格统计
    int total_vertices = 0;
    int total_faces = 0;
    float mesh_density = 0.0f;
    
    // 内存统计
    size_t peak_memory_usage_mb = 0;
    size_t cache_hit_count = 0;
    size_t cache_miss_count = 0;
};

/**
 * 细节重建集成结果
 */
struct DetailIntegrationResult {
    // 重建结果
    HybridReconstructionResult reconstruction_result;
    
    // 选择结果
    std::vector<ReconstructionRecommendation> recommendations;
    
    // 优化结果
    std::vector<ReconstructionCluster> optimized_clusters;
    
    // 统计信息
    DetailIntegrationStats stats;
    
    // 状态信息
    bool success = false;
    std::string error_message;
    std::vector<std::string> warnings;
};

/**
 * 细节重建集成器
 */
class DetailReconstructionIntegrator {
public:
    using PointT = pcl::PointXYZRGBNormal;
    using PointCloudT = pcl::PointCloud<PointT>;
    using GridT = openvdb::FloatGrid;

    /**
     * 构造函数
     */
    explicit DetailReconstructionIntegrator(
        const DetailIntegrationConfig& config = DetailIntegrationConfig{});
    
    /**
     * 析构函数
     */
    ~DetailReconstructionIntegrator();
    
    /**
     * 执行完整的细节重建集成
     */
    bool integrateDetailReconstruction(
        const PointCloudT::Ptr& cloud,
        const GridT::Ptr& shell_grid,
        DetailIntegrationResult& result
    );
    
    /**
     * 执行自适应选择
     */
    bool performAdaptiveSelection(
        const PointCloudT::Ptr& cloud,
        const GridT::Ptr& shell_grid,
        std::vector<ReconstructionRecommendation>& recommendations
    );
    
    /**
     * 执行混合重建
     */
    bool performHybridReconstruction(
        const std::vector<ReconstructionRecommendation>& recommendations,
        const GridT::Ptr& shell_grid,
        HybridReconstructionResult& result
    );
    
    /**
     * 执行跨簇优化
     */
    bool performCrossClusterOptimization(
        std::vector<ReconstructionCluster>& clusters,
        const GridT::Ptr& shell_grid
    );
    
    /**
     * 质量反馈细化
     */
    bool refineWithQualityFeedback(
        std::vector<ReconstructionCluster>& clusters,
        const GridT::Ptr& shell_grid
    );
    
    /**
     * 评估集成质量
     */
    float evaluateIntegrationQuality(
        const HybridReconstructionResult& result,
        const PointCloudT& original_cloud
    );
    
    /**
     * 获取配置
     */
    const DetailIntegrationConfig& getConfig() const { return config_; }
    
    /**
     * 设置配置
     */
    void setConfig(const DetailIntegrationConfig& config);
    
    /**
     * 获取统计信息
     */
    const DetailIntegrationStats& getStats() const { return stats_; }
    
    /**
     * 获取最后结果
     */
    const DetailIntegrationResult& getLastResult() const { return last_result_; }
    
    /**
     * 清理缓存
     */
    void clearCache();

private:
    DetailIntegrationConfig config_;
    DetailIntegrationStats stats_;
    DetailIntegrationResult last_result_;
    
    // 核心组件
    std::unique_ptr<AdaptiveDetailSelector> selector_;
    std::unique_ptr<HybridReconstructor> reconstructor_;
    
    // 缓存系统
    std::unordered_map<size_t, std::vector<ReconstructionRecommendation>> selection_cache_;
    std::unordered_map<size_t, HybridReconstructionResult> reconstruction_cache_;
    
    // 性能监控
    std::chrono::high_resolution_clock::time_point start_time_;
    size_t current_memory_usage_mb_;
    
    /**
     * 初始化组件
     */
    bool initializeComponents();
    
    /**
     * 转换推荐为簇
     */
    std::vector<ReconstructionCluster> convertRecommendationsToClusters(
        const std::vector<ReconstructionRecommendation>& recommendations
    );
    
    /**
     * 验证推荐质量
     */
    bool validateRecommendations(
        const std::vector<ReconstructionRecommendation>& recommendations
    );
    
    /**
     * 优化簇分配
     */
    bool optimizeClusterAssignment(
        std::vector<ReconstructionCluster>& clusters
    );
    
    /**
     * 平衡负载
     */
    bool balanceWorkload(
        std::vector<ReconstructionCluster>& clusters
    );
    
    /**
     * 处理失败的簇
     */
    bool handleFailedClusters(
        std::vector<ReconstructionCluster>& clusters,
        const GridT::Ptr& shell_grid
    );
    
    /**
     * 计算缓存键
     */
    size_t computeCacheKey(const PointCloudT& cloud, const GridT& grid);
    
    /**
     * 检查缓存
     */
    bool checkCache(size_t key, HybridReconstructionResult& result);
    
    /**
     * 更新缓存
     */
    void updateCache(size_t key, const HybridReconstructionResult& result);
    
    /**
     * 监控内存使用
     */
    void monitorMemoryUsage();
    
    /**
     * 更新统计信息
     */
    void updateStatistics(const DetailIntegrationResult& result);
    
    /**
     * 保存调试信息
     */
    void saveDebugInfo(const DetailIntegrationResult& result);
    
    /**
     * 验证结果
     */
    bool validateResult(const DetailIntegrationResult& result);
    
    /**
     * 清理资源
     */
    void cleanup();
};

/**
 * 集成器工厂
 */
class DetailIntegratorFactory {
public:
    /**
     * 创建标准集成器
     */
    static std::unique_ptr<DetailReconstructionIntegrator> createStandardIntegrator();
    
    /**
     * 创建高质量集成器
     */
    static std::unique_ptr<DetailReconstructionIntegrator> createHighQualityIntegrator();
    
    /**
     * 创建快速集成器
     */
    static std::unique_ptr<DetailReconstructionIntegrator> createFastIntegrator();
    
    /**
     * 创建内存优化集成器
     */
    static std::unique_ptr<DetailReconstructionIntegrator> createMemoryOptimizedIntegrator();
    
    /**
     * 创建调试集成器
     */
    static std::unique_ptr<DetailReconstructionIntegrator> createDebugIntegrator();
};

} // namespace recon

#endif // DETAIL_RECONSTRUCTION_INTEGRATOR_H

