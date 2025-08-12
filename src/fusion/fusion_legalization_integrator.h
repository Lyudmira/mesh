/**
 * 融合与合法化集成器
 * 统一管理Alpha包装、平面对齐、交线重投影和合法化处理
 * 
 * 版本: 1.0
 * 日期: 2025-08-12
 */

#ifndef FUSION_LEGALIZATION_INTEGRATOR_H
#define FUSION_LEGALIZATION_INTEGRATOR_H

#include "alpha_wrapping_fusion.h"
#include "legalization_processor.h"
#include "../detail_layer/detail_reconstruction_integrator.h"
#include <memory>
#include <vector>
#include <unordered_map>
#include <openvdb/openvdb.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/PolygonMesh.h>

namespace recon {

/**
 * 融合合法化配置
 */
struct FusionLegalizationConfig {
    // Alpha包装配置
    AlphaWrappingConfig alpha_wrapping_config;
    
    // 合法化配置
    LegalizationConfig legalization_config;
    
    // 集成参数
    bool enable_alpha_wrapping = true;      // 启用Alpha包装
    bool enable_plane_alignment = true;     // 启用平面对齐
    bool enable_line_projection = true;     // 启用交线重投影
    bool enable_legalization = true;        // 启用合法化
    bool enable_quality_validation = true;  // 启用质量验证
    
    // 融合策略
    enum FusionStrategy {
        SHELL_PRIORITY,     // 外壳优先
        DETAIL_PRIORITY,    // 细节优先
        BALANCED,           // 平衡融合
        ADAPTIVE            // 自适应融合
    } fusion_strategy = BALANCED;
    
    // 质量控制
    double min_fusion_quality = 0.5;       // 最小融合质量
    double quality_improvement_threshold = 0.1; // 质量改进阈值
    int max_fusion_iterations = 3;          // 最大融合迭代次数
    
    // 几何约束
    bool enforce_building_constraints = true; // 强制建筑约束
    bool maintain_detail_features = true;   // 保持细节特征
    bool ensure_structural_integrity = true; // 确保结构完整性
    
    // 性能参数
    bool use_parallel_processing = true;    // 并行处理
    int num_threads = 0;                    // 线程数
    bool enable_progressive_fusion = true;  // 启用渐进式融合
    bool enable_caching = true;             // 启用缓存
    size_t max_cache_size = 1000;           // 最大缓存大小
    
    // 调试参数
    bool enable_debug_output = false;       // 调试输出
    bool save_intermediate_meshes = false;  // 保存中间网格
    std::string debug_output_dir = "./debug_fusion"; // 调试输出目录
};

/**
 * 融合阶段枚举
 */
enum class FusionStage {
    PREPARATION,        // 准备阶段
    ALPHA_WRAPPING,     // Alpha包装
    PLANE_ALIGNMENT,    // 平面对齐
    LINE_PROJECTION,    // 交线重投影
    DETAIL_FUSION,      // 细节融合
    LEGALIZATION,       // 合法化
    QUALITY_VALIDATION, // 质量验证
    FINALIZATION        // 最终化
};

/**
 * 融合合法化统计信息
 */
struct FusionLegalizationStats {
    // 时间统计
    double total_time = 0.0;
    double preparation_time = 0.0;
    double alpha_wrapping_time = 0.0;
    double plane_alignment_time = 0.0;
    double line_projection_time = 0.0;
    double detail_fusion_time = 0.0;
    double legalization_time = 0.0;
    double validation_time = 0.0;
    
    // 网格统计
    int input_shell_vertices = 0;
    int input_shell_faces = 0;
    int input_detail_vertices = 0;
    int input_detail_faces = 0;
    int final_vertices = 0;
    int final_faces = 0;
    
    // 处理统计
    int detected_planes = 0;
    int projected_lines = 0;
    int fused_details = 0;
    int fixed_violations = 0;
    int remaining_violations = 0;
    
    // 质量统计
    double initial_quality = 0.0;
    double post_wrapping_quality = 0.0;
    double post_alignment_quality = 0.0;
    double post_fusion_quality = 0.0;
    double final_quality = 0.0;
    double overall_quality_improvement = 0.0;
    
    // 迭代统计
    int fusion_iterations = 0;
    bool converged = false;
    
    // 内存统计
    size_t peak_memory_usage_mb = 0;
    size_t cache_hit_count = 0;
    size_t cache_miss_count = 0;
};

/**
 * 融合合法化结果
 */
struct FusionLegalizationResult {
    // 最终结果
    pcl::PolygonMesh final_mesh;           // 最终网格
    
    // 中间结果
    pcl::PolygonMesh wrapped_mesh;         // Alpha包装后的网格
    pcl::PolygonMesh aligned_mesh;         // 平面对齐后的网格
    pcl::PolygonMesh projected_mesh;       // 交线投影后的网格
    pcl::PolygonMesh fused_mesh;           // 细节融合后的网格
    pcl::PolygonMesh legalized_mesh;       // 合法化后的网格
    
    // 子结果
    AlphaWrappingResult alpha_wrapping_result;
    LegalizationResult legalization_result;
    
    // 统计信息
    FusionLegalizationStats stats;
    
    // 状态信息
    bool success = false;
    FusionStage failed_stage = FusionStage::PREPARATION;
    std::string error_message;
    std::vector<std::string> warnings;
    
    // 质量信息
    std::vector<ConstraintViolation> quality_issues;
    double overall_quality_score = 0.0;
};

/**
 * 融合与合法化集成器
 */
class FusionLegalizationIntegrator {
public:
    using PointT = pcl::PointXYZRGBNormal;
    using PointCloudT = pcl::PointCloud<PointT>;
    using GridT = openvdb::FloatGrid;

    /**
     * 构造函数
     */
    explicit FusionLegalizationIntegrator(
        const FusionLegalizationConfig& config = FusionLegalizationConfig{}
    );
    
    /**
     * 析构函数
     */
    ~FusionLegalizationIntegrator();
    
    /**
     * 执行完整的融合与合法化
     */
    bool performFusionAndLegalization(
        const pcl::PolygonMesh& shell_mesh,
        const HybridReconstructionResult& detail_result,
        FusionLegalizationResult& result
    );
    
    /**
     * 执行Alpha包装融合
     */
    bool performAlphaWrappingFusion(
        const pcl::PolygonMesh& shell_mesh,
        const pcl::PolygonMesh& detail_mesh,
        pcl::PolygonMesh& wrapped_mesh,
        AlphaWrappingResult& wrapping_result
    );
    
    /**
     * 执行平面对齐
     */
    bool performPlaneAlignment(
        pcl::PolygonMesh& mesh,
        const std::vector<PlaneInfo>& planes
    );
    
    /**
     * 执行交线重投影
     */
    bool performLineProjection(
        pcl::PolygonMesh& mesh,
        const std::vector<IntersectionLine>& lines
    );
    
    /**
     * 执行细节融合
     */
    bool performDetailFusion(
        pcl::PolygonMesh& shell_mesh,
        const HybridReconstructionResult& detail_result,
        pcl::PolygonMesh& fused_mesh
    );
    
    /**
     * 执行合法化处理
     */
    bool performLegalization(
        pcl::PolygonMesh& mesh,
        LegalizationResult& legalization_result
    );
    
    /**
     * 执行质量验证
     */
    bool performQualityValidation(
        const pcl::PolygonMesh& mesh,
        std::vector<ConstraintViolation>& quality_issues
    );
    
    /**
     * 渐进式融合
     */
    bool performProgressiveFusion(
        const pcl::PolygonMesh& shell_mesh,
        const std::vector<ReconstructionCluster>& detail_clusters,
        pcl::PolygonMesh& fused_mesh
    );
    
    /**
     * 自适应融合策略选择
     */
    FusionLegalizationConfig::FusionStrategy selectOptimalStrategy(
        const pcl::PolygonMesh& shell_mesh,
        const HybridReconstructionResult& detail_result
    );
    
    /**
     * 评估融合质量
     */
    double evaluateFusionQuality(
        const pcl::PolygonMesh& mesh,
        const pcl::PolygonMesh& original_shell,
        const HybridReconstructionResult& detail_result
    );
    
    /**
     * 获取配置
     */
    const FusionLegalizationConfig& getConfig() const { return config_; }
    
    /**
     * 设置配置
     */
    void setConfig(const FusionLegalizationConfig& config);
    
    /**
     * 获取统计信息
     */
    const FusionLegalizationStats& getStats() const { return stats_; }
    
    /**
     * 获取最后结果
     */
    const FusionLegalizationResult& getLastResult() const { return last_result_; }
    
    /**
     * 清理缓存
     */
    void clearCache();

private:
    FusionLegalizationConfig config_;
    FusionLegalizationStats stats_;
    FusionLegalizationResult last_result_;
    
    // 核心组件
    std::unique_ptr<AlphaWrappingFusion> alpha_wrapper_;
    std::unique_ptr<LegalizationProcessor> legalizer_;
    
    // 缓存系统
    std::unordered_map<size_t, pcl::PolygonMesh> fusion_cache_;
    std::unordered_map<size_t, LegalizationResult> legalization_cache_;
    
    // 性能监控
    std::chrono::high_resolution_clock::time_point start_time_;
    size_t current_memory_usage_mb_;
    
    /**
     * 初始化组件
     */
    bool initializeComponents();
    
    /**
     * 准备输入数据
     */
    bool prepareInputData(
        const pcl::PolygonMesh& shell_mesh,
        const HybridReconstructionResult& detail_result,
        pcl::PolygonMesh& prepared_shell,
        pcl::PolygonMesh& prepared_detail
    );
    
    /**
     * 验证输入数据
     */
    bool validateInputData(
        const pcl::PolygonMesh& shell_mesh,
        const HybridReconstructionResult& detail_result
    );
    
    /**
     * 合并细节网格
     */
    bool mergeDetailMeshes(
        const std::vector<ReconstructionCluster>& clusters,
        pcl::PolygonMesh& merged_mesh
    );
    
    /**
     * 局部细节融合
     */
    bool performLocalDetailFusion(
        pcl::PolygonMesh& shell_mesh,
        const ReconstructionCluster& detail_cluster,
        const Eigen::Vector3d& fusion_center,
        double fusion_radius
    );
    
    /**
     * 边界混合
     */
    bool performBoundaryBlending(
        pcl::PolygonMesh& mesh,
        const std::vector<int>& boundary_vertices,
        double blending_radius
    );
    
    /**
     * 特征保持
     */
    bool preserveFeatures(
        pcl::PolygonMesh& mesh,
        const std::vector<int>& feature_vertices,
        const std::vector<int>& feature_edges
    );
    
    /**
     * 迭代质量改进
     */
    bool performIterativeQualityImprovement(
        pcl::PolygonMesh& mesh,
        double target_quality
    );
    
    /**
     * 计算缓存键
     */
    size_t computeCacheKey(
        const pcl::PolygonMesh& shell_mesh,
        const HybridReconstructionResult& detail_result
    );
    
    /**
     * 检查缓存
     */
    bool checkFusionCache(size_t key, pcl::PolygonMesh& result);
    
    /**
     * 更新缓存
     */
    void updateFusionCache(size_t key, const pcl::PolygonMesh& result);
    
    /**
     * 监控内存使用
     */
    void monitorMemoryUsage();
    
    /**
     * 更新统计信息
     */
    void updateStatistics(
        const FusionLegalizationResult& result,
        FusionStage current_stage
    );
    
    /**
     * 保存中间结果
     */
    void saveIntermediateResult(
        const pcl::PolygonMesh& mesh,
        const std::string& stage_name
    );
    
    /**
     * 保存调试信息
     */
    void saveDebugInfo(const FusionLegalizationResult& result);
    
    /**
     * 验证结果
     */
    bool validateResult(const FusionLegalizationResult& result);
    
    /**
     * 清理资源
     */
    void cleanup();
};

/**
 * 融合合法化集成器工厂
 */
class FusionLegalizationIntegratorFactory {
public:
    /**
     * 创建标准集成器
     */
    static std::unique_ptr<FusionLegalizationIntegrator> createStandardIntegrator();
    
    /**
     * 创建高质量集成器
     */
    static std::unique_ptr<FusionLegalizationIntegrator> createHighQualityIntegrator();
    
    /**
     * 创建快速集成器
     */
    static std::unique_ptr<FusionLegalizationIntegrator> createFastIntegrator();
    
    /**
     * 创建建筑专用集成器
     */
    static std::unique_ptr<FusionLegalizationIntegrator> createArchitecturalIntegrator();
    
    /**
     * 创建工程专用集成器
     */
    static std::unique_ptr<FusionLegalizationIntegrator> createEngineeringIntegrator();
    
    /**
     * 创建调试集成器
     */
    static std::unique_ptr<FusionLegalizationIntegrator> createDebugIntegrator();
};

} // namespace recon

#endif // FUSION_LEGALIZATION_INTEGRATOR_H

