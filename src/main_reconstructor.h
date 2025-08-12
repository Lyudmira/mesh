/**
 * 主重建器
 * 统一集成所有重建模块，提供完整的3D重建解决方案
 * 
 * 版本: 2.0 - 完整集成版本
 * 日期: 2025-08-12
 */

#ifndef MAIN_RECONSTRUCTOR_H
#define MAIN_RECONSTRUCTOR_H

#include "udf_builder/enhanced_udf_builder.h"
#include "graph_cut/enhanced_graph_cut.h"
#include "integration/udf_graphcut_integrator.h"
#include "detail_layer/detail_reconstruction_integrator.h"
#include "fusion/fusion_legalization_integrator.h"
#include <memory>
#include <vector>
#include <string>
#include <chrono>
#include <openvdb/openvdb.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/PolygonMesh.h>

namespace recon {

/**
 * 主重建器配置
 */
struct MainReconstructorConfig {
    // 子模块配置
    EnhancedUDFBuilderConfig udf_config;
    UDFGraphCutIntegratorConfig integration_config;
    DetailIntegrationConfig detail_config;
    FusionLegalizationConfig fusion_config;
    
    // 全局参数
    bool enable_udf_building = true;        // 启用UDF构建
    bool enable_graph_cut = true;           // 启用图割
    bool enable_detail_reconstruction = true; // 启用细节重建
    bool enable_fusion_legalization = true; // 启用融合合法化
    
    // 质量控制
    double min_overall_quality = 0.6;      // 最小整体质量
    int max_global_iterations = 5;         // 最大全局迭代次数
    double convergence_threshold = 0.01;   // 收敛阈值
    
    // 性能参数
    bool use_parallel_processing = true;   // 并行处理
    int num_threads = 0;                   // 线程数
    bool enable_progressive_reconstruction = true; // 渐进式重建
    bool enable_memory_optimization = true; // 内存优化
    
    // 输出控制
    bool save_intermediate_results = false; // 保存中间结果
    bool enable_detailed_logging = true;   // 详细日志
    std::string output_directory = "./output"; // 输出目录
    
    // 调试参数
    bool enable_debug_mode = false;        // 调试模式
    bool enable_performance_profiling = true; // 性能分析
    std::string debug_output_dir = "./debug_main"; // 调试输出目录
};

/**
 * 重建阶段枚举
 */
enum class ReconstructionStage {
    INITIALIZATION,     // 初始化
    UDF_BUILDING,       // UDF构建
    GRAPH_CUT,          // 图割
    DETAIL_RECONSTRUCTION, // 细节重建
    FUSION_LEGALIZATION,   // 融合合法化
    FINALIZATION,       // 最终化
    COMPLETED           // 完成
};

/**
 * 主重建器统计信息
 */
struct MainReconstructorStats {
    // 时间统计
    double total_time = 0.0;
    double initialization_time = 0.0;
    double udf_building_time = 0.0;
    double graph_cut_time = 0.0;
    double detail_reconstruction_time = 0.0;
    double fusion_legalization_time = 0.0;
    double finalization_time = 0.0;
    
    // 数据统计
    int input_points = 0;
    int shell_vertices = 0;
    int shell_faces = 0;
    int detail_vertices = 0;
    int detail_faces = 0;
    int final_vertices = 0;
    int final_faces = 0;
    
    // 处理统计
    int udf_voxels = 0;
    int graph_cut_nodes = 0;
    int detail_clusters = 0;
    int fusion_operations = 0;
    int legalization_fixes = 0;
    
    // 质量统计
    double initial_quality = 0.0;
    double shell_quality = 0.0;
    double detail_quality = 0.0;
    double fusion_quality = 0.0;
    double final_quality = 0.0;
    double overall_improvement = 0.0;
    
    // 迭代统计
    int global_iterations = 0;
    bool converged = false;
    
    // 内存统计
    size_t peak_memory_usage_mb = 0;
    size_t current_memory_usage_mb = 0;
};

/**
 * 主重建器结果
 */
struct MainReconstructorResult {
    // 最终结果
    pcl::PolygonMesh final_mesh;           // 最终重建网格
    openvdb::FloatGrid::Ptr final_grid;    // 最终UDF网格
    
    // 中间结果
    pcl::PolygonMesh shell_mesh;           // 外壳网格
    HybridReconstructionResult detail_result; // 细节重建结果
    FusionLegalizationResult fusion_result;   // 融合合法化结果
    
    // 统计信息
    MainReconstructorStats stats;
    
    // 状态信息
    bool success = false;
    ReconstructionStage failed_stage = ReconstructionStage::INITIALIZATION;
    std::string error_message;
    std::vector<std::string> warnings;
    
    // 质量评估
    double overall_quality_score = 0.0;
    std::vector<std::string> quality_issues;
    std::vector<std::string> recommendations;
};

/**
 * 主重建器
 */
class MainReconstructor {
public:
    using PointT = pcl::PointXYZRGBNormal;
    using PointCloudT = pcl::PointCloud<PointT>;
    using GridT = openvdb::FloatGrid;

    /**
     * 构造函数
     */
    explicit MainReconstructor(const MainReconstructorConfig& config = MainReconstructorConfig{});
    
    /**
     * 析构函数
     */
    ~MainReconstructor();
    
    /**
     * 执行完整的3D重建
     */
    bool performReconstruction(
        const PointCloudT::Ptr& input_cloud,
        MainReconstructorResult& result
    );
    
    /**
     * 执行UDF构建阶段
     */
    bool performUDFBuilding(
        const PointCloudT::Ptr& input_cloud,
        GridT::Ptr& udf_grid
    );
    
    /**
     * 执行图割阶段
     */
    bool performGraphCut(
        const PointCloudT::Ptr& input_cloud,
        const GridT::Ptr& udf_grid,
        pcl::PolygonMesh& shell_mesh
    );
    
    /**
     * 执行细节重建阶段
     */
    bool performDetailReconstruction(
        const PointCloudT::Ptr& input_cloud,
        const GridT::Ptr& udf_grid,
        HybridReconstructionResult& detail_result
    );
    
    /**
     * 执行融合合法化阶段
     */
    bool performFusionLegalization(
        const pcl::PolygonMesh& shell_mesh,
        const HybridReconstructionResult& detail_result,
        FusionLegalizationResult& fusion_result
    );
    
    /**
     * 渐进式重建
     */
    bool performProgressiveReconstruction(
        const PointCloudT::Ptr& input_cloud,
        MainReconstructorResult& result
    );
    
    /**
     * 迭代质量改进
     */
    bool performIterativeImprovement(
        const PointCloudT::Ptr& input_cloud,
        MainReconstructorResult& result,
        double target_quality
    );
    
    /**
     * 评估重建质量
     */
    double evaluateReconstructionQuality(
        const MainReconstructorResult& result,
        const PointCloudT& original_cloud
    );
    
    /**
     * 生成质量报告
     */
    bool generateQualityReport(
        const MainReconstructorResult& result,
        const std::string& report_path
    );
    
    /**
     * 保存重建结果
     */
    bool saveReconstructionResult(
        const MainReconstructorResult& result,
        const std::string& output_path
    );
    
    /**
     * 获取配置
     */
    const MainReconstructorConfig& getConfig() const { return config_; }
    
    /**
     * 设置配置
     */
    void setConfig(const MainReconstructorConfig& config);
    
    /**
     * 获取统计信息
     */
    const MainReconstructorStats& getStats() const { return stats_; }
    
    /**
     * 获取最后结果
     */
    const MainReconstructorResult& getLastResult() const { return last_result_; }
    
    /**
     * 获取当前阶段
     */
    ReconstructionStage getCurrentStage() const { return current_stage_; }
    
    /**
     * 获取进度百分比
     */
    double getProgressPercentage() const;
    
    /**
     * 是否正在运行
     */
    bool isRunning() const { return is_running_; }
    
    /**
     * 停止重建
     */
    void stopReconstruction() { should_stop_ = true; }
    
    /**
     * 重置状态
     */
    void reset();

private:
    MainReconstructorConfig config_;
    MainReconstructorStats stats_;
    MainReconstructorResult last_result_;
    
    // 当前状态
    ReconstructionStage current_stage_;
    bool is_running_;
    bool should_stop_;
    
    // 核心组件
    std::unique_ptr<EnhancedUDFBuilder> udf_builder_;
    std::unique_ptr<UDFGraphCutIntegrator> graph_cut_integrator_;
    std::unique_ptr<DetailReconstructionIntegrator> detail_integrator_;
    std::unique_ptr<FusionLegalizationIntegrator> fusion_integrator_;
    
    // 性能监控
    std::chrono::high_resolution_clock::time_point start_time_;
    std::chrono::high_resolution_clock::time_point stage_start_time_;
    
    /**
     * 初始化组件
     */
    bool initializeComponents();
    
    /**
     * 验证输入数据
     */
    bool validateInputData(const PointCloudT& input_cloud);
    
    /**
     * 预处理点云
     */
    bool preprocessPointCloud(
        const PointCloudT::Ptr& input_cloud,
        PointCloudT::Ptr& processed_cloud
    );
    
    /**
     * 检查收敛性
     */
    bool checkConvergence(
        const MainReconstructorResult& current_result,
        const MainReconstructorResult& previous_result
    );
    
    /**
     * 更新阶段
     */
    void updateStage(ReconstructionStage new_stage);
    
    /**
     * 监控内存使用
     */
    void monitorMemoryUsage();
    
    /**
     * 更新统计信息
     */
    void updateStatistics(const MainReconstructorResult& result);
    
    /**
     * 记录阶段时间
     */
    void recordStageTime(ReconstructionStage stage, double time);
    
    /**
     * 保存中间结果
     */
    void saveIntermediateResult(
        const std::string& stage_name,
        const pcl::PolygonMesh& mesh
    );
    
    /**
     * 保存调试信息
     */
    void saveDebugInfo(const MainReconstructorResult& result);
    
    /**
     * 生成推荐
     */
    std::vector<std::string> generateRecommendations(
        const MainReconstructorResult& result
    );
    
    /**
     * 验证结果
     */
    bool validateResult(const MainReconstructorResult& result);
    
    /**
     * 清理资源
     */
    void cleanup();
};

/**
 * 主重建器工厂
 */
class MainReconstructorFactory {
public:
    /**
     * 创建标准重建器
     */
    static std::unique_ptr<MainReconstructor> createStandardReconstructor();
    
    /**
     * 创建高质量重建器
     */
    static std::unique_ptr<MainReconstructor> createHighQualityReconstructor();
    
    /**
     * 创建快速重建器
     */
    static std::unique_ptr<MainReconstructor> createFastReconstructor();
    
    /**
     * 创建内存优化重建器
     */
    static std::unique_ptr<MainReconstructor> createMemoryOptimizedReconstructor();
    
    /**
     * 创建建筑专用重建器
     */
    static std::unique_ptr<MainReconstructor> createArchitecturalReconstructor();
    
    /**
     * 创建调试重建器
     */
    static std::unique_ptr<MainReconstructor> createDebugReconstructor();
};

/**
 * 重建器管理器
 * 支持多个重建任务的并行管理
 */
class ReconstructorManager {
public:
    /**
     * 添加重建任务
     */
    int addReconstructionTask(
        const PointCloudT::Ptr& input_cloud,
        const MainReconstructorConfig& config,
        const std::string& task_name = ""
    );
    
    /**
     * 执行所有任务
     */
    bool executeAllTasks();
    
    /**
     * 获取任务状态
     */
    ReconstructionStage getTaskStatus(int task_id);
    
    /**
     * 获取任务进度
     */
    double getTaskProgress(int task_id);
    
    /**
     * 获取任务结果
     */
    bool getTaskResult(int task_id, MainReconstructorResult& result);
    
    /**
     * 停止任务
     */
    bool stopTask(int task_id);
    
    /**
     * 停止所有任务
     */
    void stopAllTasks();
    
    /**
     * 清理完成的任务
     */
    void cleanupCompletedTasks();
    
    /**
     * 获取活跃任务数
     */
    int getActiveTaskCount() const;

private:
    struct ReconstructionTask {
        int task_id;
        std::string task_name;
        PointCloudT::Ptr input_cloud;
        std::unique_ptr<MainReconstructor> reconstructor;
        MainReconstructorResult result;
        std::chrono::high_resolution_clock::time_point start_time;
        bool completed = false;
    };
    
    std::vector<std::unique_ptr<ReconstructionTask>> tasks_;
    std::atomic<int> next_task_id_{0};
    mutable std::mutex tasks_mutex_;
};

} // namespace recon

#endif // MAIN_RECONSTRUCTOR_H

