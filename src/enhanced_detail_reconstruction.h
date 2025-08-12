/**
 * 增强版细节重建模块头文件
 * 负责在外壳偏移带内重建高保真细节
 * 
 * 版本: 2.0 - 增强版本
 * 日期: 2025-08-12
 * 
 * 主要改进:
 * - 完整RIMLS算法实现
 * - 高级去噪算法 (WLOP, RIMLS去噪)
 * - 精确距离计算
 * - 自适应参数计算
 * - 并行处理优化
 * - 质量控制系统
 * - 网格融合增强
 */

#ifndef ENHANCED_DETAIL_RECONSTRUCTION_H
#define ENHANCED_DETAIL_RECONSTRUCTION_H

#include <memory>
#include <vector>
#include <unordered_map>
#include <atomic>
#include <mutex>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/PolygonMesh.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <Eigen/Dense>
#include <Eigen/SVD>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace recon {

/**
 * 增强版细节重建方法枚举
 */
enum class EnhancedDetailMethod {
    GP3_ENHANCED,       // 增强贪婪投影三角化
    RIMLS_FULL,         // 完整RIMLS实现
    POISSON_ADAPTIVE,   // 自适应泊松重建
    HYBRID              // 混合方法
};

/**
 * 去噪方法枚举
 */
enum class DenoisingMethod {
    BILATERAL,          // 双边滤波
    RIMLS_DENOISING,    // RIMLS去噪
    WLOP,               // 加权局部最优投影
    ADAPTIVE            // 自适应去噪
};

/**
 * 增强版细节重建配置参数
 */
struct EnhancedDetailReconstructionConfig {
    // 几何提取参数
    float offset_distance = 0.08f;                 // 偏移带距离(8厘米)
    bool outside_only = true;                      // 仅提取外壳外侧点
    bool use_precise_distance = true;              // 使用精确距离计算
    int distance_samples = 8;                      // 距离采样数
    
    // 去噪参数
    bool enable_denoising = true;                  // 启用去噪
    DenoisingMethod denoising_method = DenoisingMethod::ADAPTIVE;
    float noise_threshold = 0.005f;                // 噪声阈值(5毫米)
    
    // 双边滤波参数
    struct BilateralConfig {
        float sigma_s = 0.02f;                     // 空间标准差(2厘米)
        float sigma_r = 0.01f;                     // 范围标准差(1厘米)
    } bilateral;
    
    // WLOP去噪参数
    struct WLOPConfig {
        int max_iterations = 35;                   // 最大迭代次数
        float convergence_threshold = 1e-6f;      // 收敛阈值
        float mu = 0.45f;                          // μ参数
        int neighborhood_size = 16;                // 邻域大小
        bool preserve_features = true;             // 保持特征
        float feature_threshold = 0.1f;            // 特征阈值
    } wlop;
    
    // RIMLS去噪参数
    struct RIMLSDenoisingConfig {
        float bandwidth_multiplier = 1.5f;        // 带宽倍数
        int polynomial_order = 2;                 // 多项式阶数
        float outlier_threshold = 2.0f;           // 离群点阈值
        bool iterative_refinement = true;         // 迭代细化
        int max_iterations = 5;                   // 最大迭代次数
    } rimls_denoising;
    
    // 重建方法选择
    EnhancedDetailMethod primary_method = EnhancedDetailMethod::RIMLS_FULL;
    
    // 增强GP3参数
    struct EnhancedGP3Config {
        bool adaptive_parameters = true;           // 自适应参数
        float base_edge_length = 0.02f;            // 基础边长(2厘米)
        float edge_length_multiplier = 2.5f;       // 边长倍数
        float mu = 2.5f;                           // μ参数
        float angle_threshold_planar = 40.0f;      // 平面区域角度阈值(度)
        float angle_threshold_edge = 25.0f;        // 边缘区域角度阈值(度)
        float min_angle = 10.0f;                   // 最小三角形角度(度)
        float max_angle = 120.0f;                  // 最大三角形角度(度)
        int max_nearest_neighbors = 150;           // 最大近邻数
        bool normal_consistency = true;            // 法向量一致性
        bool density_adaptive = true;              // 密度自适应
    } enhanced_gp3;
    
    // 完整RIMLS参数
    struct FullRIMLSConfig {
        float bandwidth_multiplier = 1.2f;        // 带宽倍数
        bool adaptive_bandwidth = true;           // 自适应带宽
        int polynomial_order = 2;                 // 多项式阶数
        float regularization = 0.001f;            // 正则化参数
        
        // 权重函数参数
        enum class WeightFunction {
            WENDLAND,                              // Wendland函数
            GAUSSIAN,                              // 高斯函数
            INVERSE_DISTANCE                       // 反距离权重
        } weight_function = WeightFunction::WENDLAND;
        
        float weight_parameter = 2.0f;            // 权重参数
        
        // 隐式表面提取
        float voxel_size = 0.005f;                // 体素大小(5毫米)
        float iso_value = 0.0f;                   // 等值面值
        bool use_marching_cubes = true;           // 使用行进立方体
        
        // 质量控制
        float min_confidence = 0.1f;              // 最小置信度
        bool remove_low_confidence = true;        // 移除低置信度区域
        
        // 并行处理
        bool use_parallel_fitting = true;         // 并行拟合
        int chunk_size = 1000;                    // 分块大小
    } full_rimls;
    
    // 自适应泊松参数
    struct AdaptivePoissonConfig {
        int octree_depth = 9;                     // 八叉树深度
        bool adaptive_depth = true;               // 自适应深度
        float samples_per_node = 3.0f;            // 每节点样本数
        float point_weight = 4.0f;                // 点权重
        bool use_color = false;                   // 使用颜色
        bool manifold = true;                     // 流形约束
        float confidence_threshold = 0.1f;        // 置信度阈值
    } adaptive_poisson;
    
    // 混合方法参数
    struct HybridConfig {
        bool use_density_based_selection = true;  // 基于密度的方法选择
        float high_density_threshold = 100.0f;    // 高密度阈值(点/立方厘米)
        float low_density_threshold = 20.0f;      // 低密度阈值(点/立方厘米)
        
        EnhancedDetailMethod high_density_method = EnhancedDetailMethod::GP3_ENHANCED;
        EnhancedDetailMethod medium_density_method = EnhancedDetailMethod::RIMLS_FULL;
        EnhancedDetailMethod low_density_method = EnhancedDetailMethod::POISSON_ADAPTIVE;
        
        bool blend_boundaries = true;             // 混合边界
        float blend_distance = 0.02f;             // 混合距离(2厘米)
    } hybrid;
    
    // 后处理参数
    struct EnhancedPostProcessingConfig {
        // 基础清理
        bool remove_hanging_edges = true;         // 移除悬挂边
        bool filter_small_components = true;      // 过滤小组件
        int min_component_size = 100;             // 最小组件大小
        bool remove_isolated_vertices = true;     // 移除孤立顶点
        
        // 高级修复
        bool enable_advanced_repair = true;       // 启用高级修复
        bool fix_self_intersections = true;       // 修复自相交
        bool fix_non_manifold = true;             // 修复非流形
        bool fix_degenerate_faces = true;         // 修复退化面
        bool fix_orientation = true;              // 修复方向
        
        // 网格优化
        bool enable_mesh_optimization = true;     // 启用网格优化
        bool smooth_mesh = true;                  // 平滑网格
        int smoothing_iterations = 3;             // 平滑迭代次数
        float smoothing_factor = 0.5f;            // 平滑因子
        bool preserve_boundaries = true;          // 保持边界
        
        // 简化
        bool enable_simplification = false;       // 启用简化
        float reduction_ratio = 0.95f;            // 简化比例
        bool preserve_sharp_edges = true;         // 保持尖锐边缘
        float sharp_edge_threshold = 45.0f;       // 尖锐边缘阈值(度)
        
        // 重新网格化
        bool enable_remeshing = false;            // 启用重新网格化
        float target_edge_length = 0.01f;         // 目标边长(1厘米)
        int remeshing_iterations = 5;             // 重新网格化迭代次数
    } enhanced_post_processing;
    
    // 质量控制参数
    struct EnhancedQualityControlConfig {
        // 几何质量
        bool check_manifold = true;               // 检查流形
        bool check_orientation = true;            // 检查方向
        bool check_watertight = true;             // 检查水密性
        float max_edge_length = 0.05f;            // 最大边长(5厘米)
        float min_edge_length = 0.001f;           // 最小边长(1毫米)
        float max_aspect_ratio = 10.0f;           // 最大长宽比
        
        // 拓扑质量
        bool check_connectivity = true;           // 检查连通性
        bool check_genus = false;                 // 检查亏格
        int max_components = 10;                  // 最大组件数
        
        // 特征保持
        bool preserve_sharp_edges = true;         // 保持尖锐边缘
        float sharp_edge_threshold = 30.0f;       // 尖锐边缘阈值(度)
        bool preserve_boundaries = true;          // 保持边界
        bool preserve_corners = true;             // 保持角点
        float corner_threshold = 60.0f;           // 角点阈值(度)
        
        // 质量评分
        bool compute_quality_metrics = true;      // 计算质量指标
        bool generate_quality_report = false;     // 生成质量报告
    } enhanced_quality_control;
    
    // 性能优化参数
    struct PerformanceConfig {
        bool use_parallel_processing = true;      // 使用并行处理
        int num_threads = 0;                      // 线程数(0=自动)
        bool use_spatial_indexing = true;         // 使用空间索引
        bool enable_caching = true;               // 启用缓存
        int cache_size = 10000;                   // 缓存大小
        bool memory_efficient = false;            // 内存高效模式
    } performance;
    
    // 调试参数
    bool enable_debug_output = false;             // 启用调试输出
    bool save_intermediate_results = false;       // 保存中间结果
    std::string debug_output_dir = "./debug/";    // 调试输出目录
};

/**
 * RIMLS局部拟合数据结构
 */
struct RIMLSLocalFit {
    Eigen::Vector3f center;                       // 拟合中心
    Eigen::Vector3f normal;                       // 拟合法向量
    Eigen::Matrix3f covariance;                   // 协方差矩阵
    std::vector<float> polynomial_coeffs;         // 多项式系数
    float confidence;                             // 置信度
    float bandwidth;                              // 局部带宽
    int num_neighbors;                            // 邻居数量
    float fitting_error;                          // 拟合误差
    bool is_valid;                                // 是否有效
    
    RIMLSLocalFit() 
        : center(Eigen::Vector3f::Zero()), normal(Eigen::Vector3f::UnitZ()),
          covariance(Eigen::Matrix3f::Identity()), confidence(0.0f),
          bandwidth(0.0f), num_neighbors(0), fitting_error(0.0f), is_valid(false) {}
};

/**
 * 密度信息结构
 */
struct DensityInfo {
    float local_density;                          // 局部密度
    float neighborhood_radius;                    // 邻域半径
    int neighbor_count;                           // 邻居数量
    float uniformity;                             // 均匀性
    bool is_boundary;                             // 是否边界
    bool is_feature;                              // 是否特征
    
    DensityInfo() 
        : local_density(0.0f), neighborhood_radius(0.0f), neighbor_count(0),
          uniformity(0.0f), is_boundary(false), is_feature(false) {}
};

/**
 * 增强版细节重建器类
 */
class EnhancedDetailReconstructor {
public:
    using PointT = pcl::PointXYZRGBNormal;
    using PointCloudT = pcl::PointCloud<PointT>;

    /**
     * 构造函数
     */
    explicit EnhancedDetailReconstructor(const EnhancedDetailReconstructionConfig& config = EnhancedDetailReconstructionConfig{});
    
    /**
     * 析构函数
     */
    ~EnhancedDetailReconstructor();

    /**
     * 从外壳网格和原始点云重建细节
     * @param shell_mesh 外壳网格
     * @param original_cloud 原始点云
     * @param detail_mesh 输出细节网格
     * @return 成功返回true
     */
    bool reconstructDetails(const pcl::PolygonMesh& shell_mesh,
                           const PointCloudT::Ptr& original_cloud,
                           pcl::PolygonMesh& detail_mesh);

    /**
     * 设置配置参数
     */
    void setConfig(const EnhancedDetailReconstructionConfig& config) { config_ = config; }
    
    /**
     * 获取配置参数
     */
    const EnhancedDetailReconstructionConfig& getConfig() const { return config_; }

    /**
     * 获取重建统计信息
     */
    struct Statistics {
        // 输入统计
        int original_points = 0;
        int extracted_points = 0;
        int denoised_points = 0;
        
        // 输出统计
        int output_vertices = 0;
        int output_triangles = 0;
        int removed_components = 0;
        int fixed_intersections = 0;
        int fixed_non_manifold = 0;
        
        // 时间统计
        double total_time = 0.0;
        double extraction_time = 0.0;
        double denoising_time = 0.0;
        double reconstruction_time = 0.0;
        double post_processing_time = 0.0;
        double quality_control_time = 0.0;
        
        // 质量统计
        float average_edge_length = 0.0f;
        float max_aspect_ratio = 0.0f;
        float mesh_quality_score = 0.0f;
        bool is_manifold = false;
        bool is_watertight = false;
        int num_components = 0;
        
        // RIMLS特定统计
        int rimls_fits_computed = 0;
        float average_fitting_error = 0.0f;
        float average_confidence = 0.0f;
        int low_confidence_regions = 0;
        
        // 性能统计
        float points_per_second = 0.0f;
        float triangles_per_second = 0.0f;
        size_t peak_memory_usage = 0;
    };
    
    const Statistics& getStatistics() const { return stats_; }

private:
    EnhancedDetailReconstructionConfig config_;
    Statistics stats_;
    
    // 缓存和索引
    pcl::KdTreeFLANN<PointT>::Ptr kdtree_;
    std::unordered_map<int, DensityInfo> density_cache_;
    std::unordered_map<int, RIMLSLocalFit> rimls_cache_;
    
    // 线程安全
    mutable std::mutex cache_mutex_;
    mutable std::mutex stats_mutex_;
    
    /**
     * 精确偏移带点提取
     */
    bool extractOffsetBandPointsPrecise(const pcl::PolygonMesh& shell_mesh,
                                       const PointCloudT::Ptr& original_cloud,
                                       PointCloudT::Ptr& extracted_points);
    
    /**
     * 计算点到网格的精确距离
     */
    float computePrecisePointToMeshDistance(const PointT& point, 
                                           const pcl::PolygonMesh& mesh);
    
    /**
     * 增强去噪处理
     */
    bool enhancedDenoising(const PointCloudT::Ptr& input_points,
                          PointCloudT::Ptr& denoised_points);
    
    /**
     * WLOP去噪实现
     */
    bool wlopDenoising(const PointCloudT::Ptr& input, PointCloudT::Ptr& output);
    
    /**
     * RIMLS去噪实现
     */
    bool rimlsDenoising(const PointCloudT::Ptr& input, PointCloudT::Ptr& output);
    
    /**
     * 自适应去噪
     */
    bool adaptiveDenoising(const PointCloudT::Ptr& input, PointCloudT::Ptr& output);
    
    /**
     * 增强GP3重建
     */
    bool reconstructWithEnhancedGP3(const PointCloudT::Ptr& points,
                                   pcl::PolygonMesh& mesh);
    
    /**
     * 完整RIMLS重建
     */
    bool reconstructWithFullRIMLS(const PointCloudT::Ptr& points,
                                 pcl::PolygonMesh& mesh);
    
    /**
     * 自适应泊松重建
     */
    bool reconstructWithAdaptivePoisson(const PointCloudT::Ptr& points,
                                       pcl::PolygonMesh& mesh);
    
    /**
     * 混合方法重建
     */
    bool reconstructWithHybridMethod(const PointCloudT::Ptr& points,
                                    pcl::PolygonMesh& mesh);
    
    /**
     * RIMLS局部拟合计算
     */
    bool computeRIMLSLocalFits(const PointCloudT::Ptr& points,
                              std::vector<RIMLSLocalFit>& fits);
    
    /**
     * 单点RIMLS拟合
     */
    RIMLSLocalFit computeSingleRIMLSFit(const PointT& query_point,
                                       const PointCloudT::Ptr& points,
                                       const std::vector<int>& neighbors);
    
    /**
     * 权重函数计算
     */
    float computeWeightFunction(float distance, float bandwidth, 
                               EnhancedDetailReconstructionConfig::FullRIMLSConfig::WeightFunction type);
    
    /**
     * 隐式表面提取
     */
    bool extractImplicitSurface(const std::vector<RIMLSLocalFit>& fits,
                               const PointCloudT::Ptr& points,
                               pcl::PolygonMesh& mesh);
    
    /**
     * 自适应带宽计算
     */
    float computeAdaptiveBandwidth(const PointT& point, 
                                  const PointCloudT::Ptr& points,
                                  const std::vector<int>& neighbors);
    
    /**
     * 密度信息计算
     */
    DensityInfo computeDensityInfo(const PointT& point, 
                                  const PointCloudT::Ptr& points);
    
    /**
     * 自适应参数计算
     */
    void computeAdaptiveParameters(const PointCloudT::Ptr& points);
    
    /**
     * 增强后处理
     */
    bool enhancedPostProcessing(pcl::PolygonMesh& mesh);
    
    /**
     * 高级网格修复
     */
    bool advancedMeshRepair(pcl::PolygonMesh& mesh);
    
    /**
     * 网格优化
     */
    bool optimizeMesh(pcl::PolygonMesh& mesh);
    
    /**
     * 增强质量控制
     */
    bool enhancedQualityControl(const pcl::PolygonMesh& mesh);
    
    /**
     * 计算网格质量指标
     */
    void computeQualityMetrics(const pcl::PolygonMesh& mesh);
    
    /**
     * 检查流形性
     */
    bool checkManifold(const pcl::PolygonMesh& mesh);
    
    /**
     * 检查水密性
     */
    bool checkWatertight(const pcl::PolygonMesh& mesh);
    
    /**
     * 更新统计信息
     */
    void updateStatistics();
    
    /**
     * 并行处理工具
     */
    void processPointsInParallel(const PointCloudT::Ptr& points,
                                std::function<void(int)> processor);
    
    /**
     * 内存使用监控
     */
    size_t getCurrentMemoryUsage();
    
    /**
     * 调试输出
     */
    void saveDebugResults(const std::string& stage, const pcl::PolygonMesh& mesh);
};

/**
 * 增强版网格融合器类
 */
class EnhancedMeshFuser {
public:
    using PointT = pcl::PointXYZRGBNormal;
    using PointCloudT = pcl::PointCloud<PointT>;

    /**
     * 增强融合配置参数
     */
    struct EnhancedFusionConfig {
        // 基础融合参数
        bool shell_priority = true;               // 外壳优先
        float detail_threshold = 0.005f;          // 细节阈值(5毫米)
        float overlap_tolerance = 0.002f;         // 重叠容差(2毫米)
        
        // 布尔运算参数
        enum class BooleanMethod {
            UNION,                                // 并集
            DIFFERENCE,                           // 差集
            INTERSECTION,                         // 交集
            ADAPTIVE                              // 自适应
        } boolean_method = BooleanMethod::ADAPTIVE;
        
        bool use_robust_boolean = true;           // 使用鲁棒布尔运算
        float boolean_tolerance = 0.001f;         // 布尔运算容差(1毫米)
        bool fix_intersections_before = true;     // 运算前修复相交
        bool fix_intersections_after = true;      // 运算后修复相交
        
        // 高级融合策略
        bool use_distance_based_blending = true;  // 基于距离的混合
        float blending_distance = 0.01f;          // 混合距离(1厘米)
        bool preserve_shell_topology = true;      // 保持外壳拓扑
        bool preserve_detail_features = true;     // 保持细节特征
        
        // 顶点焊接参数
        float welding_threshold = 0.001f;         // 焊接阈值(1毫米)
        bool smart_welding = true;                // 智能焊接
        bool preserve_boundaries = true;          // 保持边界
        
        // 颜色融合参数
        struct ColorFusionConfig {
            bool enable_color_fusion = true;      // 启用颜色融合
            enum class ColorMethod {
                SHELL_PRIORITY,                   // 外壳优先
                DETAIL_PRIORITY,                  // 细节优先
                DISTANCE_WEIGHTED,                // 距离加权
                FEATURE_AWARE                     // 特征感知
            } color_method = ColorMethod::FEATURE_AWARE;
            
            float shell_weight = 0.6f;            // 外壳权重
            float detail_weight = 0.4f;           // 细节权重
            float transition_distance = 0.005f;   // 过渡距离(5毫米)
            bool smooth_transitions = true;       // 平滑过渡
        } color_fusion;
        
        // 质量保证参数
        bool validate_result = true;              // 验证结果
        bool fix_result_topology = true;          // 修复结果拓扑
        bool optimize_result = true;              // 优化结果
        
        // 性能参数
        bool use_parallel_processing = true;      // 并行处理
        bool use_spatial_acceleration = true;     // 空间加速
        int chunk_size = 5000;                    // 分块大小
    };

    /**
     * 构造函数
     */
    explicit EnhancedMeshFuser(const EnhancedFusionConfig& config = EnhancedFusionConfig{});
    
    /**
     * 增强网格融合
     */
    bool fuseMeshesEnhanced(const pcl::PolygonMesh& shell_mesh,
                           const pcl::PolygonMesh& detail_mesh,
                           const PointCloudT::Ptr& original_cloud,
                           pcl::PolygonMesh& fused_mesh);

    /**
     * 设置配置参数
     */
    void setConfig(const EnhancedFusionConfig& config) { config_ = config; }

    /**
     * 获取融合统计信息
     */
    struct FusionStatistics {
        int shell_vertices = 0;
        int shell_triangles = 0;
        int detail_vertices = 0;
        int detail_triangles = 0;
        int fused_vertices = 0;
        int fused_triangles = 0;
        int welded_vertices = 0;
        int fixed_intersections = 0;
        double fusion_time = 0.0;
        double boolean_time = 0.0;
        double welding_time = 0.0;
        double color_time = 0.0;
        bool fusion_successful = false;
    };
    
    const FusionStatistics& getStatistics() const { return stats_; }

private:
    EnhancedFusionConfig config_;
    FusionStatistics stats_;
    
    /**
     * 鲁棒布尔运算
     */
    bool robustBooleanOperation(const pcl::PolygonMesh& mesh1,
                               const pcl::PolygonMesh& mesh2,
                               pcl::PolygonMesh& result);
    
    /**
     * 基于距离的网格混合
     */
    bool distanceBasedBlending(const pcl::PolygonMesh& shell_mesh,
                              const pcl::PolygonMesh& detail_mesh,
                              pcl::PolygonMesh& blended_mesh);
    
    /**
     * 智能顶点焊接
     */
    bool smartVertexWelding(pcl::PolygonMesh& mesh);
    
    /**
     * 特征感知颜色融合
     */
    bool featureAwareColorFusion(pcl::PolygonMesh& mesh, 
                                const PointCloudT::Ptr& original_cloud);
    
    /**
     * 融合结果验证
     */
    bool validateFusionResult(const pcl::PolygonMesh& mesh);
};

} // namespace recon

#endif // ENHANCED_DETAIL_RECONSTRUCTION_H

