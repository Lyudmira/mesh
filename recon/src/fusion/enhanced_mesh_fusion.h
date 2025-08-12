/**
 * 增强版网格融合模块头文件
 * 负责将外壳网格和细节网格进行高质量融合
 * 
 * 版本: 2.0 - 增强版本
 * 日期: 2025-08-12
 * 
 * 主要改进:
 * - 鲁棒布尔运算系统
 * - 智能顶点焊接
 * - 特征感知颜色融合
 * - 自适应融合策略
 * - 质量验证和修复
 * - 并行处理优化
 */

#ifndef ENHANCED_MESH_FUSION_H
#define ENHANCED_MESH_FUSION_H

#include <memory>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <atomic>
#include <mutex>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/PolygonMesh.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <Eigen/Dense>

// 自定义pair哈希函数
namespace std {
    template<>
    struct hash<std::pair<int, int>> {
        size_t operator()(const std::pair<int, int>& p) const {
            return std::hash<int>()(p.first) ^ (std::hash<int>()(p.second) << 1);
        }
    };
}
#include <Eigen/Sparse>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace recon {

/**
 * 布尔运算方法枚举
 */
enum class BooleanMethod {
    FAST_IGL,           // 快速igl方法
    ROBUST_CGAL,        // 鲁棒CGAL方法
    ALPHA_WRAPPING,     // Alpha包装
    VOXEL_BASED,        // 基于体素
    ADAPTIVE            // 自适应选择
};

/**
 * 布尔运算类型枚举
 */
enum class BooleanOperation {
    UNION,              // 并集
    DIFFERENCE,         // 差集
    INTERSECTION,       // 交集
    SYMMETRIC_DIFF      // 对称差集
};

/**
 * 颜色融合方法枚举
 */
enum class ColorFusionMethod {
    SHELL_PRIORITY,     // 外壳优先
    DETAIL_PRIORITY,    // 细节优先
    DISTANCE_WEIGHTED,  // 距离加权
    NORMAL_WEIGHTED,    // 法向量加权
    FEATURE_AWARE,      // 特征感知
    ADAPTIVE            // 自适应融合
};

/**
 * 焊接策略枚举
 */
enum class WeldingStrategy {
    DISTANCE_ONLY,      // 仅距离
    DISTANCE_NORMAL,    // 距离+法向量
    GEOMETRIC_FEATURE,  // 几何特征
    TOPOLOGICAL,        // 拓扑优化
    SMART_ADAPTIVE      // 智能自适应
};

/**
 * 增强版网格融合配置参数
 */
struct EnhancedMeshFusionConfig {
    // 基础融合参数
    bool shell_priority = true;                    // 外壳优先
    float detail_threshold = 0.005f;               // 细节阈值(5毫米)
    float overlap_tolerance = 0.002f;              // 重叠容差(2毫米)
    bool preserve_shell_topology = true;           // 保持外壳拓扑
    bool preserve_detail_features = true;          // 保持细节特征
    
    // 布尔运算参数
    struct BooleanConfig {
        BooleanOperation operation = BooleanOperation::UNION;
        BooleanMethod primary_method = BooleanMethod::ADAPTIVE;
        std::vector<BooleanMethod> fallback_methods = {
            BooleanMethod::FAST_IGL,
            BooleanMethod::ROBUST_CGAL,
            BooleanMethod::ALPHA_WRAPPING
        };
        
        float tolerance = 0.001f;                  // 布尔运算容差(1毫米)
        bool fix_intersections_before = true;      // 运算前修复相交
        bool fix_intersections_after = true;       // 运算后修复相交
        bool remove_duplicates = true;             // 移除重复顶点
        bool validate_result = true;               // 验证结果
        
        // 高级参数
        bool use_exact_arithmetic = false;         // 使用精确算术
        int max_iterations = 3;                    // 最大重试次数
        float mesh_resolution = 0.001f;            // 网格分辨率(1毫米)
        bool enable_self_intersection_removal = true; // 启用自相交移除
    } boolean_ops;
    
    // 顶点焊接参数
    struct WeldingConfig {
        WeldingStrategy strategy = WeldingStrategy::SMART_ADAPTIVE;
        float distance_threshold = 0.001f;         // 距离阈值(1毫米)
        float normal_threshold = 0.9f;             // 法向量阈值(cos 25.8°)
        float feature_threshold = 0.7f;            // 特征阈值(cos 45.5°)
        
        bool preserve_boundaries = true;           // 保持边界
        bool preserve_sharp_edges = true;          // 保持尖锐边缘
        float sharp_edge_threshold = 0.5f;         // 尖锐边缘阈值(cos 60°)
        
        bool use_spatial_hashing = true;           // 使用空间哈希
        int hash_grid_resolution = 1000;           // 哈希网格分辨率
        bool parallel_welding = true;              // 并行焊接
        
        // 质量控制
        bool validate_manifold = true;             // 验证流形性
        bool fix_non_manifold = true;              // 修复非流形
        int max_welding_iterations = 5;           // 最大焊接迭代次数
    } welding;
    
    // 颜色融合参数
    struct ColorFusionConfig {
        bool enable_color_fusion = true;           // 启用颜色融合
        ColorFusionMethod method = ColorFusionMethod::FEATURE_AWARE;
        
        float shell_weight = 0.6f;                 // 外壳权重
        float detail_weight = 0.4f;                // 细节权重
        float transition_distance = 0.01f;         // 过渡距离(1厘米)
        bool smooth_transitions = true;            // 平滑过渡
        
        // 距离权重参数
        float max_search_radius = 0.05f;           // 最大搜索半径(5厘米)
        int max_neighbors = 10;                    // 最大邻居数
        float distance_falloff = 2.0f;             // 距离衰减指数
        
        // 法向量权重参数
        bool use_normal_weighting = true;          // 使用法向量权重
        float normal_weight_power = 2.0f;          // 法向量权重指数
        
        // 特征感知参数
        bool detect_features = true;               // 检测特征
        float feature_angle_threshold = 30.0f;     // 特征角度阈值(度)
        float feature_weight_multiplier = 2.0f;    // 特征权重倍数
        
        // 颜色空间参数
        enum class ColorSpace {
            RGB,                                   // RGB颜色空间
            LAB,                                   // LAB颜色空间
            HSV                                    // HSV颜色空间
        } color_space = ColorSpace::LAB;
        
        bool gamma_correction = true;              // 伽马校正
        float gamma_value = 2.2f;                  // 伽马值
    } color_fusion;
    
    // 质量控制参数
    struct QualityControlConfig {
        bool enable_quality_control = true;        // 启用质量控制
        
        // 几何质量
        bool check_manifold = true;                // 检查流形性
        bool check_watertight = true;              // 检查水密性
        bool check_orientation = true;             // 检查方向一致性
        bool check_self_intersections = true;      // 检查自相交
        
        float max_edge_length = 0.1f;              // 最大边长(10厘米)
        float min_edge_length = 0.0001f;           // 最小边长(0.1毫米)
        float max_aspect_ratio = 20.0f;            // 最大长宽比
        float min_triangle_area = 1e-8f;           // 最小三角形面积
        
        // 拓扑质量
        int max_components = 1;                    // 最大连通组件数
        bool allow_boundaries = false;             // 允许边界
        bool fix_topology_errors = true;           // 修复拓扑错误
        
        // 修复参数
        bool enable_mesh_repair = true;            // 启用网格修复
        bool remove_degenerate_faces = true;       // 移除退化面
        bool remove_duplicate_faces = true;        // 移除重复面
        bool fix_face_orientation = true;          // 修复面方向
        bool fill_holes = false;                   // 填充孔洞
        float max_hole_size = 0.01f;               // 最大孔洞大小(1厘米)
    } quality_control;
    
    // 自适应策略参数
    struct AdaptiveConfig {
        bool enable_adaptive_strategy = true;      // 启用自适应策略
        
        // 复杂度评估
        bool analyze_mesh_complexity = true;       // 分析网格复杂度
        float complexity_threshold_low = 0.3f;     // 低复杂度阈值
        float complexity_threshold_high = 0.7f;    // 高复杂度阈值
        
        // 重叠分析
        bool analyze_overlap_regions = true;       // 分析重叠区域
        float overlap_ratio_threshold = 0.1f;      // 重叠比例阈值
        
        // 质量要求
        enum class QualityLevel {
            FAST,                                  // 快速
            BALANCED,                              // 平衡
            HIGH_QUALITY                           // 高质量
        } target_quality = QualityLevel::BALANCED;
        
        // 性能约束
        float max_processing_time = 60.0f;         // 最大处理时间(秒)
        size_t max_memory_usage = 2ULL * 1024 * 1024 * 1024; // 最大内存使用(2GB)
        
        // 策略选择权重
        float speed_weight = 0.3f;                 // 速度权重
        float quality_weight = 0.5f;               // 质量权重
        float robustness_weight = 0.2f;            // 鲁棒性权重
    } adaptive;
    
    // 性能优化参数
    struct PerformanceConfig {
        bool use_parallel_processing = true;       // 使用并行处理
        int num_threads = 0;                       // 线程数(0=自动)
        
        bool use_spatial_acceleration = true;      // 使用空间加速
        bool use_memory_mapping = false;           // 使用内存映射
        bool enable_caching = true;                // 启用缓存
        
        int chunk_size = 10000;                    // 分块大小
        bool memory_efficient_mode = false;        // 内存高效模式
        
        // 进度监控
        bool enable_progress_monitoring = true;    // 启用进度监控
        bool verbose_output = false;               // 详细输出
    } performance;
    
    // 调试参数
    bool enable_debug_output = false;              // 启用调试输出
    bool save_intermediate_results = false;        // 保存中间结果
    std::string debug_output_dir = "./debug/";     // 调试输出目录
};

/**
 * 网格质量信息结构
 */
struct MeshQualityInfo {
    // 基础统计
    int num_vertices = 0;
    int num_faces = 0;
    int num_edges = 0;
    int num_components = 0;
    
    // 几何质量
    float min_edge_length = 0.0f;
    float max_edge_length = 0.0f;
    float avg_edge_length = 0.0f;
    float min_triangle_area = 0.0f;
    float max_triangle_area = 0.0f;
    float avg_triangle_area = 0.0f;
    float max_aspect_ratio = 0.0f;
    
    // 拓扑质量
    bool is_manifold = false;
    bool is_watertight = false;
    bool has_boundaries = false;
    int num_boundary_edges = 0;
    int num_non_manifold_edges = 0;
    int num_self_intersections = 0;
    
    // 方向一致性
    bool consistent_orientation = false;
    int num_flipped_faces = 0;
    
    // 质量评分
    float geometric_quality_score = 0.0f;
    float topological_quality_score = 0.0f;
    float overall_quality_score = 0.0f;
};

/**
 * 融合统计信息结构
 */
struct FusionStatistics {
    // 输入统计
    int shell_vertices = 0;
    int shell_faces = 0;
    int detail_vertices = 0;
    int detail_faces = 0;
    
    // 输出统计
    int fused_vertices = 0;
    int fused_faces = 0;
    int welded_vertices = 0;
    int removed_faces = 0;
    int fixed_intersections = 0;
    
    // 时间统计
    double total_time = 0.0;
    double preprocessing_time = 0.0;
    double boolean_time = 0.0;
    double welding_time = 0.0;
    double color_fusion_time = 0.0;
    double quality_control_time = 0.0;
    double postprocessing_time = 0.0;
    
    // 质量统计
    MeshQualityInfo input_shell_quality;
    MeshQualityInfo input_detail_quality;
    MeshQualityInfo output_quality;
    
    // 融合效果
    float overlap_ratio = 0.0f;
    float fusion_success_rate = 0.0f;
    bool fusion_successful = false;
    
    // 性能统计
    float vertices_per_second = 0.0f;
    float faces_per_second = 0.0f;
    size_t peak_memory_usage = 0;
    
    // 方法使用统计
    BooleanMethod used_boolean_method = BooleanMethod::FAST_IGL;
    WeldingStrategy used_welding_strategy = WeldingStrategy::DISTANCE_ONLY;
    ColorFusionMethod used_color_method = ColorFusionMethod::DISTANCE_WEIGHTED;
    int num_method_retries = 0;
};

/**
 * 增强版网格融合器类
 */
class EnhancedMeshFuser {
public:
    using PointT = pcl::PointXYZRGBNormal;
    using PointCloudT = pcl::PointCloud<PointT>;

    /**
     * 构造函数
     */
    explicit EnhancedMeshFuser(const EnhancedMeshFusionConfig& config = EnhancedMeshFusionConfig{});
    
    /**
     * 析构函数
     */
    ~EnhancedMeshFuser();

    /**
     * 增强网格融合主函数
     * @param shell_mesh 外壳网格
     * @param detail_mesh 细节网格
     * @param original_cloud 原始点云(用于颜色分配)
     * @param fused_mesh 输出融合网格
     * @return 成功返回true
     */
    bool fuseMeshesEnhanced(const pcl::PolygonMesh& shell_mesh,
                           const pcl::PolygonMesh& detail_mesh,
                           const PointCloudT::Ptr& original_cloud,
                           pcl::PolygonMesh& fused_mesh);

    /**
     * 设置配置参数
     */
    void setConfig(const EnhancedMeshFusionConfig& config) { config_ = config; }
    
    /**
     * 获取配置参数
     */
    const EnhancedMeshFusionConfig& getConfig() const { return config_; }

    /**
     * 获取融合统计信息
     */
    const FusionStatistics& getStatistics() const { return stats_; }
    
    /**
     * 获取网格质量信息
     */
    MeshQualityInfo analyzeMeshQuality(const pcl::PolygonMesh& mesh);

private:
    EnhancedMeshFusionConfig config_;
    FusionStatistics stats_;
    
    // 缓存和索引
    std::unordered_map<int, std::vector<int>> vertex_neighbors_cache_;
    std::unordered_map<std::string, bool> boolean_result_cache_;
    
    // 线程安全
    mutable std::mutex cache_mutex_;
    mutable std::mutex stats_mutex_;
    
    /**
     * 预处理网格
     */
    bool preprocessMeshes(pcl::PolygonMesh& shell_mesh,
                         pcl::PolygonMesh& detail_mesh);
    
    /**
     * 分析网格复杂度
     */
    float analyzeMeshComplexity(const pcl::PolygonMesh& mesh);
    
    /**
     * 分析重叠区域
     */
    float analyzeOverlapRegions(const pcl::PolygonMesh& shell_mesh,
                               const pcl::PolygonMesh& detail_mesh);
    
    /**
     * 自适应策略选择
     */
    void selectAdaptiveStrategy();
    
    /**
     * 鲁棒布尔运算
     */
    bool robustBooleanOperation(const pcl::PolygonMesh& mesh1,
                               const pcl::PolygonMesh& mesh2,
                               pcl::PolygonMesh& result);
    
    /**
     * 尝试特定布尔方法
     */
    bool tryBooleanMethod(BooleanMethod method,
                         const pcl::PolygonMesh& mesh1,
                         const pcl::PolygonMesh& mesh2,
                         pcl::PolygonMesh& result);
    
    /**
     * 快速igl布尔运算
     */
    bool fastIglBoolean(const pcl::PolygonMesh& mesh1,
                       const pcl::PolygonMesh& mesh2,
                       pcl::PolygonMesh& result);
    
    /**
     * 鲁棒CGAL布尔运算
     */
    bool robustCgalBoolean(const pcl::PolygonMesh& mesh1,
                          const pcl::PolygonMesh& mesh2,
                          pcl::PolygonMesh& result);
    
    /**
     * Alpha包装布尔运算
     */
    bool alphaWrappingBoolean(const pcl::PolygonMesh& mesh1,
                             const pcl::PolygonMesh& mesh2,
                             pcl::PolygonMesh& result);
    
    /**
     * 基于体素的布尔运算
     */
    bool voxelBasedBoolean(const pcl::PolygonMesh& mesh1,
                          const pcl::PolygonMesh& mesh2,
                          pcl::PolygonMesh& result);
    
    /**
     * 智能顶点焊接
     */
    bool smartVertexWelding(pcl::PolygonMesh& mesh);
    
    /**
     * 距离焊接
     */
    bool distanceWelding(pcl::PolygonMesh& mesh);
    
    /**
     * 几何特征焊接
     */
    bool geometricFeatureWelding(pcl::PolygonMesh& mesh);
    
    /**
     * 拓扑优化焊接
     */
    bool topologicalWelding(pcl::PolygonMesh& mesh);
    
    /**
     * 特征感知颜色融合
     */
    bool featureAwareColorFusion(pcl::PolygonMesh& mesh, 
                                const PointCloudT::Ptr& original_cloud);
    
    /**
     * 距离加权颜色融合
     */
    bool distanceWeightedColorFusion(pcl::PolygonMesh& mesh,
                                    const PointCloudT::Ptr& original_cloud);
    
    /**
     * 法向量加权颜色融合
     */
    bool normalWeightedColorFusion(pcl::PolygonMesh& mesh,
                                  const PointCloudT::Ptr& original_cloud);
    
    /**
     * 自适应颜色融合
     */
    bool adaptiveColorFusion(pcl::PolygonMesh& mesh,
                            const PointCloudT::Ptr& original_cloud);
    
    /**
     * 质量控制和验证
     */
    bool qualityControlAndValidation(pcl::PolygonMesh& mesh);
    
    /**
     * 网格修复
     */
    bool repairMesh(pcl::PolygonMesh& mesh);
    
    /**
     * 检查流形性
     */
    bool checkManifold(const pcl::PolygonMesh& mesh);
    
    /**
     * 检查水密性
     */
    bool checkWatertight(const pcl::PolygonMesh& mesh);
    
    /**
     * 检查自相交
     */
    bool checkSelfIntersections(const pcl::PolygonMesh& mesh);
    
    /**
     * 修复自相交
     */
    bool fixSelfIntersections(pcl::PolygonMesh& mesh);
    
    /**
     * 修复非流形
     */
    bool fixNonManifold(pcl::PolygonMesh& mesh);
    
    /**
     * 数据转换工具
     */
    bool pclMeshToEigen(const pcl::PolygonMesh& pcl_mesh,
                       Eigen::MatrixXd& vertices,
                       Eigen::MatrixXi& faces);
    
    bool eigenToPclMesh(const Eigen::MatrixXd& vertices,
                       const Eigen::MatrixXi& faces,
                       const Eigen::MatrixXd& normals,
                       const std::vector<Eigen::Vector3f>& colors,
                       pcl::PolygonMesh& pcl_mesh);
    
    /**
     * 空间哈希工具
     */
    class SpatialHashGrid {
    public:
        SpatialHashGrid(float cell_size, int grid_resolution);
        void insert(int vertex_id, const Eigen::Vector3f& position);
        std::vector<int> query(const Eigen::Vector3f& position, float radius);
        void clear();
        
    private:
        float cell_size_;
        int grid_resolution_;
        std::unordered_map<size_t, std::vector<int>> grid_;
        
        size_t hashPosition(const Eigen::Vector3f& pos);
    };
    
    std::unique_ptr<SpatialHashGrid> spatial_hash_;
    
    /**
     * 并行处理工具
     */
    void processVerticesInParallel(const std::vector<int>& vertex_indices,
                                  std::function<void(int)> processor);
    
    void processFacesInParallel(const std::vector<int>& face_indices,
                               std::function<void(int)> processor);
    
    /**
     * 内存使用监控
     */
    size_t getCurrentMemoryUsage();
    
    /**
     * 更新统计信息
     */
    void updateStatistics();
    
    /**
     * 调试输出
     */
    void saveDebugMesh(const std::string& stage, const pcl::PolygonMesh& mesh);
    
    /**
     * 进度监控
     */
    void reportProgress(const std::string& stage, float progress);
};

} // namespace recon

#endif // ENHANCED_MESH_FUSION_H

