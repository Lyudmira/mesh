/**
 * 优化版UDF构建模块头文件
 * 负责将点云转换为无符号距离场(UDF)
 * 
 * 版本: 2.0 - 高性能优化版本
 * 日期: 2025-08-12
 * 
 * 主要改进:
 * - 预计算和缓存机制
 * - 并行化处理
 * - 改进的几何算法
 * - 自适应细化策略
 * - 内存优化
 */

#ifndef OPTIMIZED_UDF_BUILDER_H
#define OPTIMIZED_UDF_BUILDER_H

#include <memory>
#include <vector>
#include <unordered_map>
#include <atomic>
#include <mutex>
#include <openvdb/openvdb.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <Eigen/Dense>

namespace recon {

/**
 * 优化版UDF构建器配置参数
 */
struct OptimizedUDFConfig {
    // 体素化参数
    float coarse_voxel_size = 0.03f;    // 粗体素大小(3厘米)
    float fine_voxel_size = 0.01f;      // 细体素大小(1厘米)
    
    // 细化条件 - 自适应阈值
    float curvature_threshold = 0.1f;           // 曲率阈值
    float color_gradient_threshold = 0.2f;      // 颜色梯度阈值
    float plane_distance_threshold = 0.05f;     // 平面距离阈值(5厘米)
    float density_threshold_multiplier = 2.0f;  // 密度阈值倍数
    float edge_threshold = 0.15f;               // 边缘检测阈值
    
    // 距离场参数
    float truncation_distance = 0.09f;  // 截断距离(9厘米)
    
    // 置信度权重
    float density_weight = 0.3f;        // 密度权重
    float color_weight = 0.2f;          // 颜色权重
    float planarity_weight = 0.2f;      // 平面性权重
    float curvature_weight = 0.3f;      // 曲率权重
    
    // 膨胀带参数
    float expansion_distance = 0.4f;     // 膨胀距离(40厘米)
    
    // 滤波参数
    bool use_gaussian_filter = true;     // 使用高斯滤波
    float gaussian_width = 1.0f;         // 高斯滤波宽度
    
    // 性能优化参数
    int num_threads = 0;                 // 线程数(0=自动检测)
    bool use_parallel_processing = true; // 使用并行处理
    bool use_feature_cache = true;       // 使用特征缓存
    int chunk_size = 1000;               // 分块大小
    
    // RANSAC参数
    int ransac_max_iterations = 100;     // RANSAC最大迭代次数
    float ransac_distance_threshold = 0.01f; // RANSAC距离阈值
    int ransac_min_inliers = 10;         // RANSAC最小内点数
    
    // 搜索半径参数
    float curvature_search_radius = 0.06f;   // 曲率计算搜索半径
    float density_search_radius = 0.04f;     // 密度计算搜索半径
    float color_search_radius = 0.04f;       // 颜色梯度搜索半径
    float plane_search_radius = 0.08f;       // 平面拟合搜索半径
};

/**
 * 局部特征信息
 */
struct LocalFeatures {
    float curvature = 0.0f;              // 几何曲率
    float density = 0.0f;                // 局部密度
    float color_variance = 0.0f;         // 颜色方差
    Eigen::Vector3f normal = Eigen::Vector3f::Zero(); // 局部法向量
    Eigen::Vector4f plane_coeffs = Eigen::Vector4f::Zero(); // 平面系数
    bool is_planar = false;              // 是否为平面区域
    bool is_edge = false;                // 是否为边缘区域
    float plane_distance = std::numeric_limits<float>::max(); // 到平面距离
    
    // 置信度相关
    float feature_confidence = 0.0f;     // 特征置信度
    int neighbor_count = 0;              // 邻居点数量
};

/**
 * 优化版体素细化信息
 */
struct OptimizedVoxelRefinementInfo {
    bool needs_refinement = false;
    float refinement_score = 0.0f;      // 细化评分
    LocalFeatures features;
    
    // 细化原因标记
    bool refine_for_curvature = false;
    bool refine_for_edge = false;
    bool refine_for_color = false;
    bool refine_for_density = false;
};

/**
 * 自适应细化策略
 */
struct AdaptiveRefinementCriteria {
    float curvature_threshold;
    float edge_threshold;
    float color_gradient_threshold;
    float density_variation_threshold;
    
    // 根据局部特征自适应调整阈值
    void adaptThresholds(const LocalFeatures& features);
    
    // 计算细化评分
    float computeRefinementScore(const LocalFeatures& features) const;
};

/**
 * 体素块信息（用于并行处理）
 */
struct VoxelChunk {
    openvdb::CoordBBox bbox;
    std::vector<openvdb::Coord> coords;
    std::vector<LocalFeatures> features;
    int chunk_id;
    
    VoxelChunk(int id, const openvdb::CoordBBox& box) 
        : bbox(box), chunk_id(id) {}
};

/**
 * 优化版UDF构建器类
 */
class OptimizedUDFBuilder {
public:
    using PointT = pcl::PointXYZRGBNormal;
    using PointCloudT = pcl::PointCloud<PointT>;
    using GridT = openvdb::FloatGrid;
    using ConfidenceGridT = openvdb::FloatGrid;
    using KdTreePtr = pcl::KdTreeFLANN<PointT>::Ptr;

    /**
     * 构造函数
     */
    explicit OptimizedUDFBuilder(const OptimizedUDFConfig& config = OptimizedUDFConfig{});
    
    /**
     * 析构函数
     */
    ~OptimizedUDFBuilder();

    /**
     * 从点云构建UDF（优化版本）
     * @param cloud 输入点云
     * @param udf_grid 输出UDF网格
     * @param confidence_grid 输出置信度网格
     * @return 成功返回true
     */
    bool buildUDF(PointCloudT::Ptr cloud, 
                  GridT::Ptr& udf_grid, 
                  ConfidenceGridT::Ptr& confidence_grid);

    /**
     * 设置配置参数
     */
    void setConfig(const OptimizedUDFConfig& config) { config_ = config; }
    
    /**
     * 获取配置参数
     */
    const OptimizedUDFConfig& getConfig() const { return config_; }
    
    /**
     * 获取性能统计信息
     */
    struct PerformanceStats {
        double total_time = 0.0;
        double precompute_time = 0.0;
        double voxel_processing_time = 0.0;
        double refinement_time = 0.0;
        double filtering_time = 0.0;
        
        int total_voxels = 0;
        int refined_voxels = 0;
        int parallel_chunks = 0;
        
        double voxels_per_second = 0.0;
        double memory_usage_mb = 0.0;
    };
    
    const PerformanceStats& getPerformanceStats() const { return stats_; }

private:
    OptimizedUDFConfig config_;
    PerformanceStats stats_;
    
    // 预计算的数据结构
    KdTreePtr kdtree_;
    std::unordered_map<int, LocalFeatures> feature_cache_;
    std::vector<VoxelChunk> chunks_;
    
    // 线程安全相关
    mutable std::mutex cache_mutex_;
    mutable std::mutex stats_mutex_;
    
    /**
     * 预计算点云特征
     */
    bool precomputeFeatures(const PointCloudT& cloud);
    
    /**
     * 初始化体素网格
     */
    bool initializeGrids(const PointCloudT& cloud, 
                        GridT::Ptr& udf_grid, 
                        ConfidenceGridT::Ptr& confidence_grid);
    
    /**
     * 分块处理点云
     */
    std::vector<VoxelChunk> partitionVoxelSpace(const PointCloudT& cloud);
    
    /**
     * 并行处理体素块
     */
    bool processChunksParallel(const std::vector<VoxelChunk>& chunks,
                              const PointCloudT& cloud,
                              GridT::Ptr& udf_grid,
                              ConfidenceGridT::Ptr& confidence_grid);
    
    /**
     * 处理单个体素块
     */
    void processChunk(const VoxelChunk& chunk,
                     const PointCloudT& cloud,
                     GridT::Ptr& udf_grid,
                     ConfidenceGridT::Ptr& confidence_grid);
    
    /**
     * 执行体素细化
     */
    int performRefinement(const openvdb::Coord& base_coord,
                         const openvdb::Vec3f& base_world_pos,
                         const PointCloudT& cloud,
                         GridT::Ptr& udf_grid,
                         ConfidenceGridT::Ptr& confidence_grid);
    
    /**
     * 线程安全的体素处理
     */
    void processVoxelThreadSafe(const openvdb::Coord& coord,
                               const PointCloudT& cloud,
                               GridT::Ptr& udf_grid,
                               ConfidenceGridT::Ptr& confidence_grid);
    
    /**
     * 计算优化版体素细化信息
     */
    OptimizedVoxelRefinementInfo computeOptimizedRefinementInfo(
        const openvdb::Coord& coord,
        const PointCloudT& cloud,
        const GridT& grid);
    
    /**
     * 自适应细化判断
     */
    bool needsAdaptiveRefinement(const OptimizedVoxelRefinementInfo& info,
                               const AdaptiveRefinementCriteria& criteria);
    
    /**
     * 计算体素的UDF值和置信度（优化版）
     */
    std::pair<float, float> computeOptimizedVoxelValues(
        const openvdb::Vec3f& world_pos,
        const PointCloudT& cloud);
    
    /**
     * 计算改进的置信度
     */
    float computeEnhancedConfidence(const openvdb::Vec3f& world_pos,
                                   const PointCloudT& cloud,
                                   float distance,
                                   const LocalFeatures& features);
    
    /**
     * 几何曲率计算（基于主成分分析）
     */
    float computeGeometricCurvature(const openvdb::Vec3f& pos, 
                                   const PointCloudT& cloud);
    
    /**
     * RANSAC平面拟合
     */
    float computePlaneDistanceRANSAC(const openvdb::Vec3f& pos,
                                    const PointCloudT& cloud,
                                    Eigen::Vector4f& plane_coeffs);
    
    /**
     * 改进的颜色梯度计算
     */
    float computeEnhancedColorGradient(const openvdb::Vec3f& pos, 
                                      const PointCloudT& cloud);
    
    /**
     * 边缘检测
     */
    bool detectEdge(const openvdb::Vec3f& pos, const PointCloudT& cloud);
    
    /**
     * 计算协方差矩阵
     */
    Eigen::Matrix3f computeCovarianceMatrix(const openvdb::Vec3f& pos,
                                           const PointCloudT& cloud,
                                           float search_radius);
    
    /**
     * RANSAC平面拟合实现
     */
    bool fitPlaneRANSAC(const openvdb::Vec3f& center,
                       const PointCloudT& cloud,
                       Eigen::Vector4f& plane_coeffs,
                       std::vector<int>& inliers);
    
    /**
     * 获取缓存的特征（线程安全）
     */
    LocalFeatures getCachedFeatures(int point_index) const;
    
    /**
     * 设置缓存的特征（线程安全）
     */
    void setCachedFeatures(int point_index, const LocalFeatures& features);
    
    /**
     * 更新性能统计（线程安全）
     */
    void updateStats(const std::string& operation, double time, int count = 1);
    
    /**
     * 应用优化的高斯滤波
     */
    void applyOptimizedGaussianFilter(GridT::Ptr grid);
};

} // namespace recon

#endif // OPTIMIZED_UDF_BUILDER_H

