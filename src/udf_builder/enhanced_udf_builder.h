/**
 * 增强版UDF构建器
 * 实现基于曲率、颜色梯度、平面距离及局部密度的动态体素细化
 * 
 * 版本: 2.0
 * 日期: 2025-08-12
 */

#ifndef ENHANCED_UDF_BUILDER_H
#define ENHANCED_UDF_BUILDER_H

#include <memory>
#include <vector>
#include <unordered_map>
#include <openvdb/openvdb.h>
#include <openvdb/tree/Tree.h>
#include <openvdb/tree/LeafManager.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/features/principal_curvatures.h>
#include <Eigen/Dense>

namespace recon {

/**
 * 增强版UDF配置参数
 */
struct EnhancedUDFConfig {
    // 多级体素参数
    float base_voxel_size = 0.05f;       // 基础体素大小(5cm)
    float finest_voxel_size = 0.005f;    // 最细体素大小(5mm)
    float coarsest_voxel_size = 0.2f;    // 最粗体素大小(20cm)
    int max_refinement_levels = 4;       // 最大细化级别
    
    // 自适应细化阈值
    float curvature_threshold = 0.15f;           // 曲率阈值
    float color_gradient_threshold = 30.0f;      // 颜色梯度阈值(0-255)
    float plane_distance_threshold = 0.03f;      // 平面距离阈值(3cm)
    float density_variance_threshold = 2.0f;     // 密度方差阈值
    
    // 距离场参数
    float truncation_distance = 0.15f;   // 截断距离(15cm)
    float max_search_radius = 0.3f;      // 最大搜索半径(30cm)
    
    // 置信度权重
    float density_weight = 0.25f;        // 密度权重
    float color_weight = 0.25f;          // 颜色权重
    float planarity_weight = 0.25f;      // 平面性权重
    float normal_weight = 0.25f;         // 法向量权重
    
    // 平面拟合参数
    float plane_fitting_radius = 0.1f;   // 平面拟合半径(10cm)
    float plane_inlier_threshold = 0.01f; // 平面内点阈值(1cm)
    int min_plane_points = 10;           // 最少平面点数
    
    // 曲率估计参数
    float curvature_radius = 0.08f;      // 曲率估计半径(8cm)
    int min_curvature_neighbors = 15;    // 最少曲率邻居数
    
    // 性能优化参数
    bool use_parallel_processing = true;  // 使用并行处理
    int num_threads = 0;                 // 线程数(0=自动)
    bool use_octree_acceleration = true; // 使用八叉树加速
    
    // 滤波参数
    bool use_gaussian_filter = true;     // 使用高斯滤波
    float gaussian_width = 1.5f;         // 高斯滤波宽度
    int filter_iterations = 2;           // 滤波迭代次数
};

/**
 * 体素细化决策信息
 */
struct VoxelRefinementDecision {
    bool needs_refinement = false;
    float refinement_priority = 0.0f;    // 细化优先级[0,1]
    float target_voxel_size = 0.0f;      // 目标体素大小
    
    // 各项指标
    float curvature = 0.0f;
    float color_gradient = 0.0f;
    float plane_distance = 0.0f;
    float local_density = 0.0f;
    float normal_variation = 0.0f;
    
    // 局部特征
    bool is_planar_region = false;
    bool is_edge_region = false;
    bool is_corner_region = false;
    bool is_high_detail_region = false;
};

/**
 * 平面拟合结果
 */
struct PlaneInfo {
    Eigen::Vector4f coefficients;  // 平面方程系数 ax+by+cz+d=0
    float confidence = 0.0f;       // 拟合置信度
    int inlier_count = 0;          // 内点数量
    Eigen::Vector3f normal;        // 平面法向量
    Eigen::Vector3f centroid;      // 平面中心点
};

/**
 * 增强版UDF构建器
 */
class EnhancedUDFBuilder {
public:
    using PointT = pcl::PointXYZRGBNormal;
    using PointCloudT = pcl::PointCloud<PointT>;
    using GridT = openvdb::FloatGrid;
    using ConfidenceGridT = openvdb::FloatGrid;
    using RefinementGridT = openvdb::Int32Grid;  // 存储细化级别

    /**
     * 构造函数
     */
    explicit EnhancedUDFBuilder(const EnhancedUDFConfig& config = EnhancedUDFConfig{});
    
    /**
     * 析构函数
     */
    ~EnhancedUDFBuilder();

    /**
     * 从点云构建增强UDF
     */
    bool buildEnhancedUDF(PointCloudT::Ptr cloud, 
                         GridT::Ptr& udf_grid, 
                         ConfidenceGridT::Ptr& confidence_grid,
                         RefinementGridT::Ptr& refinement_grid);

    /**
     * 设置配置参数
     */
    void setConfig(const EnhancedUDFConfig& config) { config_ = config; }
    
    /**
     * 获取配置参数
     */
    const EnhancedUDFConfig& getConfig() const { return config_; }
    
    /**
     * 获取统计信息
     */
    struct Statistics {
        int total_voxels = 0;
        int refined_voxels = 0;
        int planar_voxels = 0;
        int edge_voxels = 0;
        int corner_voxels = 0;
        float avg_refinement_level = 0.0f;
        double build_time_seconds = 0.0;
    };
    
    const Statistics& getStatistics() const { return stats_; }

private:
    EnhancedUDFConfig config_;
    Statistics stats_;
    
    // 缓存的数据结构
    std::unique_ptr<pcl::KdTreeFLANN<PointT>> kdtree_;
    std::unordered_map<openvdb::Coord, PlaneInfo> plane_cache_;
    std::unordered_map<openvdb::Coord, float> curvature_cache_;
    
    /**
     * 初始化数据结构
     */
    bool initialize(const PointCloudT& cloud);
    
    /**
     * 构建多级自适应体素网格
     */
    bool buildAdaptiveGrid(const PointCloudT& cloud,
                          GridT::Ptr& udf_grid,
                          ConfidenceGridT::Ptr& confidence_grid,
                          RefinementGridT::Ptr& refinement_grid);
    
    /**
     * 计算体素细化决策
     */
    VoxelRefinementDecision computeRefinementDecision(const openvdb::Vec3f& world_pos,
                                                     const PointCloudT& cloud,
                                                     int current_level);
    
    /**
     * 递归细化体素
     */
    void refineVoxelRecursive(const openvdb::Vec3f& center,
                             float voxel_size,
                             int level,
                             const PointCloudT& cloud,
                             GridT::Ptr& udf_grid,
                             ConfidenceGridT::Ptr& confidence_grid,
                             RefinementGridT::Ptr& refinement_grid);
    
    /**
     * 计算增强的UDF值和置信度
     */
    std::pair<float, float> computeEnhancedVoxelValues(const openvdb::Vec3f& world_pos,
                                                      const PointCloudT& cloud,
                                                      int refinement_level);
    
    /**
     * 使用PCL进行平面拟合
     */
    PlaneInfo fitPlaneRANSAC(const openvdb::Vec3f& center,
                            const PointCloudT& cloud,
                            float radius);
    
    /**
     * 计算精确的局部曲率
     */
    float computePrincipalCurvature(const openvdb::Vec3f& pos, 
                                   const PointCloudT& cloud);
    
    /**
     * 计算增强的颜色梯度
     */
    float computeEnhancedColorGradient(const openvdb::Vec3f& pos, 
                                      const PointCloudT& cloud);
    
    /**
     * 计算到最近平面的精确距离
     */
    float computeAccuratePlaneDistance(const openvdb::Vec3f& pos, 
                                      const PointCloudT& cloud);
    
    /**
     * 计算局部密度统计
     */
    struct DensityStats {
        float density = 0.0f;
        float variance = 0.0f;
        float uniformity = 0.0f;  // 密度均匀性
    };
    
    DensityStats computeDensityStatistics(const openvdb::Vec3f& pos, 
                                         const PointCloudT& cloud);
    
    /**
     * 计算法向量变化
     */
    float computeNormalVariation(const openvdb::Vec3f& pos, 
                                const PointCloudT& cloud);
    
    /**
     * 计算增强置信度
     */
    float computeEnhancedConfidence(const openvdb::Vec3f& world_pos,
                                   const PointCloudT& cloud,
                                   const std::vector<int>& indices,
                                   const std::vector<float>& distances,
                                   int refinement_level);
    
    /**
     * 应用多级高斯滤波
     */
    void applyMultiLevelFiltering(GridT::Ptr grid, 
                                 const RefinementGridT& refinement_grid);
    
    /**
     * 优化网格拓扑
     */
    void optimizeGridTopology(GridT::Ptr grid);
    
    /**
     * 使用OpenVDB的动态树结构减少遍历
     */
    void optimizeTreeTraversal(GridT::Ptr grid);
    
    /**
     * 并行处理体素
     */
    void processVoxelsParallel(const std::vector<openvdb::Vec3f>& positions,
                              const PointCloudT& cloud,
                              GridT::Ptr& udf_grid,
                              ConfidenceGridT::Ptr& confidence_grid,
                              RefinementGridT::Ptr& refinement_grid);
};

/**
 * UDF构建器工厂类
 */
class UDFBuilderFactory {
public:
    enum class BuilderType {
        BASIC,      // 基础版本
        ENHANCED,   // 增强版本
        ADAPTIVE    // 自适应版本
    };
    
    static std::unique_ptr<EnhancedUDFBuilder> create(BuilderType type,
                                                     const EnhancedUDFConfig& config = EnhancedUDFConfig{});
};

} // namespace recon

#endif // ENHANCED_UDF_BUILDER_H

