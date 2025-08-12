/**
 * UDF构建模块头文件
 * 负责将点云转换为无符号距离场(UDF)
 * 
 * 版本: 1.0
 * 日期: 2025-08-12
 */

#ifndef UDF_BUILDER_H
#define UDF_BUILDER_H

#include <memory>
#include <vector>
#include <openvdb/openvdb.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

namespace recon {

/**
 * UDF构建器配置参数
 */
struct UDFConfig {
    // 体素化参数
    float coarse_voxel_size = 0.03f;    // 粗体素大小(3厘米)
    float fine_voxel_size = 0.01f;      // 细体素大小(1厘米)
    
    // 细化条件
    float curvature_threshold = 0.1f;           // 曲率阈值
    float color_gradient_threshold = 0.2f;      // 颜色梯度阈值
    float plane_distance_threshold = 0.05f;     // 平面距离阈值(5厘米)
    float density_threshold_multiplier = 2.0f;  // 密度阈值倍数
    
    // 距离场参数
    float truncation_distance = 0.09f;  // 截断距离(9厘米)
    
    // 置信度权重
    float density_weight = 0.4f;        // 密度权重
    float color_weight = 0.3f;          // 颜色权重
    float planarity_weight = 0.3f;      // 平面性权重
    
    // 膨胀带参数
    float expansion_distance = 0.4f;     // 膨胀距离(40厘米)
    
    // 滤波参数
    bool use_gaussian_filter = true;     // 使用高斯滤波
    float gaussian_width = 1.0f;         // 高斯滤波宽度
};

/**
 * 体素细化信息
 */
struct VoxelRefinementInfo {
    bool needs_refinement = false;
    float curvature = 0.0f;
    float color_gradient = 0.0f;
    float plane_distance = 0.0f;
    float local_density = 0.0f;
};

/**
 * UDF构建器类
 */
class UDFBuilder {
public:
    using PointT = pcl::PointXYZRGBNormal;
    using PointCloudT = pcl::PointCloud<PointT>;
    using GridT = openvdb::FloatGrid;
    using ConfidenceGridT = openvdb::FloatGrid;

    /**
     * 构造函数
     */
    explicit UDFBuilder(const UDFConfig& config = UDFConfig{});
    
    /**
     * 析构函数
     */
    ~UDFBuilder();

    /**
     * 从点云构建UDF
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
    void setConfig(const UDFConfig& config) { config_ = config; }
    
    /**
     * 获取配置参数
     */
    const UDFConfig& getConfig() const { return config_; }

private:
    UDFConfig config_;
    
    /**
     * 初始化体素网格
     */
    bool initializeGrids(const PointCloudT& cloud, 
                        GridT::Ptr& udf_grid, 
                        ConfidenceGridT::Ptr& confidence_grid);
    
    /**
     * 计算体素细化信息
     */
    VoxelRefinementInfo computeRefinementInfo(const openvdb::Coord& coord,
                                             const PointCloudT& cloud,
                                             const GridT& grid);
    
    /**
     * 检查是否需要细化
     */
    bool needsRefinement(const VoxelRefinementInfo& info);
    
    /**
     * 计算体素的UDF值和置信度
     */
    std::pair<float, float> computeVoxelValues(const openvdb::Vec3f& world_pos,
                                              const PointCloudT& cloud);
    
    /**
     * 计算点到体素中心的距离
     */
    float computePointDistance(const openvdb::Vec3f& voxel_center,
                              const PointT& point);
    
    /**
     * 计算置信度
     */
    float computeConfidence(const openvdb::Vec3f& world_pos,
                           const PointCloudT& cloud,
                           float distance);
    
    /**
     * 应用高斯滤波
     */
    void applyGaussianFilter(GridT::Ptr grid);
    
    /**
     * 计算局部曲率
     */
    float computeLocalCurvature(const openvdb::Vec3f& pos, const PointCloudT& cloud);
    
    /**
     * 计算颜色梯度
     */
    float computeColorGradient(const openvdb::Vec3f& pos, const PointCloudT& cloud);
    
    /**
     * 计算到最近平面的距离
     */
    float computePlaneDistance(const openvdb::Vec3f& pos, const PointCloudT& cloud);
    
    /**
     * 计算局部密度
     */
    float computeLocalDensity(const openvdb::Vec3f& pos, const PointCloudT& cloud);
};

} // namespace recon

#endif // UDF_BUILDER_H

