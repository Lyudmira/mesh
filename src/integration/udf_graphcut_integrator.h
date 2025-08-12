/**
 * UDF与图割集成器
 * 建立体素属性到图割能量参数的映射关系
 * 
 * 版本: 1.0
 * 日期: 2025-08-12
 */

#ifndef UDF_GRAPHCUT_INTEGRATOR_H
#define UDF_GRAPHCUT_INTEGRATOR_H

#include "../udf_builder/enhanced_udf_builder.h"
#include "../graph_cut/enhanced_graph_cut.h"
#include <memory>
#include <vector>
#include <unordered_map>
#include <openvdb/openvdb.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

namespace recon {

/**
 * 集成配置参数
 */
struct IntegrationConfig {
    // 权重映射参数
    float alpha_base = 1.0f;              // α基础权重
    float alpha_curvature_factor = 0.5f;  // 曲率对α的影响因子
    float alpha_planar_boost = 1.5f;      // 平面区域α增强
    float alpha_edge_reduction = 0.3f;    // 边缘区域α削减
    
    float lambda_base = 0.5f;             // λ基础权重
    float lambda_planar_boost = 2.0f;     // 平面区域λ增强
    float lambda_edge_reduction = 0.2f;   // 边缘区域λ削减
    float lambda_color_factor = 0.3f;     // 颜色一致性对λ的影响
    
    // 区域分类阈值
    float planar_threshold = 0.7f;        // 平面区域阈值
    float edge_threshold = 0.5f;          // 边缘区域阈值
    float corner_threshold = 0.8f;        // 角点区域阈值
    float detail_threshold = 0.6f;        // 细节区域阈值
    
    // 细化级别权重
    std::vector<float> refinement_weights = {1.0f, 0.8f, 0.6f, 0.4f, 0.2f}; // 各级别权重
    
    // 置信度阈值
    float min_confidence = 0.1f;          // 最小置信度
    float high_confidence = 0.8f;         // 高置信度阈值
    
    // 自适应参数
    bool enable_adaptive_weighting = true; // 启用自适应权重
    bool use_refinement_guidance = true;   // 使用细化指导
    bool apply_confidence_scaling = true;  // 应用置信度缩放
};

/**
 * 体素属性信息
 */
struct VoxelAttributes {
    // 几何属性
    float curvature = 0.0f;
    float color_gradient = 0.0f;
    float plane_distance = 0.0f;
    float local_density = 0.0f;
    float normal_variation = 0.0f;
    
    // 区域分类
    bool is_planar = false;
    bool is_edge = false;
    bool is_corner = false;
    bool is_detail = false;
    
    // 细化信息
    int refinement_level = 0;
    float refinement_priority = 0.0f;
    
    // 置信度
    float confidence = 1.0f;
    
    // 图割参数
    float alpha_weight = 1.0f;
    float lambda_weight = 0.5f;
    float visibility_penalty = 0.0f;
};

/**
 * 能量参数映射结果
 */
struct EnergyParameterMapping {
    std::unordered_map<openvdb::Coord, VoxelAttributes> voxel_attributes;
    
    // 统计信息
    int total_voxels = 0;
    int planar_voxels = 0;
    int edge_voxels = 0;
    int corner_voxels = 0;
    int detail_voxels = 0;
    
    float avg_alpha = 1.0f;
    float avg_lambda = 0.5f;
    float avg_confidence = 1.0f;
};

/**
 * UDF与图割集成器
 */
class UDFGraphCutIntegrator {
public:
    using PointT = pcl::PointXYZRGBNormal;
    using PointCloudT = pcl::PointCloud<PointT>;
    using GridT = openvdb::FloatGrid;
    using ConfidenceGridT = openvdb::FloatGrid;
    using RefinementGridT = openvdb::Int32Grid;

    /**
     * 构造函数
     */
    explicit UDFGraphCutIntegrator(const IntegrationConfig& config = IntegrationConfig{});
    
    /**
     * 析构函数
     */
    ~UDFGraphCutIntegrator();
    
    /**
     * 执行UDF与图割集成
     */
    bool integrateUDFWithGraphCut(
        const GridT& udf_grid,
        const ConfidenceGridT& confidence_grid,
        const RefinementGridT& refinement_grid,
        const PointCloudT& cloud,
        EnergyParameterMapping& mapping
    );
    
    /**
     * 计算体素属性
     */
    VoxelAttributes computeVoxelAttributes(
        const openvdb::Coord& coord,
        const GridT& udf_grid,
        const ConfidenceGridT& confidence_grid,
        const RefinementGridT& refinement_grid,
        const PointCloudT& cloud
    );
    
    /**
     * 建立能量参数映射
     */
    bool buildEnergyParameterMapping(
        const std::vector<openvdb::Coord>& voxel_coords,
        const GridT& udf_grid,
        const ConfidenceGridT& confidence_grid,
        const RefinementGridT& refinement_grid,
        const PointCloudT& cloud,
        EnergyParameterMapping& mapping
    );
    
    /**
     * 应用自适应权重调整
     */
    void applyAdaptiveWeighting(VoxelAttributes& attributes) const;
    
    /**
     * 基于细化网格设置光滑项权重
     */
    float computeRefinementBasedSmoothness(
        const openvdb::Coord& coord1,
        const openvdb::Coord& coord2,
        const RefinementGridT& refinement_grid
    ) const;
    
    /**
     * 计算区域分类
     */
    void classifyVoxelRegion(VoxelAttributes& attributes) const;
    
    /**
     * 获取配置参数
     */
    const IntegrationConfig& getConfig() const { return config_; }
    
    /**
     * 设置配置参数
     */
    void setConfig(const IntegrationConfig& config) { config_ = config; }
    
    /**
     * 获取统计信息
     */
    const EnergyParameterMapping& getLastMapping() const { return last_mapping_; }

private:
    IntegrationConfig config_;
    EnergyParameterMapping last_mapping_;
    
    // 缓存和优化
    std::unique_ptr<pcl::KdTreeFLANN<PointT>> kdtree_;
    std::unordered_map<openvdb::Coord, VoxelAttributes> attribute_cache_;
    
    /**
     * 初始化
     */
    bool initialize(const PointCloudT& cloud);
    
    /**
     * 计算曲率属性
     */
    float computeCurvatureAttribute(
        const openvdb::Vec3f& world_pos,
        const PointCloudT& cloud
    ) const;
    
    /**
     * 计算颜色梯度属性
     */
    float computeColorGradientAttribute(
        const openvdb::Vec3f& world_pos,
        const PointCloudT& cloud
    ) const;
    
    /**
     * 计算平面距离属性
     */
    float computePlaneDistanceAttribute(
        const openvdb::Vec3f& world_pos,
        const PointCloudT& cloud
    ) const;
    
    /**
     * 计算局部密度属性
     */
    float computeLocalDensityAttribute(
        const openvdb::Vec3f& world_pos,
        const PointCloudT& cloud
    ) const;
    
    /**
     * 计算法向量变化属性
     */
    float computeNormalVariationAttribute(
        const openvdb::Vec3f& world_pos,
        const PointCloudT& cloud
    ) const;
    
    /**
     * 计算α权重映射
     */
    float computeAlphaMapping(const VoxelAttributes& attributes) const;
    
    /**
     * 计算λ权重映射
     */
    float computeLambdaMapping(const VoxelAttributes& attributes) const;
    
    /**
     * 计算可见性惩罚映射
     */
    float computeVisibilityPenaltyMapping(
        const openvdb::Vec3f& world_pos,
        const VoxelAttributes& attributes,
        const PointCloudT& cloud
    ) const;
    
    /**
     * 应用置信度缩放
     */
    void applyConfidenceScaling(VoxelAttributes& attributes) const;
    
    /**
     * 验证映射结果
     */
    bool validateMapping(const EnergyParameterMapping& mapping) const;
    
    /**
     * 优化缓存使用
     */
    void optimizeCacheUsage();
    
    /**
     * 更新统计信息
     */
    void updateStatistics(EnergyParameterMapping& mapping) const;
};

/**
 * 集成器工厂
 */
class IntegratorFactory {
public:
    /**
     * 创建标准集成器
     */
    static std::unique_ptr<UDFGraphCutIntegrator> createStandardIntegrator();
    
    /**
     * 创建高精度集成器
     */
    static std::unique_ptr<UDFGraphCutIntegrator> createHighPrecisionIntegrator();
    
    /**
     * 创建快速集成器
     */
    static std::unique_ptr<UDFGraphCutIntegrator> createFastIntegrator();
};

} // namespace recon

#endif // UDF_GRAPHCUT_INTEGRATOR_H

