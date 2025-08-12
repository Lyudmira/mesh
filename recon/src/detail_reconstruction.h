/**
 * 细节重建模块头文件
 * 负责在外壳偏移带内重建高保真细节
 * 
 * 版本: 1.0
 * 日期: 2025-08-12
 */

#ifndef DETAIL_RECONSTRUCTION_H
#define DETAIL_RECONSTRUCTION_H

#include <memory>
#include <vector>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/PolygonMesh.h>
#include <pcl/surface/gp3.h>
#include <pcl/surface/marching_cubes_hoppe.h>
#include <pcl/kdtree/kdtree_flann.h>

namespace recon {

/**
 * 细节重建方法枚举
 */
enum class DetailMethod {
    GP3,        // 贪婪投影三角化
    RIMLS,      // 鲁棒隐式移动最小二乘
    POISSON     // 泊松重建
};

/**
 * 细节重建配置参数
 */
struct DetailReconstructionConfig {
    // 几何提取参数
    float offset_distance = 0.08f;         // 偏移带距离(8厘米)
    bool outside_only = true;              // 仅提取外壳外侧点
    
    // 去噪参数
    bool enable_denoising = true;          // 启用去噪
    std::string denoising_method = "bilateral";  // bilateral, rimls, wlop
    float radius_multiplier = 1.5f;        // 半径倍数
    
    // 重建方法选择
    DetailMethod primary_method = DetailMethod::GP3;
    
    // GP3参数
    struct GP3Config {
        float edge_length_multiplier = 3.0f;       // 边长倍数
        float mu = 2.8f;                           // μ参数
        float angle_threshold_planar = 45.0f;      // 平面区域角度阈值(度)
        float angle_threshold_edge = 30.0f;        // 边缘区域角度阈值(度)
        float min_angle = 12.0f;                   // 最小三角形角度(度)
        float max_angle = 115.0f;                  // 最大三角形角度(度)
        int max_nearest_neighbors = 100;           // 最大近邻数
        bool normal_consistency = true;            // 法向量一致性
    } gp3;
    
    // RIMLS参数
    struct RIMLSConfig {
        float bandwidth = 1.6f;                    // 带宽
        float voxel_size = 0.01f;                  // 体素大小(1厘米)
        int smoothing_steps = 5;                   // 平滑步数
    } rimls;
    
    // 后处理参数
    struct PostProcessingConfig {
        bool remove_hanging_edges = true;         // 移除悬挂边
        bool filter_small_components = true;      // 过滤小组件
        int min_component_size = 50;              // 最小组件大小
        
        // PMP修复
        bool enable_pmp_repair = true;            // 启用PMP修复
        bool fix_self_intersections = true;       // 修复自相交
        bool fix_non_manifold = true;             // 修复非流形
        
        // 简化
        bool enable_simplification = true;        // 启用简化
        std::string simplification_method = "quadric_edge_collapse";
        bool preserve_sharp_edges = true;         // 保持尖锐边缘
        float angle_threshold = 60.0f;            // 角度阈值(度)
        float reduction_ratio = 0.9f;             // 简化比例
    } post_processing;
    
    // 质量控制
    struct QualityControlConfig {
        bool check_manifold = true;               // 检查流形
        bool check_orientation = true;            // 检查方向
        float max_edge_length = 0.1f;             // 最大边长(10厘米)
        float sharp_edge_threshold = 45.0f;       // 尖锐边缘阈值(度)
        bool preserve_boundaries = true;          // 保持边界
        bool preserve_corners = true;             // 保持角点
    } quality_control;
};

/**
 * 细节重建器类
 */
class DetailReconstructor {
public:
    using PointT = pcl::PointXYZRGBNormal;
    using PointCloudT = pcl::PointCloud<PointT>;

    /**
     * 构造函数
     */
    explicit DetailReconstructor(const DetailReconstructionConfig& config = DetailReconstructionConfig{});
    
    /**
     * 析构函数
     */
    ~DetailReconstructor();

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
    void setConfig(const DetailReconstructionConfig& config) { config_ = config; }
    
    /**
     * 获取配置参数
     */
    const DetailReconstructionConfig& getConfig() const { return config_; }

    /**
     * 获取重建统计信息
     */
    struct Statistics {
        int original_points = 0;
        int extracted_points = 0;
        int denoised_points = 0;
        int output_vertices = 0;
        int output_triangles = 0;
        int removed_components = 0;
        double extraction_time = 0.0;
        double reconstruction_time = 0.0;
        double post_processing_time = 0.0;
    };
    
    const Statistics& getStatistics() const { return stats_; }

private:
    DetailReconstructionConfig config_;
    Statistics stats_;
    float base_radius_ = 0.02f;
    
    /**
     * 从外壳网格提取偏移带内的点
     */
    bool extractOffsetBandPoints(const pcl::PolygonMesh& shell_mesh,
                                const PointCloudT::Ptr& original_cloud,
                                PointCloudT::Ptr& extracted_points);
    
    /**
     * 对提取的点进行去噪
     */
    bool denoisePoints(const PointCloudT::Ptr& input_points,
                      PointCloudT::Ptr& denoised_points);
    
    /**
     * 使用GP3方法重建
     */
    bool reconstructWithGP3(const PointCloudT::Ptr& points,
                           pcl::PolygonMesh& mesh);
    
    /**
     * 使用RIMLS方法重建
     */
    bool reconstructWithRIMLs(const PointCloudT::Ptr& points,
                             pcl::PolygonMesh& mesh);
    
    /**
     * 使用泊松重建方法
     */
    bool reconstructWithPoisson(const PointCloudT::Ptr& points,
                               pcl::PolygonMesh& mesh);
    
    /**
     * 后处理网格
     */
    bool postProcessMesh(pcl::PolygonMesh& mesh);
    
    /**
     * 移除悬挂边缘
     */
    bool removeHangingEdges(pcl::PolygonMesh& mesh);
    
    /**
     * 过滤小组件
     */
    bool filterSmallComponents(pcl::PolygonMesh& mesh);
    
    /**
     * PMP修复
     */
    bool repairWithPMP(pcl::PolygonMesh& mesh);
    
    /**
     * 网格简化
     */
    bool simplifyMesh(pcl::PolygonMesh& mesh);
    
    /**
     * 质量控制检查
     */
    bool performQualityControl(const pcl::PolygonMesh& mesh);
    
    /**
     * 计算点到网格的距离
     */
    float computePointToMeshDistance(const PointT& point, const pcl::PolygonMesh& mesh);
    
    /**
     * 检查点是否在外壳外侧
     */
    bool isPointOutsideShell(const PointT& point, const pcl::PolygonMesh& shell_mesh);
    
    /**
     * 双边滤波去噪
     */
    bool bilateralFilter(const PointCloudT::Ptr& input, PointCloudT::Ptr& output);
    
    /**
     * WLOP去噪
     */
    bool wlopDenoising(const PointCloudT::Ptr& input, PointCloudT::Ptr& output);
    
    /**
     * 自适应参数计算
     */
    void computeAdaptiveParameters(const PointCloudT::Ptr& points);
    
    /**
     * 检测平面区域和边缘区域
     */
    std::vector<bool> detectPlanarRegions(const PointCloudT::Ptr& points);
    
    /**
     * 计算局部密度
     */
    std::vector<float> computeLocalDensity(const PointCloudT::Ptr& points);
};

/**
 * 融合器类
 * 负责将外壳网格和细节网格融合
 */
class MeshFuser {
public:
    /**
     * 融合配置参数
     */
    struct FusionConfig {
        bool shell_priority = true;               // 外壳优先
        float detail_threshold = 0.005f;          // 细节阈值(5毫米)
        
        // 布尔运算
        std::string boolean_method = "union";     // union, difference
        bool use_cgal = true;                     // 使用CGAL
        float tolerance = 0.001f;                 // 容差(1毫米)
        
        // Alpha包装(替代方案)
        bool enable_alpha_fallback = true;       // 启用Alpha包装后备
        float alpha_offset = 0.002f;             // Alpha偏移(2毫米)
        bool post_align_planes = true;           // 后对齐平面
        
        // 焊接参数
        float welding_threshold = 0.002f;        // 焊接阈值(2毫米)
        bool preserve_boundaries = true;         // 保持边界
        bool recompute_normals = true;           // 重新计算法向量
        
        // 颜色分配
        struct ColorConfig {
            std::string shell_method = "plane_sampling";  // 外壳颜色方法
            bool distance_weight = true;                  // 距离权重
            float neighborhood_size = 0.05f;              // 邻域大小(5厘米)
            
            std::string detail_method = "inherit_from_points";  // 细节颜色方法
            bool fallback_to_interpolation = true;              // 后备插值
            
            bool enable_blending = true;                  // 启用混合
            float shell_weight = 0.7f;                    // 外壳权重
            float detail_weight = 0.3f;                   // 细节权重
            float transition_distance = 0.01f;            // 过渡距离(1厘米)
        } color;
    };

    /**
     * 构造函数
     */
    explicit MeshFuser(const FusionConfig& config = FusionConfig{});
    
    /**
     * 融合外壳网格和细节网格
     */
    bool fuseMeshes(const pcl::PolygonMesh& shell_mesh,
                   const pcl::PolygonMesh& detail_mesh,
                   const PointCloudT::Ptr& original_cloud,
                   pcl::PolygonMesh& fused_mesh);

    /**
     * 设置配置参数
     */
    void setConfig(const FusionConfig& config) { config_ = config; }

private:
    FusionConfig config_;
    
    /**
     * 布尔运算融合
     */
    bool booleanUnion(const pcl::PolygonMesh& mesh1,
                     const pcl::PolygonMesh& mesh2,
                     pcl::PolygonMesh& result);
    
    /**
     * Alpha包装融合
     */
    bool alphaWrapping(const pcl::PolygonMesh& shell_mesh,
                      const pcl::PolygonMesh& detail_mesh,
                      pcl::PolygonMesh& result);
    
    /**
     * 顶点焊接
     */
    bool weldVertices(pcl::PolygonMesh& mesh);
    
    /**
     * 颜色分配
     */
    bool assignColors(pcl::PolygonMesh& mesh, const PointCloudT::Ptr& original_cloud);
};

} // namespace recon

#endif // DETAIL_RECONSTRUCTION_H

