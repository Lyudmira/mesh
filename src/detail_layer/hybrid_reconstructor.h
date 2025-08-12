/**
 * 混合重建器
 * 对不同簇分别调用GP3、Poisson、RIMLS等算法并进行局部合并
 * 
 * 版本: 1.0
 * 日期: 2025-08-12
 */

#ifndef HYBRID_RECONSTRUCTOR_H
#define HYBRID_RECONSTRUCTOR_H

#include "adaptive_detail_selector.h"
#include <memory>
#include <vector>
#include <unordered_map>
#include <openvdb/openvdb.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/PolygonMesh.h>
#include <pcl/surface/gp3.h>
#include <pcl/surface/poisson.h>
#include <pcl/surface/marching_cubes_hoppe.h>
#include <pcl/surface/impl/mls.hpp>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <Eigen/Dense>

namespace recon {

/**
 * 混合重建配置
 */
struct HybridReconstructorConfig {
    // GP3参数
    struct GP3Config {
        float search_radius = 0.1f;         // 搜索半径
        float mu = 2.5f;                    // 最近邻距离倍数
        int maximum_nearest_neighbors = 100; // 最大近邻数
        float minimum_angle = M_PI/18;       // 最小角度(10度)
        float maximum_angle = 2*M_PI/3;      // 最大角度(120度)
        float maximum_surface_angle = M_PI/4; // 最大表面角度(45度)
        bool normal_consistency = false;     // 法向量一致性检查
    } gp3;
    
    // Poisson参数
    struct PoissonConfig {
        int depth = 8;                      // 八叉树深度
        float point_weight = 4.0f;          // 点权重
        float samples_per_node = 1.5f;      // 每节点采样数
        float scale = 1.1f;                 // 缩放因子
        bool is_density = false;            // 是否输出密度
        int solver_divide = 8;              // 求解器分割
        int iso_divide = 8;                 // 等值面分割
        float confidence = 0.0f;            // 置信度阈值
        bool manifold = true;               // 流形输出
        bool output_polygons = false;       // 输出多边形
    } poisson;
    
    // RIMLS参数
    struct RIMLSConfig {
        float search_radius = 0.08f;        // 搜索半径
        float step_size = 0.005f;           // 步长
        float gauss_param = 0.02f;          // 高斯参数
        bool use_polynomial_fit = true;     // 使用多项式拟合
        int polynomial_order = 2;           // 多项式阶数
        bool compute_normals = true;        // 计算法向量
        float normal_radius = 0.1f;         // 法向量计算半径
    } rimls;
    
    // 混合参数
    float transition_bandwidth = 0.05f;     // 过渡带宽
    float quality_weight = 0.6f;            // 质量权重
    float continuity_weight = 0.4f;         // 连续性权重
    bool enable_seamless_blending = true;   // 启用无缝混合
    bool use_distance_weighting = true;     // 使用距离权重
    
    // 后处理参数
    bool remove_duplicates = true;          // 移除重复顶点
    float duplicate_threshold = 1e-6f;      // 重复阈值
    bool smooth_boundaries = true;          // 平滑边界
    int smoothing_iterations = 3;           // 平滑迭代次数
    bool fill_holes = true;                 // 填充孔洞
    float max_hole_size = 0.1f;             // 最大孔洞大小
    
    // 性能参数
    bool use_parallel_processing = true;    // 并行处理
    int num_threads = 0;                    // 线程数
    bool use_memory_optimization = true;    // 内存优化
    size_t max_memory_mb = 4096;            // 最大内存(MB)
};

/**
 * 重建簇信息
 */
struct ReconstructionCluster {
    int cluster_id = -1;                    // 簇ID
    ReconstructionMethod method;            // 选择的重建方法
    PointCloudT::Ptr points;                // 簇点云
    pcl::PolygonMesh::Ptr mesh;             // 重建网格
    LocalQualityAssessment quality;         // 质量评估
    
    // 边界信息
    std::vector<int> boundary_indices;      // 边界点索引
    std::vector<Eigen::Vector3f> boundary_normals; // 边界法向量
    
    // 邻接信息
    std::vector<int> neighbor_clusters;     // 邻接簇ID
    std::unordered_map<int, float> transition_weights; // 过渡权重
    
    // 统计信息
    float reconstruction_time = 0.0f;       // 重建时间
    int vertex_count = 0;                   // 顶点数
    int face_count = 0;                     // 面数
    float mesh_quality_score = 0.0f;       // 网格质量分数
};

/**
 * 混合重建结果
 */
struct HybridReconstructionResult {
    pcl::PolygonMesh::Ptr final_mesh;      // 最终合并网格
    std::vector<ReconstructionCluster> clusters; // 重建簇
    
    // 统计信息
    int total_clusters = 0;
    int gp3_clusters = 0;
    int poisson_clusters = 0;
    int rimls_clusters = 0;
    int hybrid_clusters = 0;
    
    float total_reconstruction_time = 0.0f;
    float merging_time = 0.0f;
    float post_processing_time = 0.0f;
    
    int final_vertex_count = 0;
    int final_face_count = 0;
    float overall_quality_score = 0.0f;
    
    bool success = false;
    std::string error_message;
};

/**
 * 混合重建器
 */
class HybridReconstructor {
public:
    using PointT = pcl::PointXYZRGBNormal;
    using PointCloudT = pcl::PointCloud<PointT>;
    using GridT = openvdb::FloatGrid;

    /**
     * 构造函数
     */
    explicit HybridReconstructor(const HybridReconstructorConfig& config = HybridReconstructorConfig{});
    
    /**
     * 析构函数
     */
    ~HybridReconstructor();
    
    /**
     * 执行混合重建
     */
    bool reconstruct(
        const std::vector<ReconstructionCluster>& clusters,
        const GridT& shell_grid,
        HybridReconstructionResult& result
    );
    
    /**
     * 重建单个簇
     */
    bool reconstructCluster(
        ReconstructionCluster& cluster,
        const GridT& shell_grid
    );
    
    /**
     * 合并重建结果
     */
    bool mergeReconstructionResults(
        const std::vector<ReconstructionCluster>& clusters,
        pcl::PolygonMesh::Ptr& merged_mesh
    );
    
    /**
     * GP3重建
     */
    bool reconstructWithGP3(
        const PointCloudT& cloud,
        pcl::PolygonMesh::Ptr& mesh
    );
    
    /**
     * Poisson重建
     */
    bool reconstructWithPoisson(
        const PointCloudT& cloud,
        pcl::PolygonMesh::Ptr& mesh
    );
    
    /**
     * RIMLS重建
     */
    bool reconstructWithRIMLS(
        const PointCloudT& cloud,
        pcl::PolygonMesh::Ptr& mesh
    );
    
    /**
     * 自适应混合重建
     */
    bool reconstructWithHybrid(
        const PointCloudT& cloud,
        const LocalQualityAssessment& quality,
        pcl::PolygonMesh::Ptr& mesh
    );
    
    /**
     * 计算簇边界
     */
    void computeClusterBoundaries(ReconstructionCluster& cluster);
    
    /**
     * 计算簇邻接关系
     */
    void computeClusterAdjacency(std::vector<ReconstructionCluster>& clusters);
    
    /**
     * 无缝边界混合
     */
    bool blendBoundaries(
        std::vector<ReconstructionCluster>& clusters,
        pcl::PolygonMesh::Ptr& merged_mesh
    );
    
    /**
     * 后处理网格
     */
    bool postProcessMesh(pcl::PolygonMesh::Ptr& mesh);
    
    /**
     * 评估重建质量
     */
    float evaluateReconstructionQuality(
        const pcl::PolygonMesh& mesh,
        const PointCloudT& original_cloud
    );
    
    /**
     * 获取配置
     */
    const HybridReconstructorConfig& getConfig() const { return config_; }
    
    /**
     * 设置配置
     */
    void setConfig(const HybridReconstructorConfig& config) { config_ = config; }
    
    /**
     * 获取统计信息
     */
    const HybridReconstructionResult& getLastResult() const { return last_result_; }

private:
    HybridReconstructorConfig config_;
    HybridReconstructionResult last_result_;
    
    // 缓存和优化
    std::unique_ptr<pcl::KdTreeFLANN<PointT>> kdtree_;
    std::unordered_map<int, pcl::PolygonMesh::Ptr> mesh_cache_;
    
    /**
     * 初始化
     */
    bool initialize();
    
    /**
     * 预处理点云
     */
    PointCloudT::Ptr preprocessPointCloud(const PointCloudT& cloud);
    
    /**
     * 估计法向量
     */
    bool estimateNormals(PointCloudT::Ptr& cloud, float radius = 0.1f);
    
    /**
     * 创建GP3重建器
     */
    std::unique_ptr<pcl::GreedyProjectionTriangulation<PointT>> createGP3Reconstructor();
    
    /**
     * 创建Poisson重建器
     */
    std::unique_ptr<pcl::Poisson<PointT>> createPoissonReconstructor();
    
    /**
     * 创建RIMLS重建器
     */
    bool setupRIMLSReconstruction();
    
    /**
     * 计算过渡权重
     */
    float computeTransitionWeight(
        const Eigen::Vector3f& point,
        const ReconstructionCluster& cluster1,
        const ReconstructionCluster& cluster2
    );
    
    /**
     * 插值网格顶点
     */
    Eigen::Vector3f interpolateVertices(
        const std::vector<Eigen::Vector3f>& vertices,
        const std::vector<float>& weights
    );
    
    /**
     * 检测网格缺陷
     */
    bool detectMeshDefects(const pcl::PolygonMesh& mesh);
    
    /**
     * 修复网格缺陷
     */
    bool repairMeshDefects(pcl::PolygonMesh::Ptr& mesh);
    
    /**
     * 移除重复顶点
     */
    bool removeDuplicateVertices(pcl::PolygonMesh::Ptr& mesh);
    
    /**
     * 平滑网格边界
     */
    bool smoothMeshBoundaries(pcl::PolygonMesh::Ptr& mesh);
    
    /**
     * 填充小孔洞
     */
    bool fillSmallHoles(pcl::PolygonMesh::Ptr& mesh);
    
    /**
     * 优化网格拓扑
     */
    bool optimizeMeshTopology(pcl::PolygonMesh::Ptr& mesh);
    
    /**
     * 验证网格质量
     */
    bool validateMeshQuality(const pcl::PolygonMesh& mesh);
    
    /**
     * 更新统计信息
     */
    void updateStatistics(HybridReconstructionResult& result);
    
    /**
     * 清理缓存
     */
    void cleanupCache();
};

/**
 * 混合重建器工厂
 */
class HybridReconstructorFactory {
public:
    /**
     * 创建标准重建器
     */
    static std::unique_ptr<HybridReconstructor> createStandardReconstructor();
    
    /**
     * 创建高质量重建器
     */
    static std::unique_ptr<HybridReconstructor> createHighQualityReconstructor();
    
    /**
     * 创建快速重建器
     */
    static std::unique_ptr<HybridReconstructor> createFastReconstructor();
    
    /**
     * 创建内存优化重建器
     */
    static std::unique_ptr<HybridReconstructor> createMemoryOptimizedReconstructor();
};

} // namespace recon

#endif // HYBRID_RECONSTRUCTOR_H

