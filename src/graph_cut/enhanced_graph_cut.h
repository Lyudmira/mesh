/**
 * 增强版图割求解器
 * 统一并优化图割求解，支持可重入、可并行处理
 * 
 * 版本: 2.0
 * 日期: 2025-08-12
 */

#ifndef ENHANCED_GRAPH_CUT_H
#define ENHANCED_GRAPH_CUT_H

#include "boykov_kolmogorov_solver.h"
#include <memory>
#include <vector>
#include <unordered_map>
#include <atomic>
#include <mutex>
#include <thread>
#include <openvdb/openvdb.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <Eigen/Dense>

namespace recon {

/**
 * 图割能量项类型
 */
enum class EnergyTermType {
    DATA_TERM,          // 数据项
    SMOOTHNESS_TERM,    // 平滑项
    VISIBILITY_TERM,    // 可见性项
    FREE_SPACE_TERM,    // 自由空间项
    PLANARITY_TERM,     // 平面性项
    COLOR_TERM          // 颜色项
};

/**
 * 图割配置参数
 */
struct GraphCutConfig {
    // 能量权重
    float data_weight = 1.0f;           // 数据项权重
    float smoothness_weight = 0.5f;     // 平滑项权重
    float visibility_weight = 0.3f;     // 可见性项权重
    float free_space_weight = 0.8f;     // 自由空间项权重
    float planarity_weight = 0.4f;      // 平面性项权重
    float color_weight = 0.2f;          // 颜色项权重
    
    // 求解器参数
    int max_iterations = 1000;          // 最大迭代次数
    float convergence_threshold = 1e-6f; // 收敛阈值
    bool use_parallel_solver = true;    // 使用并行求解器
    int num_threads = 0;                // 线程数(0=自动)
    
    // 内存优化参数
    bool use_sparse_representation = true;  // 使用稀疏表示
    bool enable_memory_mapping = false;     // 启用内存映射
    size_t max_memory_mb = 8192;            // 最大内存使用(MB)
    
    // 可见性计算参数
    float visibility_ray_step = 0.02f;      // 射线步长(2cm)
    int max_ray_samples = 100;              // 最大射线采样数
    float occlusion_threshold = 0.8f;       // 遮挡阈值
    
    // 自由空间参数
    float free_space_radius = 0.5f;         // 自由空间半径(50cm)
    int min_free_space_samples = 10;        // 最少自由空间采样数
    
    // 平面检测参数
    float plane_normal_threshold = 0.1f;    // 平面法向量阈值
    float plane_distance_threshold = 0.05f; // 平面距离阈值(5cm)
    
    // 性能优化
    bool use_hierarchical_solver = true;    // 使用分层求解器
    bool enable_early_termination = true;   // 启用早期终止
    float early_termination_threshold = 0.01f; // 早期终止阈值
};

/**
 * 图节点信息
 */
struct GraphNode {
    openvdb::Coord coord;           // 体素坐标
    openvdb::Vec3f world_pos;       // 世界坐标
    float data_cost_inside = 0.0f;  // 内部数据代价
    float data_cost_outside = 0.0f; // 外部数据代价
    float confidence = 1.0f;        // 置信度
    bool is_seed = false;           // 是否为种子点
    bool is_boundary = false;       // 是否为边界
    int cluster_id = -1;            // 聚类ID
};

/**
 * 图边信息
 */
struct GraphEdge {
    int node1_id = -1;              // 节点1 ID
    int node2_id = -1;              // 节点2 ID
    float weight = 0.0f;            // 边权重
    EnergyTermType term_type;       // 能量项类型
    openvdb::Vec3f direction;       // 边方向
};

/**
 * 求解结果
 */
struct GraphCutResult {
    std::vector<bool> labels;       // 节点标签(true=内部, false=外部)
    float total_energy = 0.0f;      // 总能量
    float data_energy = 0.0f;       // 数据能量
    float smoothness_energy = 0.0f; // 平滑能量
    int iterations = 0;             // 迭代次数
    double solve_time_seconds = 0.0; // 求解时间
    bool converged = false;         // 是否收敛
};

// 前向声明
class MaxFlowSolver;
struct EnergyParameterMapping;
using RefinementGridT = openvdb::Int32Grid;

/**
     * 重置求解器
     */
    virtual void reset() = 0;
    
    /**
     * 获取求解器名称
     */
    virtual std::string getName() const = 0;
};

/**
 * Kolmogorov最大流求解器实现
 */
class KolmogorovMaxFlowSolver : public MaxFlowSolver {
public:
    KolmogorovMaxFlowSolver();
    ~KolmogorovMaxFlowSolver() override;
    
    int addNode() override;
    void addEdge(int node1, int node2, float capacity12, float capacity21) override;
    void setTerminalWeights(int node, float source_weight, float sink_weight) override;
    float solve() override;
    bool getNodeLabel(int node) override;
    void reset() override;
    std::string getName() const override { return "Kolmogorov"; }

private:
    class Impl;
    std::unique_ptr<Impl> pimpl_;
};

/**
 * Boost Graph最大流求解器实现
 */
class BoostMaxFlowSolver : public MaxFlowSolver {
public:
    BoostMaxFlowSolver();
    ~BoostMaxFlowSolver() override;
    
    int addNode() override;
    void addEdge(int node1, int node2, float capacity12, float capacity21) override;
    void setTerminalWeights(int node, float source_weight, float sink_weight) override;
    float solve() override;
    bool getNodeLabel(int node) override;
    void reset() override;
    std::string getName() const override { return "Boost"; }

private:
    class Impl;
    std::unique_ptr<Impl> pimpl_;
};

/**
 * 增强版图割求解器
 */
class EnhancedGraphCutSolver {
public:
    using PointT = pcl::PointXYZRGBNormal;
    using PointCloudT = pcl::PointCloud<PointT>;
    using GridT = openvdb::FloatGrid;
    using ConfidenceGridT = openvdb::FloatGrid;

    /**
     * 构造函数
     */
    explicit EnhancedGraphCutSolver(const GraphCutConfig& config = GraphCutConfig{});
    
    /**
     * 析构函数
     */
    ~EnhancedGraphCutSolver();

    /**
     * 执行图割优化
     */
    bool solve(const GridT& udf_grid,
              const ConfidenceGridT& confidence_grid,
              const PointCloudT& cloud,
              GridT::Ptr& result_grid,
              GraphCutResult& result);
    
    /**
     * 执行集成版图割优化（使用UDF集成器）
     */
    bool solveWithUDFIntegration(
        const GridT& udf_grid,
        const ConfidenceGridT& confidence_grid,
        const RefinementGridT& refinement_grid,
        const PointCloudT& cloud,
        GridT::Ptr& result_grid,
        GraphCutResult& result
    );
    
    /**
     * 设置配置参数
     */
    void setConfig(const GraphCutConfig& config) { config_ = config; }
    
    /**
     * 获取配置参数
     */
    const GraphCutConfig& getConfig() const { return config_; }
    
    /**
     * 设置求解器类型
     */
    enum class SolverType {
        KOLMOGOROV,     // Kolmogorov最大流算法
        BOOST_GRAPH,    // Boost Graph库
        AUTO            // 自动选择
    };
    
    void setSolverType(SolverType type) { solver_type_ = type; }

private:
    GraphCutConfig config_;
    SolverType solver_type_ = SolverType::AUTO;
    
    // 图结构
    std::vector<GraphNode> nodes_;
    std::vector<GraphEdge> edges_;
    std::unordered_map<openvdb::Coord, int> coord_to_node_;
    
    // 求解器
    std::unique_ptr<MaxFlowSolver> solver_;
    
    // 线程安全
    mutable std::mutex graph_mutex_;
    std::atomic<bool> solving_{false};
    
    /**
     * 构建图结构
     */
    bool buildGraph(const GridT& udf_grid,
                   const ConfidenceGridT& confidence_grid,
                   const PointCloudT& cloud);
    
    /**
     * 添加图节点
     */
    int addGraphNode(const openvdb::Coord& coord,
                    const openvdb::Vec3f& world_pos,
                    const GridT& udf_grid,
                    const ConfidenceGridT& confidence_grid);
    
    /**
     * 添加图边
     */
    void addGraphEdges(const GridT& udf_grid,
                      const PointCloudT& cloud);
    
    /**
     * 计算数据项
     */
    std::pair<float, float> computeDataTerm(const GraphNode& node,
                                           const GridT& udf_grid,
                                           const ConfidenceGridT& confidence_grid);
    
    /**
     * 计算平滑项
     */
    float computeSmoothnessTerm(const GraphNode& node1,
                               const GraphNode& node2,
                               const PointCloudT& cloud);
    
    /**
     * 计算可见性项
     */
    float computeVisibilityTerm(const GraphNode& node,
                               const PointCloudT& cloud);
    
    /**
     * 计算自由空间项
     */
    float computeFreeSpaceTerm(const GraphNode& node,
                              const PointCloudT& cloud);
    
    /**
     * 计算平面性项
     */
    float computePlanarityTerm(const GraphNode& node1,
                              const GraphNode& node2,
                              const PointCloudT& cloud);
    
    /**
     * 计算颜色项
     */
    float computeColorTerm(const GraphNode& node1,
                          const GraphNode& node2,
                          const PointCloudT& cloud);
    
    /**
     * 射线投射检测可见性
     */
    bool isVisible(const openvdb::Vec3f& point,
                  const openvdb::Vec3f& camera_pos,
                  const PointCloudT& cloud);
    
    /**
     * 自动识别门窗开口
     */
    std::vector<openvdb::Vec3f> detectOpenings(const PointCloudT& cloud);
    
    /**
     * 选择最优求解器
     */
    std::unique_ptr<MaxFlowSolver> createOptimalSolver();
    
    /**
     * 计算区域自适应α权重
     */
    float computeAdaptiveAlpha(const GraphNode& node, const PointCloudT& cloud);
    
    /**
     * 计算区域自适应λ权重
     */
    float computeAdaptiveLambda(const GraphNode& node1, const GraphNode& node2, const PointCloudT& cloud);
    
    /**
     * 计算边缘强度
     */
    float computeEdgeStrength(const GraphNode& node1, const GraphNode& node2, const PointCloudT& cloud);
    
    /**
     * 计算可见性惩罚
     */
    float computeVisibilityPenalty(const GraphNode& node, const PointCloudT& cloud);
    
    /**
     * 估计扫描仪位置
     */
    std::vector<openvdb::Vec3f> estimateScannerPositions(const PointCloudT& cloud);
    
    /**
     * 检查点是否被遮挡
     */
    bool isPointOccluded(const openvdb::Vec3f& point, const PointCloudT& cloud);
    
    /**
     * 构建自适应图结构
     */
    bool buildAdaptiveGraph(
        const GridT& udf_grid,
        const ConfidenceGridT& confidence_grid,
        const EnergyParameterMapping& mapping,
        const PointCloudT& cloud
    );
    
    /**
     * 添加自适应图边
     */
    void addAdaptiveGraphEdges(
        const GridT& udf_grid,
        const EnergyParameterMapping& mapping,
        const PointCloudT& cloud
    );
    
    /**
     * 计算自适应边权重
     */
    float computeAdaptiveEdgeWeight(
        const GraphEdge& edge,
        const EnergyParameterMapping& mapping,
        const RefinementGridT& refinement_grid
    );
    
    /**
     * 计算内部代价
     */
    float computeInsideCost(const GraphNode& node, const GridT& udf_grid);
    
    /**
     * 计算外部代价
     */
    float computeOutsideCost(const GraphNode& node, const GridT& udf_grid);
    
    /**
     * 计算能量分解
     */
    void computeEnergyBreakdown(
        GraphCutResult& result,
        const EnergyParameterMapping& mapping
    );
    
    /**
     * 并行求解子图
     */
    void solveSubgraphsParallel(const std::vector<std::vector<int>>& subgraphs,
                               GraphCutResult& result);
    
    /**
     * 验证求解结果
     */
    bool validateResult(const GraphCutResult& result);
    
    /**
     * 优化内存使用
     */
    void optimizeMemoryUsage();
    
    /**
     * 更新任务进度
     */
    void updateProgress(float progress);
};

/**
 * 图割求解器工厂
 */
class GraphCutSolverFactory {
public:
    enum class SolverVariant {
        BASIC,              // 基础版本
        ENHANCED,           // 增强版本
        HIERARCHICAL,       // 分层版本
        PARALLEL            // 并行版本
    };
    
    static std::unique_ptr<EnhancedGraphCutSolver> create(
        SolverVariant variant,
        const GraphCutConfig& config = GraphCutConfig{});
    
    static std::unique_ptr<EnhancedGraphCutSolver> createStandard();
    static std::unique_ptr<EnhancedGraphCutSolver> createHighQuality();
    static std::unique_ptr<EnhancedGraphCutSolver> createFast();
};

/**
 * 可见性计算器
 */
class VisibilityCalculator {
public:
    using PointT = pcl::PointXYZRGBNormal;
    using PointCloudT = pcl::PointCloud<PointT>;

    /**
     * 构造函数
     */
    explicit VisibilityCalculator(const PointCloudT& cloud);
    
    /**
     * 计算点的可见性
     */
    float computeVisibility(const openvdb::Vec3f& point,
                           const std::vector<openvdb::Vec3f>& camera_positions);
    
    /**
     * 射线投射
     */
    bool raycast(const openvdb::Vec3f& origin,
                const openvdb::Vec3f& direction,
                float max_distance,
                openvdb::Vec3f& hit_point);
    
    /**
     * 批量可见性计算
     */
    std::vector<float> computeVisibilityBatch(const std::vector<openvdb::Vec3f>& points,
                                             const std::vector<openvdb::Vec3f>& camera_positions);

private:
    const PointCloudT& cloud_;
    std::unique_ptr<pcl::KdTreeFLANN<PointT>> kdtree_;
    
    // 射线投射加速结构
    struct OctreeNode {
        openvdb::Vec3f center;
        float size;
        std::vector<int> point_indices;
        std::vector<std::unique_ptr<OctreeNode>> children;
        bool is_leaf = true;
    };
    
    std::unique_ptr<OctreeNode> octree_root_;
    
    /**
     * 构建八叉树
     */
    void buildOctree();
    
    /**
     * 射线-八叉树相交测试
     */
    bool rayOctreeIntersect(const openvdb::Vec3f& origin,
                           const openvdb::Vec3f& direction,
                           const OctreeNode& node,
                           float& t_min,
                           float& t_max);
};

/**
 * 自由空间分析器
 */
class FreeSpaceAnalyzer {
public:
    using PointT = pcl::PointXYZRGBNormal;
    using PointCloudT = pcl::PointCloud<PointT>;

    /**
     * 构造函数
     */
    explicit FreeSpaceAnalyzer(const PointCloudT& cloud);
    
    /**
     * 检测自由空间种子点
     */
    std::vector<openvdb::Vec3f> detectFreeSpaceSeeds(const openvdb::FloatGrid& udf_grid);
    
    /**
     * 计算自由空间概率
     */
    float computeFreeSpaceProbability(const openvdb::Vec3f& point);
    
    /**
     * 自动识别门窗开口
     */
    struct Opening {
        openvdb::Vec3f center;
        openvdb::Vec3f normal;
        float width;
        float height;
        float confidence;
        enum Type { DOOR, WINDOW, OTHER } type;
    };
    
    std::vector<Opening> detectOpenings();

private:
    const PointCloudT& cloud_;
    std::unique_ptr<pcl::KdTreeFLANN<PointT>> kdtree_;
    
    /**
     * 泛洪填充检测连通区域
     */
    std::vector<std::vector<openvdb::Vec3f>> floodFillFreeSpace(const openvdb::FloatGrid& udf_grid);
    
    /**
     * 分析开口特征
     */
    Opening analyzeOpening(const std::vector<openvdb::Vec3f>& region_points);
};

/**
 * 能量项计算器
 */
class EnergyTermCalculator {
public:
    using PointT = pcl::PointXYZRGBNormal;
    using PointCloudT = pcl::PointCloud<PointT>;

    /**
     * 构造函数
     */
    explicit EnergyTermCalculator(const GraphCutConfig& config);
    
    /**
     * 计算所有能量项
     */
    float computeTotalEnergy(const GraphNode& node1,
                            const GraphNode& node2,
                            const PointCloudT& cloud,
                            EnergyTermType term_type);
    
    /**
     * 计算数据项能量
     */
    float computeDataEnergy(const GraphNode& node,
                           const openvdb::FloatGrid& udf_grid,
                           const openvdb::FloatGrid& confidence_grid);
    
    /**
     * 计算平滑项能量
     */
    float computeSmoothnessEnergy(const GraphNode& node1,
                                 const GraphNode& node2,
                                 const PointCloudT& cloud);
    
    /**
     * 计算可见性能量
     */
    float computeVisibilityEnergy(const GraphNode& node,
                                 const PointCloudT& cloud,
                                 const VisibilityCalculator& vis_calc);
    
    /**
     * 计算自由空间能量
     */
    float computeFreeSpaceEnergy(const GraphNode& node,
                                const FreeSpaceAnalyzer& fs_analyzer);

private:
    GraphCutConfig config_;
    
    /**
     * 归一化能量值
     */
    float normalizeEnergy(float energy, EnergyTermType type);
};

} // namespace recon

#endif // ENHANCED_GRAPH_CUT_H

