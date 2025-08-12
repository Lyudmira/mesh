/**
 * 图割优化模块头文件
 * 负责通过图割算法优化UDF的内外分割
 * 
 * 版本: 2.0 - 集成PyMaxflow高性能求解器
 * 日期: 2025-08-12
 */

#ifndef GRAPH_CUT_H
#define GRAPH_CUT_H

#include <memory>
#include <vector>
#include <unordered_map>
#include <openvdb/openvdb.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include "pymaxflow_solver.h"  // 引入PyMaxflow求解器

namespace recon {

/**
 * 图割标签枚举
 */
enum class Label : uint8_t {
    FREE = 0,    // 自由空间
    INSIDE = 1   // 内部空间
};

/**
 * 图割配置参数
 */
struct GraphCutConfig {
    // 数据项参数
    float inside_cost_multiplier = 1.0f;    // 内部成本倍数
    float free_cost_base = 0.5f;            // 自由空间基础成本
    float visibility_weight = 0.3f;         // 可见性权重
    
    // 平滑项参数
    float base_weight = 0.8f;               // 基础平滑权重
    float planar_multiplier = 2.0f;         // 平面区域倍数
    float cross_plane_multiplier = 0.5f;    // 跨平面倍数
    bool density_adaptive = true;           // 密度自适应
    
    // 区域自适应参数
    float planar_lambda = 1.0f;             // 平面区域lambda
    float detail_lambda = 0.5f;             // 细节区域lambda
    float planar_alpha = 0.7f;              // 平面区域alpha
    float detail_alpha = 0.5f;              // 细节区域alpha
    
    // 自由空间种子参数
    float seed_spacing = 0.8f;              // 种子间距(米)
    float min_distance_to_surface = 0.25f;  // 到表面最小距离(米)
    
    // 连通性参数
    int connectivity = 6;                   // 6, 18, 或 26连通
};

/**
 * 体素节点信息
 */
struct VoxelNode {
    openvdb::Coord coord;           // 体素坐标
    float udf_value;                // UDF值
    float confidence;               // 置信度
    bool is_planar;                 // 是否为平面区域
    float local_density;            // 局部密度
    bool is_free_region = false;    // 是否可达自由空间
    float visibility_cost = 0.0f;   // 可见性惩罚
    Label label = Label::FREE;      // 分配的标签
};

/**
 * 图边信息
 */
struct GraphEdge {
    int node1, node2;              // 节点索引
    float weight;                  // 边权重
    bool is_planar_connection;     // 是否为平面内连接
};

/**
 * 图割优化器类
 */
class GraphCutOptimizer {
public:
    using PointT = pcl::PointXYZRGBNormal;
    using PointCloudT = pcl::PointCloud<PointT>;
    using UDFGridT = openvdb::FloatGrid;
    using ConfidenceGridT = openvdb::FloatGrid;
    using LabelGridT = openvdb::Int32Grid;

    /**
     * 构造函数
     */
    explicit GraphCutOptimizer(const GraphCutConfig& config = GraphCutConfig{});
    
    /**
     * 析构函数
     */
    ~GraphCutOptimizer();

    /**
     * 执行图割优化
     * @param udf_grid UDF网格
     * @param confidence_grid 置信度网格
     * @param cloud 原始点云
     * @param label_grid 输出标签网格
     * @return 成功返回true
     */
    bool optimize(const UDFGridT::Ptr& udf_grid,
                  const ConfidenceGridT::Ptr& confidence_grid,
                  const PointCloudT::Ptr& cloud,
                  LabelGridT::Ptr& label_grid);

    /**
     * 设置配置参数
     */
    void setConfig(const GraphCutConfig& config) { config_ = config; }
    
    /**
     * 获取配置参数
     */
    const GraphCutConfig& getConfig() const { return config_; }

    /**
     * 获取优化统计信息
     */
    struct Statistics {
        int total_nodes = 0;
        int total_edges = 0;
        int inside_nodes = 0;
        int free_nodes = 0;
        float max_flow_value = 0.0f;
        double optimization_time = 0.0;
    };
    
    const Statistics& getStatistics() const { return stats_; }

private:
    GraphCutConfig config_;
    Statistics stats_;
    
    std::vector<VoxelNode> nodes_;
    std::vector<GraphEdge> edges_;
    std::unordered_map<openvdb::Coord, int, openvdb::Coord::Hash> coord_to_node_;
    openvdb::math::Transform::Ptr transform_;
    PointCloudT::Ptr cloud_;
    pcl::KdTreeFLANN<PointT> kdtree_;
    std::vector<openvdb::Coord> free_seeds_;
    float max_density_ = 1.0f;
    
    /**
     * 构建图结构
     */
    bool buildGraph(const UDFGridT::Ptr& udf_grid,
                    const ConfidenceGridT::Ptr& confidence_grid,
                    const PointCloudT::Ptr& cloud);
    
    /**
     * 添加体素节点
     */
    void addVoxelNode(const openvdb::Coord& coord,
                      float udf_value,
                      float confidence,
                      const PointCloudT::Ptr& cloud);
    
    /**
     * 构建边连接
     */
    void buildEdges();
    
    /**
     * 计算数据项成本
     */
    std::pair<float, float> computeDataCosts(const VoxelNode& node,
                                           const PointCloudT::Ptr& cloud);
    
    /**
     * 计算平滑项权重
     */
    float computeSmoothWeight(const VoxelNode& node1, const VoxelNode& node2);
    
    /**
     * 检测平面区域
     */
    bool isPlanarRegion(const openvdb::Coord& coord, const PointCloudT::Ptr& cloud);
    
    /**
     * 计算局部密度
     */
    float computeLocalDensity(const openvdb::Coord& coord, const PointCloudT::Ptr& cloud);
    
    /**
     * 生成自由空间种子
     */
    std::vector<openvdb::Coord> generateFreeSpaceSeeds(const PointCloudT::Ptr& cloud);
    
    /**
     * 计算可见性成本
     */
    float computeVisibilityCost(const openvdb::Coord& coord, 
                               const std::vector<openvdb::Coord>& seeds,
                               const PointCloudT::Ptr& cloud);
    
    /**
     * 执行最大流算法
     */
    bool solveMaxFlow();
    
    /**
     * 从图割结果构建标签网格
     */
    void buildLabelGrid(LabelGridT::Ptr& label_grid);
    
    /**
     * 获取体素的邻居坐标
     */
    std::vector<openvdb::Coord> getNeighbors(const openvdb::Coord& coord);
    
    /**
     * 检查坐标是否有效
     */
    bool isValidCoord(const openvdb::Coord& coord);
    
    /**
     * 计算两个坐标之间的距离
     */
    float computeDistance(const openvdb::Coord& coord1, const openvdb::Coord& coord2);

    /**
     * 自由空间泛洪
     */
    void floodFillFreeRegion(const std::vector<openvdb::Coord>& seeds);
};

} // namespace recon

#endif // GRAPH_CUT_H

