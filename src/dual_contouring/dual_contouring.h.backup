/**
 * 双重轮廓模块头文件
 * 负责从标签网格中提取表面网格
 * 
 * 版本: 1.0
 * 日期: 2025-08-12
 */

#ifndef DUAL_CONTOURING_H
#define DUAL_CONTOURING_H

#include <memory>
#include <vector>
#include <unordered_map>
#include <openvdb/openvdb.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/PolygonMesh.h>
#include <Eigen/Dense>

namespace recon {

/**
 * 双重轮廓配置参数
 */
struct DualContouringConfig {
    // QEF求解参数
    float qef_regularization = 0.1f;        // QEF正则化参数
    float anisotropic_normal_weight = 2.5f; // 法向量方向权重
    float anisotropic_tangent_weight = 1.0f; // 切向权重
    
    // 表面提取参数
    bool use_adaptive_normals = true;       // 使用自适应法向量
    bool gradient_fallback = true;          // 梯度后备方案
    float normal_estimation_radius = 0.03f; // 法向量估计半径
    
    // 网格质量参数
    float min_edge_length = 0.001f;         // 最小边长
    float max_edge_length = 0.1f;           // 最大边长
    bool remove_degenerate = true;          // 移除退化三角形
    
    // 特征保持参数
    bool preserve_sharp_edges = true;       // 保持尖锐边缘
    float sharp_edge_threshold = 45.0f;     // 尖锐边缘阈值(度)
    
    // 内存优化参数
    bool use_sparse_storage = true;         // 使用稀疏存储
    int max_vertices_per_cell = 8;          // 每个单元最大顶点数
};

/**
 * 边交点信息
 */
struct EdgeIntersection {
    openvdb::Vec3f position;    // 交点位置
    openvdb::Vec3f normal;      // 交点法向量
    float confidence;           // 置信度
    bool is_valid;              // 是否有效
};

/**
 * QEF(二次误差函数)数据
 */
struct QEFData {
    Eigen::Matrix3f A;          // 系数矩阵
    Eigen::Vector3f b;          // 右端向量
    float c;                    // 常数项
    int num_constraints;        // 约束数量
    
    QEFData() : A(Eigen::Matrix3f::Zero()), 
                b(Eigen::Vector3f::Zero()), 
                c(0.0f), 
                num_constraints(0) {}
    
    void addConstraint(const Eigen::Vector3f& point, 
                      const Eigen::Vector3f& normal,
                      float weight = 1.0f);
    
    Eigen::Vector3f solve(float regularization = 0.1f) const;
};

/**
 * 双重轮廓单元
 */
struct DualCell {
    openvdb::Coord coord;                           // 单元坐标
    std::vector<EdgeIntersection> intersections;    // 边交点
    QEFData qef;                                    // QEF数据
    openvdb::Vec3f vertex_position;                 // 顶点位置
    bool has_vertex;                                // 是否有顶点
    int vertex_index;                               // 顶点索引
};

/**
 * 双重轮廓提取器类
 */
class DualContouringExtractor {
public:
    using PointT = pcl::PointXYZRGBNormal;
    using PointCloudT = pcl::PointCloud<PointT>;
    using UDFGridT = openvdb::FloatGrid;
    using LabelGridT = openvdb::Int32Grid;

    /**
     * 构造函数
     */
    explicit DualContouringExtractor(const DualContouringConfig& config = DualContouringConfig{});
    
    /**
     * 析构函数
     */
    ~DualContouringExtractor();

    /**
     * 从标签网格提取表面网格
     * @param udf_grid UDF网格
     * @param label_grid 标签网格
     * @param cloud 原始点云
     * @param mesh 输出网格
     * @return 成功返回true
     */
    bool extractSurface(const UDFGridT::Ptr& udf_grid,
                       const LabelGridT::Ptr& label_grid,
                       const PointCloudT::Ptr& cloud,
                       pcl::PolygonMesh& mesh);

    /**
     * 设置配置参数
     */
    void setConfig(const DualContouringConfig& config) { config_ = config; }
    
    /**
     * 获取配置参数
     */
    const DualContouringConfig& getConfig() const { return config_; }

    /**
     * 获取提取统计信息
     */
    struct Statistics {
        int total_cells = 0;
        int active_cells = 0;
        int vertices_generated = 0;
        int triangles_generated = 0;
        int degenerate_removed = 0;
        double extraction_time = 0.0;
    };
    
    const Statistics& getStatistics() const { return stats_; }

private:
    DualContouringConfig config_;
    Statistics stats_;
    
    std::unordered_map<openvdb::Coord, DualCell> cells_;
    std::vector<openvdb::Vec3f> vertices_;
    std::vector<openvdb::Vec3f> normals_;
    std::vector<std::array<int, 3>> triangles_;
    
    /**
     * 查找所有切割边
     */
    void findCuttingEdges(const LabelGridT::Ptr& label_grid);
    
    /**
     * 计算边交点
     */
    EdgeIntersection computeEdgeIntersection(const openvdb::Coord& coord1,
                                           const openvdb::Coord& coord2,
                                           const UDFGridT::Ptr& udf_grid,
                                           const PointCloudT::Ptr& cloud);
    
    /**
     * 构建双重单元
     */
    void buildDualCells(const UDFGridT::Ptr& udf_grid,
                       const LabelGridT::Ptr& label_grid,
                       const PointCloudT::Ptr& cloud);
    
    /**
     * 求解QEF获得顶点位置
     */
    openvdb::Vec3f solveQEF(const DualCell& cell);
    
    /**
     * 生成三角形
     */
    void generateTriangles(const LabelGridT::Ptr& label_grid);
    
    /**
     * 估计交点法向量
     */
    openvdb::Vec3f estimateNormal(const openvdb::Vec3f& position,
                                 const UDFGridT::Ptr& udf_grid,
                                 const PointCloudT::Ptr& cloud);
    
    /**
     * 计算梯度法向量
     */
    openvdb::Vec3f computeGradientNormal(const openvdb::Vec3f& position,
                                        const UDFGridT::Ptr& udf_grid);
    
    /**
     * 从点云估计法向量
     */
    openvdb::Vec3f estimateNormalFromCloud(const openvdb::Vec3f& position,
                                          const PointCloudT::Ptr& cloud);
    
    /**
     * 检查边是否被切割
     */
    bool isEdgeCut(const openvdb::Coord& coord1,
                   const openvdb::Coord& coord2,
                   const LabelGridT::Ptr& label_grid);
    
    /**
     * 获取标签值
     */
    int getLabel(const openvdb::Coord& coord, const LabelGridT::Ptr& label_grid);
    
    /**
     * 获取UDF值
     */
    float getUDFValue(const openvdb::Coord& coord, const UDFGridT::Ptr& udf_grid);
    
    /**
     * 线性插值计算交点
     */
    float computeIntersectionParameter(float value1, float value2);
    
    /**
     * 构建PCL网格
     */
    void buildPCLMesh(pcl::PolygonMesh& mesh);
    
    /**
     * 清理退化三角形
     */
    void removeDegenerateTriangles();
    
    /**
     * 验证网格质量
     */
    bool validateMesh();
    
    /**
     * 获取单元的所有邻居
     */
    std::vector<openvdb::Coord> getCellNeighbors(const openvdb::Coord& coord);
    
    /**
     * 计算两点间距离
     */
    float computeDistance(const openvdb::Vec3f& p1, const openvdb::Vec3f& p2);
    
    /**
     * 检查三角形是否退化
     */
    bool isTriangleDegenerate(const std::array<int, 3>& triangle);
};

/**
 * 工具函数
 */
namespace dual_contouring_utils {
    
    /**
     * 计算三角形面积
     */
    float computeTriangleArea(const openvdb::Vec3f& v1,
                             const openvdb::Vec3f& v2,
                             const openvdb::Vec3f& v3);
    
    /**
     * 计算三角形法向量
     */
    openvdb::Vec3f computeTriangleNormal(const openvdb::Vec3f& v1,
                                        const openvdb::Vec3f& v2,
                                        const openvdb::Vec3f& v3);
    
    /**
     * 检查点是否在边界框内
     */
    bool isPointInBounds(const openvdb::Vec3f& point,
                        const openvdb::Vec3f& min_bound,
                        const openvdb::Vec3f& max_bound);
    
    /**
     * 计算向量夹角
     */
    float computeAngle(const openvdb::Vec3f& v1, const openvdb::Vec3f& v2);
    
} // namespace dual_contouring_utils

} // namespace recon

#endif // DUAL_CONTOURING_H

