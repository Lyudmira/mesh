/**
 * 增强版双重轮廓模块头文件
 * 负责从UDF网格中提取高质量表面网格
 * 
 * 版本: 2.0 - 增强版本
 * 日期: 2025-08-12
 * 
 * 主要改进:
 * - 各向异性QEF求解
 * - 正确的三角形生成逻辑
 * - 改进的法向量估计
 * - 特征保持机制
 * - 质量控制系统
 * - 并行化处理
 */

#ifndef ENHANCED_DUAL_CONTOURING_H
#define ENHANCED_DUAL_CONTOURING_H

#include <memory>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <array>
#include <mutex>
#include <openvdb/openvdb.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/PolygonMesh.h>
#include <Eigen/Dense>
#include <Eigen/SVD>

namespace recon {

/**
 * 增强版双重轮廓配置参数
 */
struct EnhancedDualContouringConfig {
    // QEF求解参数
    float qef_regularization = 0.001f;          // QEF正则化参数
    float anisotropic_normal_weight = 3.0f;     // 法向量方向权重
    float anisotropic_tangent_weight = 1.0f;    // 切向权重
    bool use_svd_solver = true;                 // 使用SVD求解器
    float svd_threshold = 1e-6f;                // SVD奇异值阈值
    
    // 法向量估计参数
    bool use_adaptive_normals = true;           // 使用自适应法向量
    bool use_multiscale_gradient = true;        // 使用多尺度梯度
    std::vector<float> gradient_scales = {0.5f, 1.0f, 2.0f}; // 梯度计算尺度
    float normal_estimation_radius = 0.03f;     // 法向量估计半径
    float gradient_confidence_threshold = 0.1f; // 梯度置信度阈值
    
    // 网格质量参数
    float min_edge_length = 0.001f;             // 最小边长
    float max_edge_length = 0.1f;               // 最大边长
    float min_triangle_area = 1e-8f;            // 最小三角形面积
    bool remove_degenerate = true;              // 移除退化三角形
    bool validate_topology = true;              // 验证拓扑
    
    // 特征保持参数
    bool preserve_sharp_edges = true;           // 保持尖锐边缘
    float sharp_edge_threshold = 30.0f;         // 尖锐边缘阈值(度)
    float feature_weight_multiplier = 2.0f;     // 特征权重倍数
    bool detect_corners = true;                 // 检测角点
    float corner_threshold = 60.0f;             // 角点阈值(度)
    
    // 性能优化参数
    bool use_parallel_processing = true;        // 使用并行处理
    int num_threads = 0;                        // 线程数(0=自动)
    bool use_spatial_hashing = true;            // 使用空间哈希
    int chunk_size = 64;                        // 分块大小
    
    // 调试参数
    bool enable_debug_output = false;           // 启用调试输出
    bool save_intermediate_results = false;     // 保存中间结果
};

/**
 * 增强版边交点信息
 */
struct EnhancedEdgeIntersection {
    openvdb::Vec3f position;        // 交点位置
    openvdb::Vec3f normal;          // 交点法向量
    float confidence;               // 置信度
    bool is_valid;                  // 是否有效
    bool is_feature;                // 是否为特征点
    float curvature;                // 局部曲率
    
    EnhancedEdgeIntersection() 
        : position(0.0f), normal(0.0f, 0.0f, 1.0f), 
          confidence(0.0f), is_valid(false), is_feature(false), curvature(0.0f) {}
};

/**
 * 增强版QEF数据结构
 */
struct EnhancedQEFData {
    Eigen::Matrix3f A;              // 系数矩阵
    Eigen::Vector3f b;              // 右端向量
    float c;                        // 常数项
    int num_constraints;            // 约束数量
    
    // 各向异性权重
    float normal_weight;            // 法向量权重
    float tangent_weight;           // 切向权重
    
    // 特征保持
    std::vector<Eigen::Vector3f> feature_points;    // 特征点
    std::vector<Eigen::Vector3f> feature_normals;   // 特征法向量
    std::vector<float> feature_weights;             // 特征权重
    
    // 质心和边界框
    Eigen::Vector3f centroid;       // 约束点质心
    Eigen::Vector3f min_bound;      // 最小边界
    Eigen::Vector3f max_bound;      // 最大边界
    
    EnhancedQEFData() 
        : A(Eigen::Matrix3f::Zero()), b(Eigen::Vector3f::Zero()), 
          c(0.0f), num_constraints(0), normal_weight(1.0f), tangent_weight(1.0f),
          centroid(Eigen::Vector3f::Zero()),
          min_bound(Eigen::Vector3f::Constant(1e6f)),
          max_bound(Eigen::Vector3f::Constant(-1e6f)) {}
    
    /**
     * 添加约束
     */
    void addConstraint(const Eigen::Vector3f& point,
                      const Eigen::Vector3f& normal,
                      float weight = 1.0f,
                      bool is_feature = false);
    
    /**
     * 设置各向异性权重
     */
    void setAnisotropicWeights(float normal_w, float tangent_w);
    
    /**
     * 标准最小二乘求解
     */
    Eigen::Vector3f solve(float regularization = 0.001f) const;
    
    /**
     * SVD-based稳定求解
     */
    Eigen::Vector3f solveSVD(float regularization = 0.001f, float svd_threshold = 1e-6f) const;
    
    /**
     * 各向异性求解
     */
    Eigen::Vector3f solveAnisotropic(float normal_weight, float tangent_weight, 
                                    float regularization = 0.001f) const;
    
    /**
     * 带特征保持的求解
     */
    Eigen::Vector3f solveWithFeatures(float regularization = 0.001f) const;
    
    /**
     * 计算QEF误差
     */
    float computeError(const Eigen::Vector3f& point) const;
    
    /**
     * 获取约束质心
     */
    Eigen::Vector3f getCentroid() const { return centroid; }
    
    /**
     * 检查解是否在合理范围内
     */
    bool isValidSolution(const Eigen::Vector3f& solution) const;
};

/**
 * 边信息结构
 */
struct EdgeInfo {
    openvdb::Coord coord1, coord2;  // 边的两个端点
    int edge_index;                 // 边索引(0-11)
    bool is_cut;                    // 是否被切割
    EnhancedEdgeIntersection intersection; // 交点信息
};

/**
 * 增强版双重单元
 */
struct EnhancedDualCell {
    openvdb::Coord coord;                           // 单元坐标
    std::vector<EnhancedEdgeIntersection> intersections; // 边交点
    EnhancedQEFData qef;                            // QEF数据
    openvdb::Vec3f vertex_position;                 // 顶点位置
    openvdb::Vec3f vertex_normal;                   // 顶点法向量
    bool has_vertex;                                // 是否有顶点
    int vertex_index;                               // 顶点索引
    
    // 特征信息
    bool is_feature_cell;                           // 是否为特征单元
    std::vector<int> feature_edges;                 // 特征边索引
    float feature_strength;                         // 特征强度
    
    // 质量信息
    float qef_error;                                // QEF求解误差
    float vertex_confidence;                        // 顶点置信度
    
    EnhancedDualCell() 
        : coord(0, 0, 0), vertex_position(0.0f), vertex_normal(0.0f, 0.0f, 1.0f),
          has_vertex(false), vertex_index(-1), is_feature_cell(false), 
          feature_strength(0.0f), qef_error(0.0f), vertex_confidence(0.0f) {}
};

/**
 * 三角形质量信息
 */
struct TriangleQuality {
    float area;                     // 面积
    float aspect_ratio;             // 长宽比
    float min_angle;                // 最小角度
    float max_angle;                // 最大角度
    bool is_degenerate;             // 是否退化
    
    TriangleQuality() 
        : area(0.0f), aspect_ratio(0.0f), min_angle(0.0f), 
          max_angle(0.0f), is_degenerate(true) {}
};

/**
 * 增强版双重轮廓提取器类
 */
class EnhancedDualContouringExtractor {
public:
    using PointT = pcl::PointXYZRGBNormal;
    using PointCloudT = pcl::PointCloud<PointT>;
    using UDFGridT = openvdb::FloatGrid;
    using LabelGridT = openvdb::Int32Grid;

    /**
     * 构造函数
     */
    explicit EnhancedDualContouringExtractor(const EnhancedDualContouringConfig& config = EnhancedDualContouringConfig{});
    
    /**
     * 析构函数
     */
    ~EnhancedDualContouringExtractor();

    /**
     * 从UDF网格提取表面网格
     * @param udf_grid UDF网格
     * @param cloud 原始点云
     * @param mesh 输出网格
     * @return 成功返回true
     */
    bool extractSurface(const UDFGridT::Ptr& udf_grid,
                       const PointCloudT::Ptr& cloud,
                       pcl::PolygonMesh& mesh);

    /**
     * 设置配置参数
     */
    void setConfig(const EnhancedDualContouringConfig& config) { config_ = config; }
    
    /**
     * 获取配置参数
     */
    const EnhancedDualContouringConfig& getConfig() const { return config_; }

    /**
     * 获取提取统计信息
     */
    struct Statistics {
        int total_cells = 0;
        int active_cells = 0;
        int feature_cells = 0;
        int vertices_generated = 0;
        int triangles_generated = 0;
        int degenerate_removed = 0;
        int topology_errors = 0;
        
        double total_time = 0.0;
        double cell_building_time = 0.0;
        double qef_solving_time = 0.0;
        double triangulation_time = 0.0;
        double quality_control_time = 0.0;
        
        float average_qef_error = 0.0f;
        float average_vertex_confidence = 0.0f;
        float mesh_quality_score = 0.0f;
    };
    
    const Statistics& getStatistics() const { return stats_; }

private:
    EnhancedDualContouringConfig config_;
    Statistics stats_;
    
    // 数据存储
    std::unordered_map<openvdb::Coord, EnhancedDualCell> cells_;
    std::vector<openvdb::Vec3f> vertices_;
    std::vector<openvdb::Vec3f> normals_;
    std::vector<std::array<int, 3>> triangles_;
    std::vector<TriangleQuality> triangle_qualities_;
    
    // 边信息
    std::vector<EdgeInfo> cutting_edges_;
    std::unordered_map<std::pair<openvdb::Coord, openvdb::Coord>, int> edge_map_;
    
    // 线程安全
    mutable std::mutex cells_mutex_;
    mutable std::mutex vertices_mutex_;
    mutable std::mutex triangles_mutex_;
    
    /**
     * 查找所有切割边
     */
    void findCuttingEdges(const UDFGridT::Ptr& udf_grid);
    
    /**
     * 并行构建双重单元
     */
    void buildDualCellsParallel(const UDFGridT::Ptr& udf_grid,
                               const PointCloudT::Ptr& cloud);
    
    /**
     * 处理单个双重单元
     */
    EnhancedDualCell processSingleCell(const openvdb::Coord& coord,
                                      const UDFGridT::Ptr& udf_grid,
                                      const PointCloudT::Ptr& cloud);
    
    /**
     * 计算增强版边交点
     */
    EnhancedEdgeIntersection computeEnhancedEdgeIntersection(
        const openvdb::Coord& coord1,
        const openvdb::Coord& coord2,
        int edge_index,
        const UDFGridT::Ptr& udf_grid,
        const PointCloudT::Ptr& cloud);
    
    /**
     * 求解增强版QEF
     */
    openvdb::Vec3f solveEnhancedQEF(const EnhancedDualCell& cell);
    
    /**
     * 生成正确的三角形
     */
    void generateCorrectTriangles(const UDFGridT::Ptr& udf_grid);
    
    /**
     * 获取边相邻的单元
     */
    std::vector<openvdb::Coord> getAdjacentCells(const openvdb::Coord& c1,
                                                const openvdb::Coord& c2,
                                                int edge_index);
    
    /**
     * 为边生成三角形
     */
    void generateTrianglesForEdge(const std::vector<openvdb::Coord>& adjacent_cells,
                                 int edge_index);
    
    /**
     * 估计增强版法向量
     */
    openvdb::Vec3f estimateEnhancedNormal(const openvdb::Vec3f& position,
                                         const UDFGridT::Ptr& udf_grid,
                                         const PointCloudT::Ptr& cloud);
    
    /**
     * 多尺度梯度计算
     */
    openvdb::Vec3f computeMultiScaleGradient(const openvdb::Vec3f& position,
                                            const UDFGridT::Ptr& udf_grid);
    
    /**
     * 计算指定尺度的梯度
     */
    openvdb::Vec3f computeGradientAtScale(const openvdb::Vec3f& position,
                                         const UDFGridT::Ptr& udf_grid,
                                         float scale);
    
    /**
     * 计算梯度置信度
     */
    float computeGradientConfidence(const openvdb::Vec3f& gradient, float scale);
    
    /**
     * 智能法向量融合
     */
    openvdb::Vec3f fuseNormals(const openvdb::Vec3f& gradient_normal,
                              const openvdb::Vec3f& cloud_normal,
                              const openvdb::Vec3f& position,
                              const PointCloudT::Ptr& cloud);
    
    /**
     * 从点云估计法向量
     */
    openvdb::Vec3f estimateNormalFromCloud(const openvdb::Vec3f& position,
                                          const PointCloudT::Ptr& cloud);
    
    /**
     * 计算点云法向量置信度
     */
    float computeCloudNormalConfidence(const openvdb::Vec3f& cloud_normal,
                                      const openvdb::Vec3f& position,
                                      const PointCloudT::Ptr& cloud);
    
    /**
     * 检测尖锐特征
     */
    void detectSharpFeatures(EnhancedDualCell& cell,
                            const UDFGridT::Ptr& udf_grid,
                            const PointCloudT::Ptr& cloud);
    
    /**
     * 检查是否为尖锐边缘
     */
    bool isSharpEdge(const openvdb::Vec3f& p1, const openvdb::Vec3f& p2,
                    const openvdb::Vec3f& n1, const openvdb::Vec3f& n2);
    
    /**
     * 检查是否为角点
     */
    bool isCorner(const std::vector<openvdb::Vec3f>& normals);
    
    /**
     * 质量控制
     */
    void performQualityControl();
    
    /**
     * 移除退化三角形
     */
    void removeDegenerateTriangles();
    
    /**
     * 计算三角形质量
     */
    TriangleQuality computeTriangleQuality(const std::array<int, 3>& triangle);
    
    /**
     * 验证网格拓扑
     */
    bool validateMeshTopology();
    
    /**
     * 修复拓扑错误
     */
    void fixTopologyErrors();
    
    /**
     * 构建PCL网格
     */
    void buildPCLMesh(pcl::PolygonMesh& mesh);
    
    /**
     * 获取UDF值
     */
    float getUDFValue(const openvdb::Vec3f& position, const UDFGridT::Ptr& udf_grid);
    
    /**
     * 检查边是否被切割
     */
    bool isEdgeCut(const openvdb::Coord& coord1, const openvdb::Coord& coord2,
                   const UDFGridT::Ptr& udf_grid);
    
    /**
     * 线性插值计算交点
     */
    float computeIntersectionParameter(float value1, float value2);
    
    /**
     * 获取边的坐标
     */
    std::pair<openvdb::Coord, openvdb::Coord> getEdgeCoords(const openvdb::Coord& base_coord,
                                                           int edge_index);
    
    /**
     * 计算两点间距离
     */
    float computeDistance(const openvdb::Vec3f& p1, const openvdb::Vec3f& p2);
    
    /**
     * 更新统计信息
     */
    void updateStatistics();
    
    /**
     * 线程安全地添加顶点
     */
    int addVertexThreadSafe(const openvdb::Vec3f& vertex, const openvdb::Vec3f& normal);
    
    /**
     * 线程安全地添加三角形
     */
    void addTriangleThreadSafe(const std::array<int, 3>& triangle);
};

/**
 * 增强版工具函数
 */
namespace enhanced_dual_contouring_utils {
    
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
     * 计算三角形长宽比
     */
    float computeTriangleAspectRatio(const openvdb::Vec3f& v1,
                                    const openvdb::Vec3f& v2,
                                    const openvdb::Vec3f& v3);
    
    /**
     * 计算三角形角度
     */
    std::array<float, 3> computeTriangleAngles(const openvdb::Vec3f& v1,
                                              const openvdb::Vec3f& v2,
                                              const openvdb::Vec3f& v3);
    
    /**
     * 计算向量夹角
     */
    float computeAngle(const openvdb::Vec3f& v1, const openvdb::Vec3f& v2);
    
    /**
     * 检查点是否在边界框内
     */
    bool isPointInBounds(const openvdb::Vec3f& point,
                        const openvdb::Vec3f& min_bound,
                        const openvdb::Vec3f& max_bound);
    
    /**
     * 计算点到平面的距离
     */
    float pointToPlaneDistance(const openvdb::Vec3f& point,
                              const openvdb::Vec3f& plane_point,
                              const openvdb::Vec3f& plane_normal);
    
    /**
     * 计算两个向量的叉积
     */
    openvdb::Vec3f crossProduct(const openvdb::Vec3f& v1, const openvdb::Vec3f& v2);
    
    /**
     * 标准化向量
     */
    openvdb::Vec3f normalize(const openvdb::Vec3f& v);
    
    /**
     * 计算向量长度
     */
    float length(const openvdb::Vec3f& v);
    
} // namespace enhanced_dual_contouring_utils

} // namespace recon

#endif // ENHANCED_DUAL_CONTOURING_H

