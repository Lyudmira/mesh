/**
 * Alpha包装融合器
 * 实现基于CGAL Alpha包装的外壳与细节层融合
 * 
 * 版本: 1.0
 * 日期: 2025-08-12
 */

#ifndef ALPHA_WRAPPING_FUSION_H
#define ALPHA_WRAPPING_FUSION_H

#include <memory>
#include <vector>
#include <unordered_map>
#include <openvdb/openvdb.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/PolygonMesh.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <Eigen/Dense>

// CGAL包含
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/alpha_wrap_3.h>
#include <CGAL/Polygon_mesh_processing/detect_features.h>
#include <CGAL/Polygon_mesh_processing/compute_normal.h>

namespace recon {

// CGAL类型定义
using Kernel = CGAL::Exact_predicates_inexact_constructions_kernel;
using Point_3 = Kernel::Point_3;
using Vector_3 = Kernel::Vector_3;
using Mesh = CGAL::Surface_mesh<Point_3>;

/**
 * Alpha包装配置
 */
struct AlphaWrappingConfig {
    // Alpha包装参数
    double alpha = 0.02;                    // Alpha值(2cm)
    double offset = 0.005;                  // 偏移量(5mm)
    bool enable_refinement = true;          // 启用细化
    int max_refinement_steps = 3;           // 最大细化步数
    
    // 平面检测参数
    double plane_angle_threshold = 15.0;    // 平面角度阈值(度)
    double plane_distance_threshold = 0.01; // 平面距离阈值(1cm)
    int min_plane_points = 100;             // 最小平面点数
    
    // 交线重投影参数
    double intersection_tolerance = 0.001;   // 交线容差(1mm)
    bool enable_line_projection = true;     // 启用直线投影
    bool enforce_orthogonality = true;      // 强制正交性
    double orthogonal_tolerance = 5.0;      // 正交容差(度)
    
    // 质量控制参数
    double min_edge_length = 0.001;         // 最小边长(1mm)
    double max_edge_length = 0.1;           // 最大边长(10cm)
    double aspect_ratio_threshold = 10.0;   // 长宽比阈值
    
    // 性能参数
    bool use_parallel_processing = true;    // 并行处理
    int num_threads = 0;                    // 线程数
    bool enable_caching = true;             // 启用缓存
    
    // 调试参数
    bool enable_debug_output = false;       // 调试输出
    bool save_intermediate_meshes = false;  // 保存中间网格
    std::string debug_output_dir = "./debug_alpha"; // 调试输出目录
};

/**
 * 平面信息
 */
struct PlaneInfo {
    Eigen::Vector4d coefficients;          // 平面方程系数
    std::vector<int> vertex_indices;       // 平面上的顶点索引
    Eigen::Vector3d normal;                // 平面法向量
    Eigen::Vector3d centroid;              // 平面中心点
    double area = 0.0;                     // 平面面积
    int plane_id = -1;                     // 平面ID
    
    // 边界信息
    std::vector<std::vector<int>> boundaries; // 边界环
    bool is_closed = false;                // 是否封闭
};

/**
 * 交线信息
 */
struct IntersectionLine {
    Eigen::Vector3d start_point;           // 起始点
    Eigen::Vector3d end_point;             // 结束点
    Eigen::Vector3d direction;             // 方向向量
    std::vector<int> vertex_indices;       // 交线上的顶点索引
    int plane1_id = -1;                    // 平面1 ID
    int plane2_id = -1;                    // 平面2 ID
    double length = 0.0;                   // 交线长度
    bool is_straight = false;              // 是否为直线
};

/**
 * Alpha包装结果
 */
struct AlphaWrappingResult {
    Mesh wrapped_mesh;                     // 包装后的网格
    std::vector<PlaneInfo> detected_planes; // 检测到的平面
    std::vector<IntersectionLine> intersection_lines; // 交线
    
    // 统计信息
    int original_vertices = 0;
    int wrapped_vertices = 0;
    int original_faces = 0;
    int wrapped_faces = 0;
    int detected_plane_count = 0;
    int intersection_line_count = 0;
    
    double wrapping_time = 0.0;
    double plane_detection_time = 0.0;
    double line_projection_time = 0.0;
    double total_time = 0.0;
    
    bool success = false;
    std::string error_message;
};

/**
 * Alpha包装融合器
 */
class AlphaWrappingFusion {
public:
    using PointT = pcl::PointXYZRGBNormal;
    using PointCloudT = pcl::PointCloud<PointT>;

    /**
     * 构造函数
     */
    explicit AlphaWrappingFusion(const AlphaWrappingConfig& config = AlphaWrappingConfig{});
    
    /**
     * 析构函数
     */
    ~AlphaWrappingFusion();
    
    /**
     * 执行Alpha包装融合
     */
    bool performAlphaWrapping(
        const pcl::PolygonMesh& shell_mesh,
        const pcl::PolygonMesh& detail_mesh,
        pcl::PolygonMesh& fused_mesh,
        AlphaWrappingResult& result
    );
    
    /**
     * 检测平面
     */
    bool detectPlanes(
        const Mesh& mesh,
        std::vector<PlaneInfo>& planes
    );
    
    /**
     * 重投影交线
     */
    bool reprojectIntersectionLines(
        Mesh& mesh,
        const std::vector<PlaneInfo>& planes,
        std::vector<IntersectionLine>& lines
    );
    
    /**
     * 强制平面对齐
     */
    bool enforcePlaneAlignment(
        Mesh& mesh,
        std::vector<PlaneInfo>& planes
    );
    
    /**
     * 强制直线约束
     */
    bool enforceStraightLines(
        Mesh& mesh,
        std::vector<IntersectionLine>& lines
    );
    
    /**
     * 优化网格质量
     */
    bool optimizeMeshQuality(Mesh& mesh);
    
    /**
     * 验证几何约束
     */
    bool validateGeometricConstraints(
        const Mesh& mesh,
        const std::vector<PlaneInfo>& planes,
        const std::vector<IntersectionLine>& lines
    );
    
    /**
     * 获取配置
     */
    const AlphaWrappingConfig& getConfig() const { return config_; }
    
    /**
     * 设置配置
     */
    void setConfig(const AlphaWrappingConfig& config) { config_ = config; }
    
    /**
     * 获取最后结果
     */
    const AlphaWrappingResult& getLastResult() const { return last_result_; }

private:
    AlphaWrappingConfig config_;
    AlphaWrappingResult last_result_;
    
    /**
     * 初始化
     */
    bool initialize();
    
    /**
     * PCL网格转CGAL网格
     */
    bool convertPCLToCGAL(const pcl::PolygonMesh& pcl_mesh, Mesh& cgal_mesh);
    
    /**
     * CGAL网格转PCL网格
     */
    bool convertCGALToPCL(const Mesh& cgal_mesh, pcl::PolygonMesh& pcl_mesh);
    
    /**
     * 执行Alpha包装
     */
    bool performWrapping(const Mesh& input_mesh, Mesh& wrapped_mesh);
    
    /**
     * 计算平面方程
     */
    bool computePlaneEquation(
        const std::vector<Point_3>& points,
        Eigen::Vector4d& coefficients
    );
    
    /**
     * 查找平面边界
     */
    bool findPlaneBoundaries(
        const Mesh& mesh,
        const PlaneInfo& plane,
        std::vector<std::vector<int>>& boundaries
    );
    
    /**
     * 计算平面交线
     */
    bool computePlaneIntersection(
        const PlaneInfo& plane1,
        const PlaneInfo& plane2,
        IntersectionLine& line
    );
    
    /**
     * 投影点到平面
     */
    Point_3 projectPointToPlane(
        const Point_3& point,
        const Eigen::Vector4d& plane_coefficients
    );
    
    /**
     * 投影点到直线
     */
    Point_3 projectPointToLine(
        const Point_3& point,
        const IntersectionLine& line
    );
    
    /**
     * 检查点是否在平面上
     */
    bool isPointOnPlane(
        const Point_3& point,
        const Eigen::Vector4d& plane_coefficients,
        double tolerance = 0.001
    );
    
    /**
     * 检查直线是否直
     */
    bool isLineStraight(
        const std::vector<Point_3>& line_points,
        double tolerance = 0.001
    );
    
    /**
     * 优化顶点位置
     */
    bool optimizeVertexPositions(
        Mesh& mesh,
        const std::vector<PlaneInfo>& planes,
        const std::vector<IntersectionLine>& lines
    );
    
    /**
     * 平滑网格
     */
    bool smoothMesh(Mesh& mesh, int iterations = 3);
    
    /**
     * 移除退化面
     */
    bool removeDegenerateFaces(Mesh& mesh);
    
    /**
     * 修复网格拓扑
     */
    bool repairMeshTopology(Mesh& mesh);
    
    /**
     * 验证网格质量
     */
    bool validateMeshQuality(const Mesh& mesh);
    
    /**
     * 保存调试网格
     */
    void saveDebugMesh(const Mesh& mesh, const std::string& filename);
    
    /**
     * 更新统计信息
     */
    void updateStatistics(AlphaWrappingResult& result, const Mesh& original, const Mesh& wrapped);
    
    /**
     * 清理资源
     */
    void cleanup();
};

/**
 * Alpha包装工厂
 */
class AlphaWrappingFactory {
public:
    /**
     * 创建标准包装器
     */
    static std::unique_ptr<AlphaWrappingFusion> createStandardWrapper();
    
    /**
     * 创建高精度包装器
     */
    static std::unique_ptr<AlphaWrappingFusion> createHighPrecisionWrapper();
    
    /**
     * 创建快速包装器
     */
    static std::unique_ptr<AlphaWrappingFusion> createFastWrapper();
    
    /**
     * 创建建筑专用包装器
     */
    static std::unique_ptr<AlphaWrappingFusion> createArchitecturalWrapper();
};

} // namespace recon

#endif // ALPHA_WRAPPING_FUSION_H

