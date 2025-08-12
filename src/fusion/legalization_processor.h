/**
 * 合法化处理器
 * 确保重建结果符合建筑几何约束和工程标准
 * 
 * 版本: 1.0
 * 日期: 2025-08-12
 */

#ifndef LEGALIZATION_PROCESSOR_H
#define LEGALIZATION_PROCESSOR_H

#include "alpha_wrapping_fusion.h"
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
 * 合法化配置
 */
struct LegalizationConfig {
    // 几何约束参数
    double wall_thickness_min = 0.1;       // 最小墙厚(10cm)
    double wall_thickness_max = 0.5;       // 最大墙厚(50cm)
    double floor_height_min = 2.2;         // 最小层高(2.2m)
    double floor_height_max = 4.0;         // 最大层高(4.0m)
    double door_width_min = 0.6;           // 最小门宽(60cm)
    double door_width_max = 1.2;           // 最大门宽(120cm)
    double door_height_min = 1.8;          // 最小门高(180cm)
    double door_height_max = 2.2;          // 最大门高(220cm)
    double window_width_min = 0.4;         // 最小窗宽(40cm)
    double window_width_max = 3.0;         // 最大窗宽(300cm)
    double window_height_min = 0.6;        // 最小窗高(60cm)
    double window_height_max = 2.0;        // 最大窗高(200cm)
    
    // 正交性约束
    double orthogonal_tolerance = 2.0;     // 正交容差(度)
    double parallel_tolerance = 2.0;       // 平行容差(度)
    double perpendicular_tolerance = 2.0;  // 垂直容差(度)
    bool enforce_manhattan_world = true;   // 强制曼哈顿世界约束
    
    // 平面约束
    double plane_flatness_tolerance = 0.01; // 平面平整度容差(1cm)
    double plane_coplanarity_tolerance = 0.005; // 共面容差(5mm)
    bool merge_coplanar_faces = true;      // 合并共面面片
    
    // 直线约束
    double line_straightness_tolerance = 0.005; // 直线度容差(5mm)
    bool enforce_straight_edges = true;    // 强制直边约束
    bool snap_to_grid = true;              // 对齐到网格
    double grid_size = 0.01;               // 网格大小(1cm)
    
    // 拓扑约束
    bool ensure_manifold = true;           // 确保流形
    bool fill_small_holes = true;          // 填充小孔洞
    double max_hole_area = 0.01;           // 最大孔洞面积(0.01m²)
    bool remove_small_components = true;   // 移除小连通分量
    double min_component_volume = 0.001;   // 最小连通分量体积(0.001m³)
    
    // 语义约束
    bool validate_room_topology = true;    // 验证房间拓扑
    bool ensure_accessibility = true;      // 确保可达性
    bool validate_structural_integrity = true; // 验证结构完整性
    
    // 质量控制
    double min_face_area = 1e-6;           // 最小面积
    double max_aspect_ratio = 100.0;       // 最大长宽比
    double min_dihedral_angle = 5.0;       // 最小二面角(度)
    double max_dihedral_angle = 175.0;     // 最大二面角(度)
    
    // 优化参数
    int max_iterations = 10;               // 最大迭代次数
    double convergence_threshold = 1e-6;   // 收敛阈值
    bool use_iterative_refinement = true;  // 使用迭代细化
    
    // 性能参数
    bool use_parallel_processing = true;   // 并行处理
    int num_threads = 0;                   // 线程数
    bool enable_caching = true;            // 启用缓存
    
    // 调试参数
    bool enable_debug_output = false;      // 调试输出
    bool save_intermediate_results = false; // 保存中间结果
    std::string debug_output_dir = "./debug_legalization"; // 调试输出目录
};

/**
 * 约束违反信息
 */
struct ConstraintViolation {
    enum Type {
        GEOMETRIC,      // 几何约束违反
        TOPOLOGICAL,    // 拓扑约束违反
        SEMANTIC,       // 语义约束违反
        QUALITY         // 质量约束违反
    };
    
    Type type;
    std::string description;
    std::vector<int> affected_vertices;
    std::vector<int> affected_faces;
    double severity = 0.0;              // 严重程度[0,1]
    bool is_fixable = true;             // 是否可修复
    std::string suggested_fix;          // 建议修复方案
};

/**
 * 合法化统计信息
 */
struct LegalizationStats {
    // 约束违反统计
    int total_violations = 0;
    int geometric_violations = 0;
    int topological_violations = 0;
    int semantic_violations = 0;
    int quality_violations = 0;
    int fixed_violations = 0;
    int unfixed_violations = 0;
    
    // 几何修正统计
    int straightened_lines = 0;
    int flattened_planes = 0;
    int orthogonalized_angles = 0;
    int snapped_vertices = 0;
    
    // 拓扑修正统计
    int filled_holes = 0;
    int removed_components = 0;
    int merged_faces = 0;
    int fixed_manifold_issues = 0;
    
    // 质量改进统计
    double initial_quality_score = 0.0;
    double final_quality_score = 0.0;
    double quality_improvement = 0.0;
    
    // 时间统计
    double total_time = 0.0;
    double constraint_checking_time = 0.0;
    double geometric_correction_time = 0.0;
    double topological_correction_time = 0.0;
    double quality_optimization_time = 0.0;
    
    // 迭代统计
    int iterations_performed = 0;
    bool converged = false;
};

/**
 * 合法化结果
 */
struct LegalizationResult {
    pcl::PolygonMesh legalized_mesh;       // 合法化后的网格
    std::vector<ConstraintViolation> violations; // 检测到的约束违反
    std::vector<ConstraintViolation> fixed_violations; // 已修复的违反
    std::vector<ConstraintViolation> remaining_violations; // 剩余违反
    
    LegalizationStats stats;               // 统计信息
    
    bool success = false;
    std::string error_message;
    std::vector<std::string> warnings;
};

/**
 * 合法化处理器
 */
class LegalizationProcessor {
public:
    using PointT = pcl::PointXYZRGBNormal;
    using PointCloudT = pcl::PointCloud<PointT>;

    /**
     * 构造函数
     */
    explicit LegalizationProcessor(const LegalizationConfig& config = LegalizationConfig{});
    
    /**
     * 析构函数
     */
    ~LegalizationProcessor();
    
    /**
     * 执行完整的合法化处理
     */
    bool performLegalization(
        const pcl::PolygonMesh& input_mesh,
        LegalizationResult& result
    );
    
    /**
     * 检测约束违反
     */
    bool detectConstraintViolations(
        const pcl::PolygonMesh& mesh,
        std::vector<ConstraintViolation>& violations
    );
    
    /**
     * 修复几何约束违反
     */
    bool fixGeometricViolations(
        pcl::PolygonMesh& mesh,
        std::vector<ConstraintViolation>& violations
    );
    
    /**
     * 修复拓扑约束违反
     */
    bool fixTopologicalViolations(
        pcl::PolygonMesh& mesh,
        std::vector<ConstraintViolation>& violations
    );
    
    /**
     * 修复语义约束违反
     */
    bool fixSemanticViolations(
        pcl::PolygonMesh& mesh,
        std::vector<ConstraintViolation>& violations
    );
    
    /**
     * 优化网格质量
     */
    bool optimizeMeshQuality(
        pcl::PolygonMesh& mesh,
        std::vector<ConstraintViolation>& violations
    );
    
    /**
     * 强制曼哈顿世界约束
     */
    bool enforceManhattanWorld(pcl::PolygonMesh& mesh);
    
    /**
     * 对齐到网格
     */
    bool snapToGrid(pcl::PolygonMesh& mesh);
    
    /**
     * 验证建筑约束
     */
    bool validateArchitecturalConstraints(
        const pcl::PolygonMesh& mesh,
        std::vector<ConstraintViolation>& violations
    );
    
    /**
     * 计算质量分数
     */
    double computeQualityScore(const pcl::PolygonMesh& mesh);
    
    /**
     * 获取配置
     */
    const LegalizationConfig& getConfig() const { return config_; }
    
    /**
     * 设置配置
     */
    void setConfig(const LegalizationConfig& config) { config_ = config; }
    
    /**
     * 获取最后结果
     */
    const LegalizationResult& getLastResult() const { return last_result_; }

private:
    LegalizationConfig config_;
    LegalizationResult last_result_;
    
    /**
     * 初始化
     */
    bool initialize();
    
    /**
     * 检测几何约束违反
     */
    bool detectGeometricViolations(
        const pcl::PolygonMesh& mesh,
        std::vector<ConstraintViolation>& violations
    );
    
    /**
     * 检测拓扑约束违反
     */
    bool detectTopologicalViolations(
        const pcl::PolygonMesh& mesh,
        std::vector<ConstraintViolation>& violations
    );
    
    /**
     * 检测语义约束违反
     */
    bool detectSemanticViolations(
        const pcl::PolygonMesh& mesh,
        std::vector<ConstraintViolation>& violations
    );
    
    /**
     * 检测质量问题
     */
    bool detectQualityIssues(
        const pcl::PolygonMesh& mesh,
        std::vector<ConstraintViolation>& violations
    );
    
    /**
     * 强制直线约束
     */
    bool enforceStraightLines(pcl::PolygonMesh& mesh);
    
    /**
     * 强制平面约束
     */
    bool enforcePlanarFaces(pcl::PolygonMesh& mesh);
    
    /**
     * 强制正交约束
     */
    bool enforceOrthogonality(pcl::PolygonMesh& mesh);
    
    /**
     * 合并共面面片
     */
    bool mergeCoplanarFaces(pcl::PolygonMesh& mesh);
    
    /**
     * 填充孔洞
     */
    bool fillHoles(pcl::PolygonMesh& mesh);
    
    /**
     * 移除小连通分量
     */
    bool removeSmallComponents(pcl::PolygonMesh& mesh);
    
    /**
     * 确保流形性质
     */
    bool ensureManifold(pcl::PolygonMesh& mesh);
    
    /**
     * 验证房间拓扑
     */
    bool validateRoomTopology(
        const pcl::PolygonMesh& mesh,
        std::vector<ConstraintViolation>& violations
    );
    
    /**
     * 验证结构完整性
     */
    bool validateStructuralIntegrity(
        const pcl::PolygonMesh& mesh,
        std::vector<ConstraintViolation>& violations
    );
    
    /**
     * 计算面积
     */
    double computeFaceArea(const pcl::PolygonMesh& mesh, int face_index);
    
    /**
     * 计算体积
     */
    double computeMeshVolume(const pcl::PolygonMesh& mesh);
    
    /**
     * 计算二面角
     */
    double computeDihedralAngle(
        const pcl::PolygonMesh& mesh,
        int face1_index,
        int face2_index
    );
    
    /**
     * 检查是否共面
     */
    bool areCoplanar(
        const pcl::PolygonMesh& mesh,
        int face1_index,
        int face2_index,
        double tolerance
    );
    
    /**
     * 检查是否正交
     */
    bool areOrthogonal(
        const Eigen::Vector3d& normal1,
        const Eigen::Vector3d& normal2,
        double tolerance
    );
    
    /**
     * 检查是否平行
     */
    bool areParallel(
        const Eigen::Vector3d& normal1,
        const Eigen::Vector3d& normal2,
        double tolerance
    );
    
    /**
     * 计算面法向量
     */
    Eigen::Vector3d computeFaceNormal(const pcl::PolygonMesh& mesh, int face_index);
    
    /**
     * 更新统计信息
     */
    void updateStatistics(LegalizationResult& result);
    
    /**
     * 保存调试信息
     */
    void saveDebugInfo(const LegalizationResult& result);
    
    /**
     * 验证结果
     */
    bool validateResult(const LegalizationResult& result);
    
    /**
     * 清理资源
     */
    void cleanup();
};

/**
 * 合法化处理器工厂
 */
class LegalizationProcessorFactory {
public:
    /**
     * 创建标准处理器
     */
    static std::unique_ptr<LegalizationProcessor> createStandardProcessor();
    
    /**
     * 创建严格处理器
     */
    static std::unique_ptr<LegalizationProcessor> createStrictProcessor();
    
    /**
     * 创建宽松处理器
     */
    static std::unique_ptr<LegalizationProcessor> createLenientProcessor();
    
    /**
     * 创建建筑专用处理器
     */
    static std::unique_ptr<LegalizationProcessor> createArchitecturalProcessor();
    
    /**
     * 创建工程专用处理器
     */
    static std::unique_ptr<LegalizationProcessor> createEngineeringProcessor();
};

} // namespace recon

#endif // LEGALIZATION_PROCESSOR_H

