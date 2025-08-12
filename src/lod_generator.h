/**
 * LOD生成模块头文件
 * 负责生成多级细节(Level of Detail)网格
 * 
 * 版本: 1.0
 * 日期: 2025-08-12
 */

#ifndef LOD_GENERATOR_H
#define LOD_GENERATOR_H

#include <memory>
#include <vector>
#include <string>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/PolygonMesh.h>

namespace recon {

/**
 * LOD级别定义
 */
enum class LODLevel {
    FULL = 0,    // 完整模型
    LOD1 = 1,    // 第一级简化
    LOD2 = 2,    // 第二级简化
    LOD3 = 3     // 第三级简化
};

/**
 * LOD生成配置参数
 */
struct LODGeneratorConfig {
    // 目标三角形数量
    struct TargetCounts {
        int full = 500000;      // 完整模型
        int lod1 = 200000;      // LOD1
        int lod2 = 50000;       // LOD2
        int lod3 = 10000;       // LOD3
    } target_counts;
    
    // 区域感知简化
    struct RegionAware {
        bool enable = true;                 // 启用区域感知
        float planar_reduction = 0.8f;     // 平面区域简化比例
        float detail_reduction = 0.95f;    // 细节区域简化比例
        float edge_reduction = 0.9f;       // 边缘区域简化比例
    } region_aware;
    
    // 简化方法
    struct SimplificationConfig {
        std::string method = "quadric_edge_collapse";  // 简化方法
        bool preserve_boundaries = true;               // 保持边界
        bool preserve_sharp_edges = true;              // 保持尖锐边缘
        float edge_angle_threshold = 45.0f;            // 边缘角度阈值(度)
        bool preserve_topology = true;                 // 保持拓扑
        float aspect_ratio_threshold = 10.0f;          // 长宽比阈值
    } simplification;
    
    // 特征保持
    struct FeaturePreservation {
        bool preserve_corners = true;          // 保持角点
        bool preserve_ridges = true;           // 保持脊线
        bool preserve_valleys = true;          // 保持谷线
        float feature_angle = 30.0f;           // 特征角度(度)
        float corner_angle = 60.0f;            // 角点角度(度)
    } feature_preservation;
    
    // 质量控制
    struct QualityControl {
        float min_triangle_area = 1e-6f;       // 最小三角形面积
        float max_edge_length = 0.2f;          // 最大边长
        bool check_manifold = true;            // 检查流形
        bool check_orientation = true;         // 检查方向
        bool remove_isolated_vertices = true;  // 移除孤立顶点
    } quality_control;
    
    // 导出设置
    struct ExportConfig {
        std::vector<std::string> formats = {"obj", "ply"};  // 导出格式
        bool include_normals = true;                         // 包含法向量
        bool include_colors = true;                          // 包含颜色
        bool include_textures = false;                       // 包含纹理
        std::string naming_pattern = "{base}_lod{level}";    // 命名模式
    } export_config;
};

/**
 * 区域类型枚举
 */
enum class RegionType {
    PLANAR,     // 平面区域
    DETAIL,     // 细节区域
    EDGE,       // 边缘区域
    CORNER      // 角点区域
};

/**
 * 简化统计信息
 */
struct SimplificationStats {
    int original_vertices = 0;
    int original_triangles = 0;
    int simplified_vertices = 0;
    int simplified_triangles = 0;
    float reduction_ratio = 0.0f;
    double processing_time = 0.0;
    int preserved_features = 0;
    int removed_features = 0;
};

/**
 * LOD生成器类
 */
class LODGenerator {
public:
    using PointT = pcl::PointXYZRGBNormal;
    using PointCloudT = pcl::PointCloud<PointT>;

    /**
     * 构造函数
     */
    explicit LODGenerator(const LODGeneratorConfig& config = LODGeneratorConfig{});
    
    /**
     * 析构函数
     */
    ~LODGenerator();

    /**
     * 从完整网格生成所有LOD级别
     * @param full_mesh 完整网格
     * @param lod_meshes 输出LOD网格数组
     * @return 成功返回true
     */
    bool generateAllLODs(const pcl::PolygonMesh& full_mesh,
                        std::vector<pcl::PolygonMesh>& lod_meshes);

    /**
     * 生成指定级别的LOD
     * @param input_mesh 输入网格
     * @param level LOD级别
     * @param output_mesh 输出网格
     * @return 成功返回true
     */
    bool generateLOD(const pcl::PolygonMesh& input_mesh,
                    LODLevel level,
                    pcl::PolygonMesh& output_mesh);

    /**
     * 导出LOD网格到文件
     * @param lod_meshes LOD网格数组
     * @param base_filename 基础文件名
     * @param output_dir 输出目录
     * @return 成功返回true
     */
    bool exportLODs(const std::vector<pcl::PolygonMesh>& lod_meshes,
                   const std::string& base_filename,
                   const std::string& output_dir);

    /**
     * 设置配置参数
     */
    void setConfig(const LODGeneratorConfig& config) { config_ = config; }
    
    /**
     * 获取配置参数
     */
    const LODGeneratorConfig& getConfig() const { return config_; }

    /**
     * 获取简化统计信息
     */
    const std::vector<SimplificationStats>& getStatistics() const { return stats_; }

private:
    LODGeneratorConfig config_;
    std::vector<SimplificationStats> stats_;
    
    /**
     * 分析网格区域类型
     */
    std::vector<RegionType> analyzeRegions(const pcl::PolygonMesh& mesh);
    
    /**
     * 区域感知简化
     */
    bool regionAwareSimplification(const pcl::PolygonMesh& input_mesh,
                                  int target_triangles,
                                  pcl::PolygonMesh& output_mesh);
    
    /**
     * 二次边坍塌简化
     */
    bool quadricEdgeCollapse(const pcl::PolygonMesh& input_mesh,
                            int target_triangles,
                            pcl::PolygonMesh& output_mesh);
    
    /**
     * 顶点聚类简化
     */
    bool vertexClustering(const pcl::PolygonMesh& input_mesh,
                         int target_triangles,
                         pcl::PolygonMesh& output_mesh);
    
    /**
     * 检测特征边缘
     */
    std::vector<bool> detectFeatureEdges(const pcl::PolygonMesh& mesh);
    
    /**
     * 检测角点
     */
    std::vector<bool> detectCorners(const pcl::PolygonMesh& mesh);
    
    /**
     * 计算边缘重要性
     */
    std::vector<float> computeEdgeImportance(const pcl::PolygonMesh& mesh);
    
    /**
     * 计算顶点重要性
     */
    std::vector<float> computeVertexImportance(const pcl::PolygonMesh& mesh);
    
    /**
     * 质量控制检查
     */
    bool performQualityControl(pcl::PolygonMesh& mesh);
    
    /**
     * 移除退化三角形
     */
    bool removeDegenerateTriangles(pcl::PolygonMesh& mesh);
    
    /**
     * 移除孤立顶点
     */
    bool removeIsolatedVertices(pcl::PolygonMesh& mesh);
    
    /**
     * 计算网格统计信息
     */
    SimplificationStats computeStats(const pcl::PolygonMesh& original,
                                   const pcl::PolygonMesh& simplified,
                                   double processing_time);
    
    /**
     * 获取目标三角形数量
     */
    int getTargetTriangleCount(LODLevel level);
    
    /**
     * 生成输出文件名
     */
    std::string generateFilename(const std::string& base_filename,
                               LODLevel level,
                               const std::string& format);
    
    /**
     * 计算三角形面积
     */
    float computeTriangleArea(const pcl::PolygonMesh& mesh, int triangle_index);
    
    /**
     * 计算边长
     */
    float computeEdgeLength(const pcl::PolygonMesh& mesh, int vertex1, int vertex2);
    
    /**
     * 计算法向量角度
     */
    float computeNormalAngle(const pcl::PolygonMesh& mesh, int triangle1, int triangle2);
};

/**
 * 网格质量评估器
 */
class MeshQualityAssessor {
public:
    /**
     * 质量指标结构
     */
    struct QualityMetrics {
        // 几何质量
        float min_triangle_area = 0.0f;
        float max_triangle_area = 0.0f;
        float avg_triangle_area = 0.0f;
        float triangle_area_variance = 0.0f;
        
        float min_edge_length = 0.0f;
        float max_edge_length = 0.0f;
        float avg_edge_length = 0.0f;
        float edge_length_variance = 0.0f;
        
        float min_angle = 0.0f;
        float max_angle = 0.0f;
        float avg_angle = 0.0f;
        
        // 拓扑质量
        bool is_manifold = false;
        bool is_watertight = false;
        bool has_self_intersections = false;
        int boundary_edges = 0;
        int non_manifold_edges = 0;
        
        // 视觉质量
        float surface_area = 0.0f;
        float volume = 0.0f;
        int genus = 0;
        
        // 简化质量
        float hausdorff_distance = 0.0f;
        float rms_distance = 0.0f;
        float feature_preservation_ratio = 0.0f;
    };

    /**
     * 评估网格质量
     */
    static QualityMetrics assessQuality(const pcl::PolygonMesh& mesh);
    
    /**
     * 比较两个网格的质量
     */
    static QualityMetrics compareQuality(const pcl::PolygonMesh& original,
                                       const pcl::PolygonMesh& simplified);
    
    /**
     * 生成质量报告
     */
    static std::string generateQualityReport(const QualityMetrics& metrics);

private:
    /**
     * 计算Hausdorff距离
     */
    static float computeHausdorffDistance(const pcl::PolygonMesh& mesh1,
                                        const pcl::PolygonMesh& mesh2);
    
    /**
     * 计算RMS距离
     */
    static float computeRMSDistance(const pcl::PolygonMesh& mesh1,
                                  const pcl::PolygonMesh& mesh2);
    
    /**
     * 检查流形性
     */
    static bool checkManifold(const pcl::PolygonMesh& mesh);
    
    /**
     * 检查水密性
     */
    static bool checkWatertight(const pcl::PolygonMesh& mesh);
    
    /**
     * 检查自相交
     */
    static bool checkSelfIntersections(const pcl::PolygonMesh& mesh);
};

} // namespace recon

#endif // LOD_GENERATOR_H

