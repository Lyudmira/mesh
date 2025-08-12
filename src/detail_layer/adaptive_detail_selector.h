/**
 * 自适应细节层选择器
 * 根据局部法向质量和密度自动选择重建算法
 * 
 * 版本: 2.0
 * 日期: 2025-08-12
 */

#ifndef ADAPTIVE_DETAIL_SELECTOR_H
#define ADAPTIVE_DETAIL_SELECTOR_H

#include <memory>
#include <vector>
#include <unordered_map>
#include <openvdb/openvdb.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/PolygonMesh.h>
#include <pcl/surface/gp3.h>
#include <pcl/surface/poisson.h>
#include <pcl/surface/impl/mls.hpp>
#include <Eigen/Dense>

namespace recon {

// 类型定义
using PointT = pcl::PointXYZRGBNormal;
using PointCloudT = pcl::PointCloud<PointT>;

/**
 * 重建算法类型
 */
enum class ReconstructionMethod {
    GP3,            // Greedy Projection Triangulation
    POISSON,        // Poisson Surface Reconstruction
    RIMLS,          // Robust Implicit Moving Least Squares
    HYBRID,         // 混合方法
    AUTO            // 自动选择
};

/**
 * 细节层选择配置
 */
struct DetailSelectorConfig {
    // 质量评估阈值
    float normal_quality_threshold = 0.7f;      // 法向量质量阈值
    float density_uniformity_threshold = 0.6f;  // 密度均匀性阈值
    float noise_level_threshold = 0.05f;        // 噪声水平阈值
    float completeness_threshold = 0.8f;        // 完整性阈值
    
    // 算法选择参数
    float gp3_density_threshold = 500.0f;       // GP3密度阈值(点/m³)
    float poisson_smoothness_threshold = 0.3f;  // Poisson平滑度阈值
    float rimls_noise_threshold = 0.1f;         // RIMLS噪声阈值
    
    // 混合方法参数
    bool enable_hybrid_reconstruction = true;   // 启用混合重建
    float hybrid_transition_width = 0.1f;       // 混合过渡宽度
    int min_cluster_size = 100;                 // 最小聚类大小
    float cluster_coherence_threshold = 0.8f;   // 聚类一致性阈值
    
    // 聚类参数
    float clustering_radius = 0.15f;            // 聚类半径
    int min_points_per_cluster = 50;            // 每个聚类最少点数
    float cluster_normal_threshold = 0.9f;      // 聚类法向量阈值
    float cluster_density_threshold = 0.5f;     // 聚类密度阈值
    
    // 偏移带参数
    float offset_band_width = 0.2f;             // 偏移带宽度
    bool use_shell_normal_guidance = true;      // 使用外壳法向指导
    float shell_normal_weight = 0.7f;           // 外壳法向权重
    
    // 性能参数
    bool use_parallel_processing = true;        // 并行处理
    int num_threads = 0;                        // 线程数(0=自动)
    bool use_spatial_indexing = true;           // 空间索引
    
    // 调试参数
    bool enable_debug_output = false;           // 调试输出
    bool save_cluster_results = false;          // 保存聚类结果
};

/**
 * 局部质量评估结果
 */
struct LocalQualityAssessment {
    // 法向量质量
    float normal_consistency = 0.0f;    // 法向量一致性[0,1]
    float normal_confidence = 0.0f;     // 法向量置信度[0,1]
    float normal_smoothness = 0.0f;     // 法向量平滑度[0,1]
    
    // 密度质量
    float point_density = 0.0f;         // 点密度(点/m³)
    float density_uniformity = 0.0f;    // 密度均匀性[0,1]
    float density_stability = 0.0f;     // 密度稳定性[0,1]
    
    // 几何质量
    float surface_smoothness = 0.0f;    // 表面平滑度[0,1]
    float feature_sharpness = 0.0f;     // 特征锐度[0,1]
    float noise_level = 0.0f;           // 噪声水平[0,1]
    
    // 完整性
    float coverage_completeness = 0.0f; // 覆盖完整性[0,1]
    float sampling_adequacy = 0.0f;     // 采样充分性[0,1]
    
    // 综合评分
    float overall_quality = 0.0f;       // 总体质量[0,1]
    
    /**
     * 计算综合质量评分
     */
    void computeOverallQuality();
};

/**
 * 重建算法推荐结果
 */
struct ReconstructionRecommendation {
    ReconstructionMethod primary_method;        // 主要方法
    ReconstructionMethod secondary_method;      // 次要方法(用于混合)
    float confidence;                           // 推荐置信度[0,1]
    float mixing_weight;                        // 混合权重[0,1]
    
    // 算法特定参数
    std::unordered_map<std::string, float> parameters;
    
    // 推荐理由
    std::string reasoning;
};

/**
 * 点云聚类信息
 */
struct PointCluster {
    int cluster_id;                             // 聚类ID
    std::vector<int> point_indices;             // 点索引
    ReconstructionMethod recommended_method;    // 推荐方法
    LocalQualityAssessment quality;             // 质量评估
    
    // 聚类几何信息
    Eigen::Vector3f centroid;                   // 质心
    Eigen::Vector3f dominant_normal;            // 主导法向量
    float avg_density;                          // 平均密度
    float coherence_score;                      // 一致性评分
    
    // 边界信息
    std::vector<int> boundary_points;           // 边界点
    bool is_boundary_cluster;                   // 是否为边界聚类
    bool needs_shell_guidance;                  // 是否需要外壳指导
};

/**
 * 自适应细节层选择器
 */
class AdaptiveDetailSelector {
public:
    using PointT = pcl::PointXYZRGBNormal;
    using PointCloudT = pcl::PointCloud<PointT>;
    using GridT = openvdb::FloatGrid;

    /**
     * 构造函数
     */
    explicit AdaptiveDetailSelector(const DetailSelectorConfig& config = DetailSelectorConfig{});
    
    /**
     * 析构函数
     */
    ~AdaptiveDetailSelector();

    /**
     * 分析点云并推荐重建方法
     */
    std::vector<ReconstructionRecommendation> analyzeAndRecommend(
        const PointCloudT::Ptr& cloud,
        const GridT::Ptr& udf_grid = nullptr);

    /**
     * 执行自适应聚类
     */
    std::vector<PointCluster> performAdaptiveClustering(const PointCloudT::Ptr& cloud);
    
    /**
     * 为每个聚类选择最优重建方法
     */
    void selectOptimalMethods(std::vector<PointCluster>& clusters);
    
    /**
     * 执行混合重建
     */
    bool performHybridReconstruction(const std::vector<PointCluster>& clusters,
                                    const PointCloudT::Ptr& cloud,
                                    pcl::PolygonMesh& output_mesh);

    /**
     * 设置配置参数
     */
    void setConfig(const DetailSelectorConfig& config) { config_ = config; }
    
    /**
     * 获取配置参数
     */
    const DetailSelectorConfig& getConfig() const { return config_; }
    
    /**
     * 获取统计信息
     */
    struct Statistics {
        int total_points = 0;
        int total_clusters = 0;
        int gp3_clusters = 0;
        int poisson_clusters = 0;
        int rimls_clusters = 0;
        int hybrid_clusters = 0;
        
        float avg_cluster_quality = 0.0f;
        float avg_normal_quality = 0.0f;
        float avg_density_uniformity = 0.0f;
        
        double analysis_time_seconds = 0.0;
        double clustering_time_seconds = 0.0;
        double reconstruction_time_seconds = 0.0;
    };
    
    const Statistics& getStatistics() const { return stats_; }

private:
    DetailSelectorConfig config_;
    Statistics stats_;
    
    // 质量评估器
    std::unique_ptr<class LocalQualityEvaluator> quality_evaluator_;
    
    // 聚类器
    std::unique_ptr<class AdaptiveClusterer> clusterer_;
    
    // 重建器
    std::unique_ptr<class HybridReconstructor> reconstructor_;
    
    /**
     * 初始化组件
     */
    void initializeComponents();
    
    /**
     * 评估局部质量
     */
    LocalQualityAssessment evaluateLocalQuality(const openvdb::Vec3f& position,
                                               const PointCloudT& cloud,
                                               float radius);
    
    /**
     * 基于质量选择方法
     */
    ReconstructionMethod selectMethodByQuality(const LocalQualityAssessment& quality);
    
    /**
     * 计算方法适用性评分
     */
    float computeMethodSuitability(ReconstructionMethod method,
                                  const LocalQualityAssessment& quality);
};

/**
 * 局部质量评估器
 */
class LocalQualityEvaluator {
public:
    using PointT = pcl::PointXYZRGBNormal;
    using PointCloudT = pcl::PointCloud<PointT>;

    /**
     * 构造函数
     */
    explicit LocalQualityEvaluator(const DetailSelectorConfig& config);
    
    /**
     * 评估局部质量
     */
    LocalQualityAssessment evaluate(const openvdb::Vec3f& center,
                                   const PointCloudT& cloud,
                                   float radius);
    
    /**
     * 批量质量评估
     */
    std::vector<LocalQualityAssessment> evaluateBatch(
        const std::vector<openvdb::Vec3f>& positions,
        const PointCloudT& cloud,
        float radius);

private:
    DetailSelectorConfig config_;
    std::unique_ptr<pcl::KdTreeFLANN<PointT>> kdtree_;
    
    /**
     * 评估法向量质量
     */
    void evaluateNormalQuality(const std::vector<int>& indices,
                              const PointCloudT& cloud,
                              LocalQualityAssessment& assessment);
    
    /**
     * 评估密度质量
     */
    void evaluateDensityQuality(const std::vector<int>& indices,
                               const std::vector<float>& distances,
                               const PointCloudT& cloud,
                               LocalQualityAssessment& assessment);
    
    /**
     * 评估几何质量
     */
    void evaluateGeometricQuality(const std::vector<int>& indices,
                                 const PointCloudT& cloud,
                                 LocalQualityAssessment& assessment);
    
    /**
     * 评估完整性
     */
    void evaluateCompleteness(const openvdb::Vec3f& center,
                             const std::vector<int>& indices,
                             const PointCloudT& cloud,
                             float radius,
                             LocalQualityAssessment& assessment);
};

/**
 * 自适应聚类器
 */
class AdaptiveClusterer {
public:
    using PointT = pcl::PointXYZRGBNormal;
    using PointCloudT = pcl::PointCloud<PointT>;

    /**
     * 构造函数
     */
    explicit AdaptiveClusterer(const DetailSelectorConfig& config);
    
    /**
     * 执行自适应聚类
     */
    std::vector<PointCluster> cluster(const PointCloudT::Ptr& cloud);
    
    /**
     * 基于质量的聚类
     */
    std::vector<PointCluster> clusterByQuality(const PointCloudT::Ptr& cloud,
                                              const std::vector<LocalQualityAssessment>& qualities);
    
    /**
     * 基于几何的聚类
     */
    std::vector<PointCluster> clusterByGeometry(const PointCloudT::Ptr& cloud);
    
    /**
     * 优化聚类结果
     */
    void optimizeClusters(std::vector<PointCluster>& clusters,
                         const PointCloudT& cloud);

private:
    DetailSelectorConfig config_;
    std::unique_ptr<pcl::KdTreeFLANN<PointT>> kdtree_;
    
    /**
     * 计算聚类一致性
     */
    float computeClusterCoherence(const PointCluster& cluster,
                                 const PointCloudT& cloud);
    
    /**
     * 合并相似聚类
     */
    void mergeSimilarClusters(std::vector<PointCluster>& clusters,
                             const PointCloudT& cloud);
    
    /**
     * 分割大聚类
     */
    void splitLargeClusters(std::vector<PointCluster>& clusters,
                           const PointCloudT& cloud);
    
    /**
     * 检测边界点
     */
    std::vector<int> detectBoundaryPoints(const PointCluster& cluster,
                                         const PointCloudT& cloud);
};

/**
 * 混合重建器
 */
class HybridReconstructor {
public:
    using PointT = pcl::PointXYZRGBNormal;
    using PointCloudT = pcl::PointCloud<PointT>;

    /**
     * 构造函数
     */
    explicit HybridReconstructor(const DetailSelectorConfig& config);
    
    /**
     * 执行混合重建
     */
    bool reconstruct(const std::vector<PointCluster>& clusters,
                    const PointCloudT::Ptr& cloud,
                    pcl::PolygonMesh& output_mesh);

private:
    DetailSelectorConfig config_;
    
    /**
     * GP3重建
     */
    bool reconstructGP3(const PointCluster& cluster,
                       const PointCloudT::Ptr& cloud,
                       pcl::PolygonMesh& mesh);
    
    /**
     * Poisson重建
     */
    bool reconstructPoisson(const PointCluster& cluster,
                           const PointCloudT::Ptr& cloud,
                           pcl::PolygonMesh& mesh);
    
    /**
     * RIMLS重建
     */
    bool reconstructRIMLS(const PointCluster& cluster,
                         const PointCloudT::Ptr& cloud,
                         pcl::PolygonMesh& mesh);
    
    /**
     * 融合多个网格
     */
    bool fuseMeshes(const std::vector<pcl::PolygonMesh>& meshes,
                   pcl::PolygonMesh& output_mesh);
    
    /**
     * 处理网格边界
     */
    void processMeshBoundaries(pcl::PolygonMesh& mesh,
                              const std::vector<PointCluster>& clusters);
};

/**
 * 偏移带处理器
 */
class OffsetBandProcessor {
public:
    using PointT = pcl::PointXYZRGBNormal;
    using PointCloudT = pcl::PointCloud<PointT>;

    /**
     * 偏移带信息
     */
    struct OffsetBandInfo {
        std::vector<int> inner_points;      // 内侧点
        std::vector<int> outer_points;      // 外侧点
        std::vector<int> boundary_points;   // 边界点
        float band_width;                   // 带宽
        Eigen::Vector3f shell_normal;       // 外壳法向量
    };
    
    /**
     * 构造函数
     */
    explicit OffsetBandProcessor(const DetailSelectorConfig& config);
    
    /**
     * 提取偏移带
     */
    OffsetBandInfo extractOffsetBand(const PointCloudT::Ptr& cloud,
                                    const openvdb::FloatGrid::Ptr& udf_grid);
    
    /**
     * 使用外壳法向确定点位置
     */
    void classifyPointsByShellNormal(OffsetBandInfo& band_info,
                                    const PointCloudT& cloud);
    
    /**
     * 避免误包含壳内点
     */
    std::vector<int> filterShellInteriorPoints(const std::vector<int>& candidate_points,
                                              const PointCloudT& cloud,
                                              const OffsetBandInfo& band_info);

private:
    DetailSelectorConfig config_;
    
    /**
     * 估计外壳法向量
     */
    Eigen::Vector3f estimateShellNormal(const PointCloudT& cloud,
                                       const std::vector<int>& boundary_points);
    
    /**
     * 判断点是否在内侧
     */
    bool isPointInside(const PointT& point,
                      const Eigen::Vector3f& shell_normal,
                      const Eigen::Vector3f& reference_point);
};

/**
 * 方法选择策略
 */
class MethodSelectionStrategy {
public:
    /**
     * 选择策略类型
     */
    enum class StrategyType {
        QUALITY_BASED,      // 基于质量
        DENSITY_BASED,      // 基于密度
        FEATURE_BASED,      // 基于特征
        HYBRID_ADAPTIVE     // 混合自适应
    };
    
    /**
     * 构造函数
     */
    explicit MethodSelectionStrategy(StrategyType type, const DetailSelectorConfig& config);
    
    /**
     * 选择重建方法
     */
    ReconstructionRecommendation selectMethod(const LocalQualityAssessment& quality,
                                             const PointCluster& cluster);
    
    /**
     * 批量方法选择
     */
    std::vector<ReconstructionRecommendation> selectMethodsBatch(
        const std::vector<LocalQualityAssessment>& qualities,
        const std::vector<PointCluster>& clusters);

private:
    StrategyType strategy_type_;
    DetailSelectorConfig config_;
    
    /**
     * 基于质量的选择
     */
    ReconstructionRecommendation selectByQuality(const LocalQualityAssessment& quality);
    
    /**
     * 基于密度的选择
     */
    ReconstructionRecommendation selectByDensity(const LocalQualityAssessment& quality);
    
    /**
     * 基于特征的选择
     */
    ReconstructionRecommendation selectByFeature(const LocalQualityAssessment& quality);
    
    /**
     * 混合自适应选择
     */
    ReconstructionRecommendation selectHybridAdaptive(const LocalQualityAssessment& quality,
                                                     const PointCluster& cluster);
};

/**
 * 重建质量验证器
 */
class ReconstructionQualityValidator {
public:
    /**
     * 验证结果
     */
    struct ValidationResult {
        bool is_valid = false;
        float quality_score = 0.0f;
        std::vector<std::string> issues;
        std::vector<std::string> recommendations;
    };
    
    /**
     * 验证重建结果
     */
    static ValidationResult validate(const pcl::PolygonMesh& mesh,
                                   const PointCloudT& original_cloud);
    
    /**
     * 验证聚类质量
     */
    static ValidationResult validateClustering(const std::vector<PointCluster>& clusters,
                                              const PointCloudT& cloud);

private:
    /**
     * 检查网格完整性
     */
    static bool checkMeshIntegrity(const pcl::PolygonMesh& mesh);
    
    /**
     * 检查聚类一致性
     */
    static bool checkClusterConsistency(const PointCluster& cluster,
                                       const PointCloudT& cloud);
};

/**
 * 细节层选择器工厂
 */
class DetailSelectorFactory {
public:
    enum class SelectorType {
        BASIC,          // 基础选择器
        ADAPTIVE,       // 自适应选择器
        QUALITY_AWARE,  // 质量感知选择器
        HYBRID_SMART    // 智能混合选择器
    };
    
    static std::unique_ptr<AdaptiveDetailSelector> create(
        SelectorType type,
        const DetailSelectorConfig& config = DetailSelectorConfig{});
};

} // namespace recon

#endif // ADAPTIVE_DETAIL_SELECTOR_H

