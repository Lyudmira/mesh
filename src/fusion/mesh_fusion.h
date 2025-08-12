#pragma once

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/PolygonMesh.h>
#include <Eigen/Dense>
#include <vector>

struct MeshFusionConfig {
    bool shell_priority = true;               // 外壳优先策略
    double detail_threshold = 0.005;          // 细节距离阈值(米)
    double shell_dilation = 0.005;            // 壳主导保护的膨胀δ(米)
    bool use_cgal = true;                    // 是否使用CGAL后端
    double boolean_tolerance = 0.001;        // 布尔运算容差
    double weld_threshold = 0.002;           // 顶点焊接阈值
    bool recompute_normals = true;           // 是否重估法向量
    bool consistent_orientation = true;      // 法向量一致方向
    double shell_color_weight = 0.7;         // 颜色混合权重-外壳
    double detail_color_weight = 0.3;        // 颜色混合权重-细节
    double blend_distance = 0.01;            // 颜色过渡距离
    bool enable_alpha_fallback = true;       // 布尔失败时使用Alpha包装
    double alpha_offset = 0.002;             // Alpha包装偏移
};

class MeshFuser {
public:
    explicit MeshFuser(const MeshFusionConfig& cfg);

    using PointT = pcl::PointXYZRGB;
    using PointCloudT = pcl::PointCloud<PointT>;
    using PointNormalT = pcl::PointXYZRGBNormal;
    using PointNormalCloudT = pcl::PointCloud<PointNormalT>;

    /**
     * Fuse shell and detail meshes into a single mesh.
     * @param shell   外壳网格
     * @param detail  细节网格
     * @param samples 原始带颜色点云
     * @param out     输出融合后网格
     */
    bool fuse(const pcl::PolygonMesh& shell,
              const pcl::PolygonMesh& detail,
              const typename PointNormalCloudT::ConstPtr& samples,
              pcl::PolygonMesh& out);

private:
    MeshFusionConfig cfg_;

    bool pclMeshToEigen(const pcl::PolygonMesh& mesh,
                        Eigen::MatrixXd& V,
                        Eigen::MatrixXi& F) const;

    void eigenToPclMesh(const Eigen::MatrixXd& V,
                        const Eigen::MatrixXi& F,
                        const Eigen::MatrixXd& N,
                        const std::vector<Eigen::Vector3f>& colors,
                        pcl::PolygonMesh& mesh) const;

    void assignColors(const Eigen::MatrixXd& V,
                      const pcl::PolygonMesh& shell,
                      const pcl::PolygonMesh& detail,
                      const typename PointNormalCloudT::ConstPtr& samples,
                      std::vector<Eigen::Vector3f>& colors) const;
};

