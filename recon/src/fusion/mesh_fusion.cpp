#include "mesh_fusion.h"

#include <igl/copyleft/cgal/mesh_boolean.h>
#include <igl/copyleft/cgal/approximate_offset.h>
#include <igl/copyleft/cgal/alpha_wrap.h>
#include <igl/per_vertex_normals.h>
#include <igl/remove_duplicate_vertices.h>
#include <igl/remove_unreferenced.h>

#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/impl/io.hpp>
#include <iostream>
#include <unordered_map>

MeshFuser::MeshFuser(const MeshFusionConfig& cfg) : cfg_(cfg) {}

bool MeshFuser::pclMeshToEigen(const pcl::PolygonMesh& mesh,
                               Eigen::MatrixXd& V,
                               Eigen::MatrixXi& F) const {
    PointCloudT cloud;
    pcl::fromPCLPointCloud2(mesh.cloud, cloud);
    V.resize(cloud.size(), 3);
    for (size_t i = 0; i < cloud.size(); ++i) {
        const auto& p = cloud.points[i];
        V(i, 0) = p.x;
        V(i, 1) = p.y;
        V(i, 2) = p.z;
    }
    F.resize(mesh.polygons.size(), 3);
    for (size_t i = 0; i < mesh.polygons.size(); ++i) {
        const auto& poly = mesh.polygons[i];
        if (poly.vertices.size() != 3) return false;
        F(i, 0) = poly.vertices[0];
        F(i, 1) = poly.vertices[1];
        F(i, 2) = poly.vertices[2];
    }
    return true;
}

void MeshFuser::eigenToPclMesh(const Eigen::MatrixXd& V,
                               const Eigen::MatrixXi& F,
                               const Eigen::MatrixXd& N,
                               const std::vector<Eigen::Vector3f>& colors,
                               pcl::PolygonMesh& mesh) const {
    pcl::PointCloud<pcl::PointXYZRGBNormal> cloud;
    cloud.resize(V.rows());
    for (int i = 0; i < V.rows(); ++i) {
        auto& pt = cloud.points[i];
        pt.x = static_cast<float>(V(i,0));
        pt.y = static_cast<float>(V(i,1));
        pt.z = static_cast<float>(V(i,2));
        if (N.rows() == V.rows()) {
            pt.normal_x = static_cast<float>(N(i,0));
            pt.normal_y = static_cast<float>(N(i,1));
            pt.normal_z = static_cast<float>(N(i,2));
        } else {
            pt.normal_x = pt.normal_y = pt.normal_z = 0.f;
        }
        if (i < static_cast<int>(colors.size())) {
            Eigen::Vector3f c = colors[i].cwiseMax(0.f).cwiseMin(1.f);
            pt.r = static_cast<uint8_t>(c[0] * 255.f);
            pt.g = static_cast<uint8_t>(c[1] * 255.f);
            pt.b = static_cast<uint8_t>(c[2] * 255.f);
        } else {
            pt.r = pt.g = pt.b = 200;
        }
    }
    pcl::toPCLPointCloud2(cloud, mesh.cloud);
    mesh.polygons.resize(F.rows());
    for (int i = 0; i < F.rows(); ++i) {
        pcl::Vertices v;
        v.vertices = {static_cast<uint32_t>(F(i,0)), static_cast<uint32_t>(F(i,1)), static_cast<uint32_t>(F(i,2))};
        mesh.polygons[i] = v;
    }
}

void MeshFuser::assignColors(const Eigen::MatrixXd& V,
                             const pcl::PolygonMesh& shell,
                             const pcl::PolygonMesh& detail,
                             const typename PointNormalCloudT::ConstPtr& samples,
                             std::vector<Eigen::Vector3f>& colors) const {
    // 构建KD树
    PointCloudT sample_pts;
    pcl::copyPointCloud(*samples, sample_pts);
    pcl::KdTreeFLANN<PointT> sample_tree;
    sample_tree.setInputCloud(sample_pts.makeShared());

    PointCloudT shell_pts;
    pcl::fromPCLPointCloud2(shell.cloud, shell_pts);
    pcl::KdTreeFLANN<PointT> shell_tree;
    if (!shell_pts.empty()) shell_tree.setInputCloud(shell_pts.makeShared());

    PointCloudT detail_pts;
    pcl::fromPCLPointCloud2(detail.cloud, detail_pts);
    pcl::KdTreeFLANN<PointT> detail_tree;
    if (!detail_pts.empty()) detail_tree.setInputCloud(detail_pts.makeShared());

    colors.resize(V.rows(), Eigen::Vector3f(0.5f,0.5f,0.5f));
    for (int i = 0; i < V.rows(); ++i) {
        PointT query;
        query.x = static_cast<float>(V(i,0));
        query.y = static_cast<float>(V(i,1));
        query.z = static_cast<float>(V(i,2));

        double ds = std::numeric_limits<double>::max();
        double dd = std::numeric_limits<double>::max();
        if (!shell_pts.empty()) {
            std::vector<int> idx(1); std::vector<float> sqd(1);
            shell_tree.nearestKSearch(query, 1, idx, sqd);
            ds = std::sqrt(sqd[0]);
        }
        if (!detail_pts.empty()) {
            std::vector<int> idx(1); std::vector<float> sqd(1);
            detail_tree.nearestKSearch(query, 1, idx, sqd);
            dd = std::sqrt(sqd[0]);
        }

        // shell color: average of 8 neighbours
        Eigen::Vector3f shell_color = Eigen::Vector3f::Zero();
        std::vector<int> sidx(8); std::vector<float> sdist(8);
        if (sample_tree.nearestKSearch(query, 8, sidx, sdist) > 0) {
            for (int id : sidx) {
                const auto& pt = sample_pts.points[id];
                shell_color += Eigen::Vector3f(pt.r, pt.g, pt.b);
            }
            shell_color /= (255.f * static_cast<float>(sidx.size()));
        }
        // detail color: nearest neighbour
        Eigen::Vector3f detail_color = shell_color;
        std::vector<int> didx(1); std::vector<float> ddist(1);
        if (sample_tree.nearestKSearch(query, 1, didx, ddist) > 0) {
            const auto& pt = sample_pts.points[didx[0]];
            detail_color = Eigen::Vector3f(pt.r, pt.g, pt.b) / 255.f;
        }

        Eigen::Vector3f final_color;
        if (dd < ds && dd < cfg_.detail_threshold) {
            final_color = detail_color;
        } else if (ds < dd && ds < cfg_.detail_threshold) {
            final_color = shell_color;
        } else if (ds < cfg_.blend_distance && dd < cfg_.blend_distance) {
            final_color = cfg_.shell_color_weight * shell_color +
                          cfg_.detail_color_weight * detail_color;
        } else {
            final_color = shell_color;
        }
        colors[i] = final_color;
    }
}

bool MeshFuser::fuse(const pcl::PolygonMesh& shell,
                     const pcl::PolygonMesh& detail,
                     const typename PointNormalCloudT::ConstPtr& samples,
                     pcl::PolygonMesh& out) {
    Eigen::MatrixXd Vs, Vd;
    Eigen::MatrixXi Fs, Fd;
    if(!pclMeshToEigen(shell, Vs, Fs) || !pclMeshToEigen(detail, Vd, Fd)) {
        std::cerr << "网格转换失败" << std::endl;
        return false;
    }

    // 壳主导δ保护: detail - dilate(shell, δ)
    Eigen::MatrixXd Vd_proc = Vd;
    Eigen::MatrixXi Fd_proc = Fd;
    if (cfg_.shell_priority && cfg_.shell_dilation > 0.0 && Vs.rows() > 0 && Vd.rows() > 0) {
        try {
            Eigen::MatrixXd Vdil; Eigen::MatrixXi Fdil;
            igl::copyleft::cgal::approximate_offset(Vs, Fs, cfg_.shell_dilation, Vdil, Fdil);
            igl::copyleft::cgal::mesh_boolean(Vd, Fd, Vdil, Fdil,
                igl::MESH_BOOLEAN_TYPE_MINUS, Vd_proc, Fd_proc);
        } catch (const std::exception& e) {
            std::cerr << "δ保护失败: " << e.what() << std::endl;
            Vd_proc = Vd; Fd_proc = Fd;
        }
    }

    Eigen::MatrixXd V; Eigen::MatrixXi F;
    bool boolean_ok = true;
    try {
        igl::copyleft::cgal::mesh_boolean(Vs, Fs, Vd_proc, Fd_proc,
            igl::MESH_BOOLEAN_TYPE_UNION, V, F);
    } catch (const std::exception& e) {
        std::cerr << "布尔运算失败: " << e.what() << std::endl;
        boolean_ok = false;
    }

    // Alpha包装回退
    if (!boolean_ok) {
        if (cfg_.enable_alpha_fallback) {
            try {
                Eigen::MatrixXd Vstack(Vs.rows() + Vd_proc.rows(), 3);
                Vstack << Vs, Vd_proc;
                Eigen::MatrixXi Fstack(Fs.rows() + Fd_proc.rows(), 3);
                Fstack << Fs, (Fd_proc.array() + Vs.rows()).matrix();
                igl::copyleft::cgal::alpha_wrap_3D(
                    Vstack, Fstack,
                    cfg_.alpha_offset,
                    cfg_.alpha_offset * 0.5,
                    V, F);
                boolean_ok = true;
            } catch (const std::exception& e) {
                std::cerr << "Alpha包装回退失败: " << e.what() << std::endl;
                return false;
            }
        } else {
            return false;
        }
    }

    // Vertex welding
    Eigen::MatrixXd Vw, Vc; Eigen::MatrixXi Fw, Fc;
    Eigen::VectorXi I, J;
    igl::remove_duplicate_vertices(V, F, cfg_.weld_threshold, Vw, I, Fw, J);
    igl::remove_unreferenced(Vw, Fw, Vc, Fc, I, J);
    V = Vc; F = Fc;

    // Recompute normals
    Eigen::MatrixXd N;
    if (cfg_.recompute_normals) {
        igl::per_vertex_normals(V, F, N);
    }

    std::vector<Eigen::Vector3f> colors;
    assignColors(V, shell, detail, samples, colors);

    eigenToPclMesh(V, F, N, colors, out);
    return true;
}

