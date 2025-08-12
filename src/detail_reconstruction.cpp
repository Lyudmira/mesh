#include "detail_reconstruction.h"

#include <pcl/filters/fast_bilateral.h>
#include <pcl/surface/mls.h>
#include <pcl/surface/poisson.h>
// WLOP算法由CGAL提供，不是PCL
// #include <pcl/surface/wlop.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/common/geometry.h>
#include <pcl/common/centroid.h>
#include <pcl/console/time.h>

// libigl C++头文件在conda环境中不可用，使用Python绑定
// #include <igl/signed_distance.h>

#include <map>
#include <set>
#include <queue>
#include <cmath>

#include <Eigen/Dense>
#include <chrono>

#include <CGAL/Simple_cartesian.h>
#include <CGAL/Surface_mesh.h>
// 某些CGAL头文件在当前版本中不可用，暂时注释
// #include <CGAL/Polygon_mesh_processing/repair_self_intersections.h>
// #include <CGAL/Polygon_mesh_processing/remove_degenerate_faces.h>

namespace recon {

using Kernel = CGAL::Simple_cartesian<double>;
using SurfaceMesh = CGAL::Surface_mesh<Kernel::Point_3>;

DetailReconstructor::DetailReconstructor(const DetailReconstructionConfig& config)
    : config_(config) {}

DetailReconstructor::~DetailReconstructor() = default;

bool DetailReconstructor::reconstructDetails(const pcl::PolygonMesh& shell_mesh,
                                             const PointCloudT::Ptr& original_cloud,
                                             pcl::PolygonMesh& detail_mesh) {
    stats_ = Statistics{};
    if (!original_cloud) return false;
    stats_.original_points = static_cast<int>(original_cloud->size());

    PointCloudT::Ptr band_points(new PointCloudT);
    if (!extractOffsetBandPoints(shell_mesh, original_cloud, band_points) ||
        band_points->empty()) {
        std::cerr << "细节偏移带点提取失败或为空" << std::endl;
        return false;
    }
    stats_.extracted_points = static_cast<int>(band_points->size());

    PointCloudT::Ptr filtered(new PointCloudT);
    if (config_.enable_denoising) {
        if (!denoisePoints(band_points, filtered)) {
            std::cerr << "去噪失败" << std::endl;
            return false;
        }
        stats_.denoised_points = static_cast<int>(filtered->size());
    } else {
        filtered = band_points;
        stats_.denoised_points = static_cast<int>(filtered->size());
    }

    computeAdaptiveParameters(filtered);

    bool ok = false;
    auto start = std::chrono::high_resolution_clock::now();
    switch (config_.primary_method) {
        case DetailMethod::GP3:
            ok = reconstructWithGP3(filtered, detail_mesh);
            break;
        case DetailMethod::RIMLS:
            ok = reconstructWithRIMLs(filtered, detail_mesh);
            break;
        case DetailMethod::POISSON:
            ok = reconstructWithPoisson(filtered, detail_mesh);
            break;
    }
    if (!ok) return false;
    auto end = std::chrono::high_resolution_clock::now();
    stats_.reconstruction_time =
        std::chrono::duration<double>(end - start).count();
    stats_.output_vertices = static_cast<int>(detail_mesh.cloud.width);
    stats_.output_triangles = static_cast<int>(detail_mesh.polygons.size());

    start = std::chrono::high_resolution_clock::now();
    postProcessMesh(detail_mesh);
    end = std::chrono::high_resolution_clock::now();
    stats_.post_processing_time =
        std::chrono::duration<double>(end - start).count();

    return true;
}

bool DetailReconstructor::extractOffsetBandPoints(const pcl::PolygonMesh& shell_mesh,
                                                  const PointCloudT::Ptr& original_cloud,
                                                  PointCloudT::Ptr& extracted) {
    if (!original_cloud || !extracted) return false;

    pcl::PointCloud<pcl::PointXYZ> shell_pts;
    pcl::fromPCLPointCloud2(shell_mesh.cloud, shell_pts);
    Eigen::MatrixXd V(shell_pts.size(), 3);
    for (size_t i = 0; i < shell_pts.size(); ++i)
        V.row(i) << shell_pts[i].x, shell_pts[i].y, shell_pts[i].z;

    Eigen::MatrixXi F(shell_mesh.polygons.size(), 3);
    for (size_t i = 0; i < shell_mesh.polygons.size(); ++i) {
        const auto& poly = shell_mesh.polygons[i];
        if (poly.vertices.size() != 3) return false;
        F.row(i) << poly.vertices[0], poly.vertices[1], poly.vertices[2];
    }

    Eigen::MatrixXd P(original_cloud->size(), 3);
    for (size_t i = 0; i < original_cloud->size(); ++i)
        P.row(i) << (*original_cloud)[i].x, (*original_cloud)[i].y,
                     (*original_cloud)[i].z;

    Eigen::VectorXd S;
    igl::signed_distance(P, V, F, igl::SIGNED_DISTANCE_TYPE_WINDING_NUMBER, S);

    extracted->clear();
    for (size_t i = 0; i < original_cloud->size(); ++i) {
        double d = S[i];
        if (std::abs(d) > config_.offset_distance) continue;
        if (config_.outside_only && d <= 0) continue;
        extracted->push_back((*original_cloud)[i]);
    }
    return true;
}

bool DetailReconstructor::denoisePoints(const PointCloudT::Ptr& input,
                                        PointCloudT::Ptr& output) {
    if (config_.denoising_method == "wlop") {
        return wlopDenoising(input, output);
    }
    if (config_.denoising_method == "bilateral") {
        return bilateralFilter(input, output);
    }
    // 默认使用MLS作为轻度平滑
    pcl::MovingLeastSquares<PointT, PointT> mls;
    mls.setInputCloud(input);
    mls.setSearchRadius(config_.radius_multiplier * 0.03f);
    mls.setPolynomialFit(true);
    mls.setComputeNormals(true);
    mls.process(*output);
    return true;
}

bool DetailReconstructor::reconstructWithGP3(const PointCloudT::Ptr& points,
                                             pcl::PolygonMesh& mesh) {
    pcl::PointCloud<pcl::PointNormal>::Ptr normals(new pcl::PointCloud<pcl::PointNormal>);
    pcl::copyPointCloud(*points, *normals);

    pcl::GreedyProjectionTriangulation<pcl::PointNormal> gp3;
    float search_radius = config_.gp3.edge_length_multiplier * base_radius_;
    gp3.setSearchRadius(search_radius);
    gp3.setMu(config_.gp3.mu);
    gp3.setMaximumNearestNeighbors(config_.gp3.max_nearest_neighbors);
    gp3.setMaximumAngle(config_.gp3.angle_threshold_planar * M_PI / 180.0f);
    gp3.setMinimumAngle(config_.gp3.min_angle * M_PI / 180.0f);
    gp3.setMaximumAngle(config_.gp3.max_angle * M_PI / 180.0f);
    gp3.setNormalConsistency(config_.gp3.normal_consistency);

    std::vector<pcl::Vertices> triangles;
    gp3.reconstruct(*normals, triangles);

    pcl::toPCLPointCloud2(*points, mesh.cloud);
    mesh.polygons = triangles;
    return !mesh.polygons.empty();
}

bool DetailReconstructor::reconstructWithRIMLs(const PointCloudT::Ptr& points,
                                               pcl::PolygonMesh& mesh) {
    pcl::MovingLeastSquares<PointT, PointT> mls;
    mls.setInputCloud(points);
    mls.setSearchRadius(config_.rimls.voxel_size * 3.0f);
    mls.setPolynomialFit(true);
    mls.setComputeNormals(true);
    PointCloudT::Ptr smooth(new PointCloudT);
    mls.process(*smooth);

    pcl::MarchingCubesHoppe<PointT> mc;
    mc.setInputCloud(smooth);
    Eigen::Vector4f min_pt, max_pt;
    pcl::getMinMax3D(*smooth, min_pt, max_pt);
    Eigen::Vector3i res;
    res[0] = res[1] = res[2] = static_cast<int>((max_pt - min_pt).maxCoeff() /
                                                config_.rimls.voxel_size);
    mc.setGridResolution(res[0], res[1], res[2]);
    mc.reconstruct(mesh);
    return !mesh.polygons.empty();
}

bool DetailReconstructor::reconstructWithPoisson(const PointCloudT::Ptr& points,
                                                 pcl::PolygonMesh& mesh) {
    pcl::PointCloud<pcl::PointNormal>::Ptr normals(new pcl::PointCloud<pcl::PointNormal>);
    pcl::copyPointCloud(*points, *normals);

    pcl::Poisson<pcl::PointNormal> poisson;
    poisson.setDepth(8);
    poisson.setInputCloud(normals);
    poisson.reconstruct(mesh);
    return !mesh.polygons.empty();
}

bool DetailReconstructor::postProcessMesh(pcl::PolygonMesh& mesh) {
    if (config_.post_processing.remove_hanging_edges) {
        removeHangingEdges(mesh);
    }
    if (config_.post_processing.filter_small_components) {
        if (filterSmallComponents(mesh)) stats_.removed_components++;
    }
    if (config_.post_processing.enable_pmp_repair) {
        repairWithPMP(mesh);
    }
    if (config_.post_processing.enable_simplification) {
        simplifyMesh(mesh);
    }
    performQualityControl(mesh);
    return true;
}

bool DetailReconstructor::removeHangingEdges(pcl::PolygonMesh& mesh) {
    std::map<std::pair<uint32_t, uint32_t>, int> edge_count;
    for (const auto& poly : mesh.polygons) {
        if (poly.vertices.size() != 3) continue;
        for (int e = 0; e < 3; ++e) {
            uint32_t a = poly.vertices[e];
            uint32_t b = poly.vertices[(e + 1) % 3];
            if (a > b) std::swap(a, b);
            edge_count[{a, b}]++;
        }
    }

    std::vector<pcl::Vertices> cleaned;
    cleaned.reserve(mesh.polygons.size());
    for (const auto& poly : mesh.polygons) {
        if (poly.vertices.size() != 3) continue;
        bool keep = true;
        for (int e = 0; e < 3 && keep; ++e) {
            uint32_t a = poly.vertices[e];
            uint32_t b = poly.vertices[(e + 1) % 3];
            if (a > b) std::swap(a, b);
            if (edge_count[{a, b}] < 2) keep = false;
        }
        if (keep) cleaned.push_back(poly);
    }
    mesh.polygons.swap(cleaned);
    return true;
}

bool DetailReconstructor::filterSmallComponents(pcl::PolygonMesh& mesh) {
    const int min_size = config_.post_processing.min_component_size;
    if (min_size <= 0) return false;

    // Build adjacency
    std::vector<std::vector<int>> adj(mesh.polygons.size());
    for (size_t i = 0; i < mesh.polygons.size(); ++i) {
        const auto& poly_i = mesh.polygons[i];
        std::set<uint32_t> verts(poly_i.vertices.begin(), poly_i.vertices.end());
        for (size_t j = i + 1; j < mesh.polygons.size(); ++j) {
            const auto& poly_j = mesh.polygons[j];
            int shared = 0;
            for (auto v : poly_j.vertices) if (verts.count(v)) shared++;
            if (shared >= 2) {
                adj[i].push_back(j);
                adj[j].push_back(i);
            }
        }
    }

    std::vector<int> comp(mesh.polygons.size(), -1);
    int cid = 0;
    for (size_t i = 0; i < mesh.polygons.size(); ++i) {
        if (comp[i] != -1) continue;
        std::queue<int> q;
        q.push(i);
        comp[i] = cid;
        int count = 0;
        std::vector<int> members;
        while (!q.empty()) {
            int v = q.front(); q.pop();
            members.push_back(v);
            count++;
            for (int nb : adj[v]) if (comp[nb] == -1) { comp[nb] = cid; q.push(nb); }
        }
        if (count < min_size) {
            for (int idx : members) comp[idx] = -2; // mark for removal
        }
        cid++;
    }

    std::vector<pcl::Vertices> filtered_polys;
    for (size_t i = 0; i < mesh.polygons.size(); ++i) {
        if (comp[i] >= 0) filtered_polys.push_back(mesh.polygons[i]);
    }
    bool removed = filtered_polys.size() != mesh.polygons.size();
    mesh.polygons.swap(filtered_polys);
    return removed;
}

bool DetailReconstructor::repairWithPMP(pcl::PolygonMesh& mesh) {
    namespace PMP = CGAL::Polygon_mesh_processing;
    pcl::PointCloud<pcl::PointXYZ> cloud;
    pcl::fromPCLPointCloud2(mesh.cloud, cloud);

    SurfaceMesh sm;
    std::vector<SurfaceMesh::Vertex_index> vmap(cloud.size());
    for (size_t i = 0; i < cloud.size(); ++i) {
        vmap[i] = sm.add_vertex(Kernel::Point_3(cloud[i].x, cloud[i].y, cloud[i].z));
    }
    for (const auto& poly : mesh.polygons) {
        if (poly.vertices.size() != 3) continue;
        sm.add_face(vmap[poly.vertices[0]], vmap[poly.vertices[1]],
                    vmap[poly.vertices[2]]);
    }

    if (config_.post_processing.fix_self_intersections)
        PMP::remove_self_intersections(sm);
    if (config_.post_processing.fix_non_manifold)
        PMP::remove_degenerate_faces(sm);

    pcl::PointCloud<pcl::PointXYZ> out_cloud;
    pcl::PolygonMesh out_mesh;
    for (auto v : sm.vertices()) {
        const auto& p = sm.point(v);
        out_cloud.emplace_back(p.x(), p.y(), p.z());
    }
    pcl::toPCLPointCloud2(out_cloud, out_mesh.cloud);
    for (auto f : sm.faces()) {
        pcl::Vertices tri;
        for (auto v : CGAL::vertices_around_face(sm.halfedge(f), sm))
            tri.vertices.push_back(static_cast<uint32_t>(v));
        out_mesh.polygons.push_back(tri);
    }
    mesh.swap(out_mesh);
    return true;
}

bool DetailReconstructor::simplifyMesh(pcl::PolygonMesh& mesh) {
    // Simple voxel-grid based simplification
    pcl::PointCloud<pcl::PointXYZ> cloud;
    pcl::fromPCLPointCloud2(mesh.cloud, cloud);
    pcl::VoxelGrid<pcl::PointXYZ> vg;
    vg.setLeafSize(config_.rimls.voxel_size, config_.rimls.voxel_size,
                   config_.rimls.voxel_size);
    vg.setInputCloud(cloud.makeShared());
    pcl::PointCloud<pcl::PointXYZ> ds;
    vg.filter(ds);
    pcl::toPCLPointCloud2(ds, mesh.cloud);
    mesh.polygons.clear();
    return true;
}

bool DetailReconstructor::performQualityControl(const pcl::PolygonMesh& mesh) {
    // Basic check: ensure edges are not too long
    pcl::PointCloud<pcl::PointXYZ> cloud;
    pcl::fromPCLPointCloud2(mesh.cloud, cloud);
    for (const auto& poly : mesh.polygons) {
        if (poly.vertices.size() != 3) continue;
        const auto& a = cloud[poly.vertices[0]];
        const auto& b = cloud[poly.vertices[1]];
        const auto& c = cloud[poly.vertices[2]];
        double ab = pcl::geometry::distance(a, b);
        double bc = pcl::geometry::distance(b, c);
        double ca = pcl::geometry::distance(c, a);
        if (ab > config_.quality_control.max_edge_length ||
            bc > config_.quality_control.max_edge_length ||
            ca > config_.quality_control.max_edge_length)
            return false;
    }
    return true;
}

float DetailReconstructor::computePointToMeshDistance(const PointT& point,
                                                      const pcl::PolygonMesh& mesh) {
    pcl::PointXYZ p(point.x, point.y, point.z);
    pcl::PointCloud<pcl::PointXYZ> cloud;
    pcl::fromPCLPointCloud2(mesh.cloud, cloud);
    double min_d = std::numeric_limits<double>::max();
    for (const auto& poly : mesh.polygons) {
        if (poly.vertices.size() != 3) continue;
        const auto& a = cloud[poly.vertices[0]];
        const auto& b = cloud[poly.vertices[1]];
        const auto& c = cloud[poly.vertices[2]];
        Eigen::Vector3d pa(p.x - a.x, p.y - a.y, p.z - a.z);
        Eigen::Vector3d pb(p.x - b.x, p.y - b.y, p.z - b.z);
        Eigen::Vector3d pc(p.x - c.x, p.y - c.y, p.z - c.z);
        Eigen::Vector3d ab(b.x - a.x, b.y - a.y, b.z - a.z);
        Eigen::Vector3d ac(c.x - a.x, c.y - a.y, c.z - a.z);
        Eigen::Vector3d n = ab.cross(ac);
        double area2 = n.norm();
        if (area2 < 1e-8) continue;
        double dist = std::abs(pa.dot(n.normalized()));
        if (dist < min_d) min_d = dist;
    }
    return static_cast<float>(min_d);
}

bool DetailReconstructor::isPointOutsideShell(const PointT& point,
                                              const pcl::PolygonMesh& shell_mesh) {
    pcl::PointCloud<pcl::PointXYZ> shell_pts;
    pcl::fromPCLPointCloud2(shell_mesh.cloud, shell_pts);
    Eigen::MatrixXd V(shell_pts.size(), 3);
    for (size_t i = 0; i < shell_pts.size(); ++i)
        V.row(i) << shell_pts[i].x, shell_pts[i].y, shell_pts[i].z;
    Eigen::MatrixXi F(shell_mesh.polygons.size(), 3);
    for (size_t i = 0; i < shell_mesh.polygons.size(); ++i) {
        const auto& poly = shell_mesh.polygons[i];
        if (poly.vertices.size() != 3) return false;
        F.row(i) << poly.vertices[0], poly.vertices[1], poly.vertices[2];
    }
    Eigen::RowVector3d P(point.x, point.y, point.z);
    double s;
    igl::signed_distance(P, V, F, igl::SIGNED_DISTANCE_TYPE_WINDING_NUMBER, s);
    return s > 0;
}

bool DetailReconstructor::bilateralFilter(const PointCloudT::Ptr& input,
                                         PointCloudT::Ptr& output) {
    pcl::FastBilateralFilter<PointT> fb;
    fb.setSigmaS(config_.radius_multiplier * 0.03f);
    fb.setSigmaR(0.02f);
    fb.setInputCloud(input);
    fb.applyFilter(*output);
    return true;
}

bool DetailReconstructor::wlopDenoising(const PointCloudT::Ptr& input,
                                        PointCloudT::Ptr& output) {
    pcl::WLOP<PointT, PointT> wlop;
    wlop.setInputCloud(input);
    wlop.setSearchRadius(config_.radius_multiplier * 0.05f);
    wlop.setDensityWeight(0.5f);
    wlop.process(*output);
    return true;
}

void DetailReconstructor::computeAdaptiveParameters(const PointCloudT::Ptr& points) {
    auto density = computeLocalDensity(points);
    if (density.empty()) return;
    std::vector<float> tmp = density;
    std::nth_element(tmp.begin(), tmp.begin() + tmp.size() / 2, tmp.end());
    base_radius_ = tmp[tmp.size() / 2];
}

std::vector<bool> DetailReconstructor::detectPlanarRegions(const PointCloudT::Ptr& points) {
    const int k = 16;
    pcl::search::KdTree<PointT> tree;
    tree.setInputCloud(points);
    std::vector<bool> planar(points->size(), false);
    std::vector<int> idx(k);
    std::vector<float> dist2(k);
    for (size_t i = 0; i < points->size(); ++i) {
        tree.nearestKSearch(i, k, idx, dist2);
        Eigen::Matrix3f cov = Eigen::Matrix3f::Zero();
        Eigen::Vector3f mean = Eigen::Vector3f::Zero();
        for (int id : idx) {
            const auto& p = (*points)[id];
            Eigen::Vector3f v(p.x, p.y, p.z);
            mean += v;
        }
        mean /= static_cast<float>(k);
        for (int id : idx) {
            Eigen::Vector3f v((*points)[id].x, (*points)[id].y, (*points)[id].z);
            v -= mean;
            cov += v * v.transpose();
        }
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> es(cov);
        auto vals = es.eigenvalues();
        float curvature = vals[0] / (vals.sum() + 1e-6f);
        planar[i] = curvature < 0.01f;
    }
    return planar;
}

std::vector<float> DetailReconstructor::computeLocalDensity(const PointCloudT::Ptr& points) {
    const int k = 16;
    pcl::search::KdTree<PointT> tree;
    tree.setInputCloud(points);
    std::vector<float> density(points->size(), 0.0f);
    std::vector<int> idx(k);
    std::vector<float> dist2(k);
    for (size_t i = 0; i < points->size(); ++i) {
        if (tree.nearestKSearch(i, k, idx, dist2) > 0) {
            density[i] = std::sqrt(dist2.back());
        }
    }
    return density;
}

} // namespace recon

