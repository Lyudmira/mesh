#include "dual_contouring.h"
#include <openvdb/tools/Gradient.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <queue>
#include <chrono>
#include <algorithm>

namespace recon {

// ---------------- QEFData ----------------
void QEFData::addConstraint(const Eigen::Vector3f& point,
                            const Eigen::Vector3f& normal,
                            float weight) {
    Eigen::Vector3f n = normal * weight;
    A += n * n.transpose();
    b += n * (n.dot(point));
    c += weight * weight * (n.dot(point) * n.dot(point));
    ++num_constraints;
}

Eigen::Vector3f QEFData::solve(float regularization) const {
    Eigen::Matrix3f AtA = A + Eigen::Matrix3f::Identity() * regularization;
    return AtA.ldlt().solve(b);
}

// ---------------- DualContouringExtractor ----------------
DualContouringExtractor::DualContouringExtractor(const DualContouringConfig& config)
    : config_(config) {
    openvdb::initialize();
}

DualContouringExtractor::~DualContouringExtractor() = default;

bool DualContouringExtractor::extractSurface(const UDFGridT::Ptr& udf_grid,
                                             const LabelGridT::Ptr& label_grid,
                                             const PointCloudT::Ptr& cloud,
                                             pcl::PolygonMesh& mesh) {
    if (!udf_grid || !label_grid) return false;
    auto start = std::chrono::high_resolution_clock::now();
    cells_.clear(); vertices_.clear(); normals_.clear(); triangles_.clear();

    // Build signed distance field copy
    UDFGridT::Ptr sdf = udf_grid->deepCopy();
    auto label_acc = label_grid->getConstAccessor();
    for (auto iter = sdf->beginValueOn(); iter; ++iter) {
        openvdb::Coord c = iter.getCoord();
        int lbl = label_acc.getValue(c);
        float val = iter.getValue();
        iter.setValue(lbl == static_cast<int>(Label::INSIDE) ? -val : val);
    }

    buildDualCells(sdf, label_grid, cloud);
    generateTriangles(label_grid);
    buildPCLMesh(mesh);

    auto end = std::chrono::high_resolution_clock::now();
    stats_.extraction_time = std::chrono::duration<double>(end - start).count();
    stats_.vertices_generated = static_cast<int>(vertices_.size());
    stats_.triangles_generated = static_cast<int>(triangles_.size());
    return true;
}

void DualContouringExtractor::buildDualCells(const UDFGridT::Ptr& sdf,
                                             const LabelGridT::Ptr& label_grid,
                                             const PointCloudT::Ptr& cloud) {
    auto sdf_acc = sdf->getConstAccessor();
    auto label_acc = label_grid->getConstAccessor();
    openvdb::CoordBBox bbox;
    label_grid->evalActiveVoxelBoundingBox(bbox);
    pcl::KdTreeFLANN<PointT> tree;
    if (cloud && !cloud->empty()) tree.setInputCloud(cloud);

    static const openvdb::Coord corner_offset[8] = {
        {0,0,0},{1,0,0},{0,1,0},{1,1,0},{0,0,1},{1,0,1},{0,1,1},{1,1,1}
    };
    static const int edge_corners[12][2] = {
        {0,1},{1,3},{3,2},{2,0},
        {4,5},{5,7},{7,6},{6,4},
        {0,4},{1,5},{2,6},{3,7}
    };

    for (int z = bbox.min().z(); z < bbox.max().z(); ++z) {
        for (int y = bbox.min().y(); y < bbox.max().y(); ++y) {
            for (int x = bbox.min().x(); x < bbox.max().x(); ++x) {
                openvdb::Coord base(x,y,z);
                float sdf_vals[8];
                bool signs[8];
                bool has_pos=false, has_neg=false;
                for (int i=0;i<8;++i){
                    openvdb::Coord c = base + corner_offset[i];
                    float v = sdf_acc.getValue(c);
                    sdf_vals[i]=v;
                    signs[i]=v>=0;
                    if (signs[i]) has_pos=true; else has_neg=true;
                }
                if (!(has_pos && has_neg)) continue; // not intersected
                DualCell cell; cell.coord=base; cell.has_vertex=false; cell.vertex_index=-1;
                for(int e=0;e<12;++e){
                    int a=edge_corners[e][0]; int b=edge_corners[e][1];
                    if (signs[a]==signs[b]) continue;
                    float t=computeIntersectionParameter(sdf_vals[a], sdf_vals[b]);
                    openvdb::Vec3f p1 = sdf->transform().indexToWorld(base+corner_offset[a]);
                    openvdb::Vec3f p2 = sdf->transform().indexToWorld(base+corner_offset[b]);
                    openvdb::Vec3f pos = p1 + (p2-p1)*t;
                    openvdb::Vec3f n = estimateNormal(pos, sdf, cloud);
                    cell.qef.addConstraint(Eigen::Vector3f(pos.x(),pos.y(),pos.z()),
                                           Eigen::Vector3f(n.x(),n.y(),n.z()));
                }
                cell.vertex_position = solveQEF(cell);
                cell.has_vertex = true;
                cell.vertex_index = static_cast<int>(vertices_.size());
                vertices_.push_back(cell.vertex_position);
                cells_[base] = cell;
            }
        }
    }
    stats_.active_cells = static_cast<int>(cells_.size());
}

openvdb::Vec3f DualContouringExtractor::solveQEF(const DualCell& cell) {
    if (cell.qef.num_constraints == 0) {
        openvdb::Vec3f center = cell.coord.asVec3d();
        center += openvdb::Vec3d(0.5,0.5,0.5);
        return openvdb::Vec3f(center);
    }
    Eigen::Vector3f v = cell.qef.solve(config_.qef_regularization);
    return openvdb::Vec3f(v.x(), v.y(), v.z());
}

openvdb::Vec3f DualContouringExtractor::estimateNormal(const openvdb::Vec3f& position,
                                                       const UDFGridT::Ptr& sdf,
                                                       const PointCloudT::Ptr& cloud) {
    openvdb::Vec3f n = computeGradientNormal(position, sdf);
    if (config_.use_adaptive_normals && cloud && !cloud->empty()) {
        openvdb::Vec3f nc = estimateNormalFromCloud(position, cloud);
        if (nc.lengthSqr() > 1e-6f) n = nc;
    }
    if (n.lengthSqr() > 0) n.normalize();
    return n;
}

openvdb::Vec3f DualContouringExtractor::computeGradientNormal(const openvdb::Vec3f& position,
                                                              const UDFGridT::Ptr& sdf) {
    openvdb::Vec3d idx = sdf->transform().worldToIndex(position);
    openvdb::Coord c = openvdb::Coord::floor(idx);
    auto acc = sdf->getConstAccessor();
    float dx1 = acc.getValue(c.offsetBy(1,0,0));
    float dx0 = acc.getValue(c.offsetBy(-1,0,0));
    float dy1 = acc.getValue(c.offsetBy(0,1,0));
    float dy0 = acc.getValue(c.offsetBy(0,-1,0));
    float dz1 = acc.getValue(c.offsetBy(0,0,1));
    float dz0 = acc.getValue(c.offsetBy(0,0,-1));
    return openvdb::Vec3f((dx1-dx0)/2.0f, (dy1-dy0)/2.0f, (dz1-dz0)/2.0f);
}

openvdb::Vec3f DualContouringExtractor::estimateNormalFromCloud(const openvdb::Vec3f& position,
                                                                const PointCloudT::Ptr& cloud) {
    pcl::KdTreeFLANN<PointT> tree;
    tree.setInputCloud(cloud);
    PointT q; q.x=position.x(); q.y=position.y(); q.z=position.z();
    std::vector<int> idx(8); std::vector<float> dist2(8);
    openvdb::Vec3f n(0.0f);
    if (tree.nearestKSearch(q,8,idx,dist2)>0){
        for(int i:idx){
            const auto& p = cloud->points[i];
            n += openvdb::Vec3f(p.normal_x,p.normal_y,p.normal_z);
        }
    }
    return n;
}

float DualContouringExtractor::computeIntersectionParameter(float v1, float v2){
    return v1 / (v1 - v2);
}

void DualContouringExtractor::generateTriangles(const LabelGridT::Ptr& label_grid){
    static const openvdb::Coord neighbors[3] = { {1,0,0},{0,1,0},{0,0,1} };
    for (const auto& kv : cells_) {
        const openvdb::Coord& c = kv.first;
        int v0 = kv.second.vertex_index;
        // XY face
        auto it1 = cells_.find(c + neighbors[0]);
        auto it2 = cells_.find(c + neighbors[1]);
        auto it3 = cells_.find(c + neighbors[0] + neighbors[1]);
        if (it1!=cells_.end() && it2!=cells_.end() && it3!=cells_.end()) {
            triangles_.push_back({v0, it1->second.vertex_index, it3->second.vertex_index});
            triangles_.push_back({v0, it3->second.vertex_index, it2->second.vertex_index});
        }
        // XZ face
        it1 = cells_.find(c + neighbors[0]);
        it2 = cells_.find(c + neighbors[2]);
        it3 = cells_.find(c + neighbors[0] + neighbors[2]);
        if (it1!=cells_.end() && it2!=cells_.end() && it3!=cells_.end()) {
            triangles_.push_back({v0, it1->second.vertex_index, it3->second.vertex_index});
            triangles_.push_back({v0, it3->second.vertex_index, it2->second.vertex_index});
        }
        // YZ face
        it1 = cells_.find(c + neighbors[1]);
        it2 = cells_.find(c + neighbors[2]);
        it3 = cells_.find(c + neighbors[1] + neighbors[2]);
        if (it1!=cells_.end() && it2!=cells_.end() && it3!=cells_.end()) {
            triangles_.push_back({v0, it1->second.vertex_index, it3->second.vertex_index});
            triangles_.push_back({v0, it3->second.vertex_index, it2->second.vertex_index});
        }
    }
}

void DualContouringExtractor::buildPCLMesh(pcl::PolygonMesh& mesh){
    pcl::PointCloud<pcl::PointXYZ> cloud;
    cloud.reserve(vertices_.size());
    for(const auto& v: vertices_) cloud.emplace_back(v.x(),v.y(),v.z());
    pcl::toPCLPointCloud2(cloud, mesh.cloud);
    mesh.polygons.clear();
    mesh.polygons.reserve(triangles_.size());
    for(const auto& t: triangles_){
        pcl::Vertices v; v.vertices={t[0],t[1],t[2]};
        mesh.polygons.push_back(v);
    }
}

} // namespace recon

